"""Read an RF-Track element's state, and rebuild the element from it,
for `lume_rftrack.archive`

RF-Track elements can't be pickled and have no save/load API, so an
element is read generically, as RF-Track itself exposes it: every
`get_X()`/`set_X()` pair whose getter takes no arguments (`attributes()`),
restored by building the class with no arguments and calling each
`set_X()`. A class whose state isn't all settable that way (a field map's
data, a pillbox's cell count, ...) also has a `_constructor_args_<class>()`,
giving the arguments to build it with, in order. Everything RF-Track
doesn't report directly is recovered here: field maps without an array
getter are sampled, a setter without a getter (an SBend's fint) is
measured, and RF-Track version differences (2.7.4's smoothed 1D field
maps and separate Volume placement) are handled. `rebuild_error()`
rebuilds an element and compares its field with the original's, so
anything missed is caught instead of archived wrong.

`to_dict()`'s result is plain Python and numpy values, which
`lume_rftrack.archive` writes as HDF5; `from_dict()` takes the same dict
back (or `read_archive()`'s) and builds the element.
"""

import contextlib

import numpy
import pykern.pkinspect
import scipy.linalg
from pykern.pkcollections import PKDict

# Element accessors archived by name rather than generically: `name` and
# `t0` (only when `t0_is_set()`), the aperture (whose setter is
# `set_aperture(x, y, shape)`), and `offsets` and `position` (RF-Track
# 2.8.0+'s `set_position()`; the Lattice's `elements_offsets` or the Volume
# position instead)
_NOT_GENERIC = frozenset(
    ("aperture_shape", "aperture_x", "aperture_y", "name", "offsets", "position", "t0")
)

# Class -> the class whose constructor functions it shares (same
# constructor and getters; `_CINT` is cubic rather than linear
# interpolation)
_SAME_CONSTRUCTOR = PKDict(
    RF_FieldMap_1d_CINT="RF_FieldMap_1d",
    RF_FieldMap_2d_CINT="RF_FieldMap_2d",
    Static_Magnetic_FieldMap_1d_CINT="Static_Magnetic_FieldMap_1d",
    Static_Magnetic_FieldMap_2d_CINT="Static_Magnetic_FieldMap_2d",
)

# Class -> attributes set again, in this order, after all the others:
# an SBend's angle, field, curvature and focusing are interdependent, so
# setting them alphabetically leaves the wrong angle
_SET_LAST = PKDict(
    SBend=("length", "P_over_Q", "angle", "E1", "E2", "hgap", "K1"),
)


def attributes(e):
    """Every `get_X()`/`set_X()` pair of `e` whose getter takes no
    arguments (not, e.g., `Quadrupole.get_K1(P_over_Q)`), except
    `_NOT_GENERIC`'s, as `{X: get_X()}`."""
    res = PKDict()
    for m in sorted(dir(e)):
        a = m[4:]
        if not m.startswith("get_") or a in _NOT_GENERIC or not hasattr(e, "set_" + a):
            continue
        try:
            res[a] = getattr(e, m)()
        except TypeError:
            continue
    return res


def constructor_class(class_name):
    """The class whose constructor arguments `class_name` archives with,
    e.g. "RF_FieldMap_1d" for "RF_FieldMap_1d_CINT", which takes the same
    arguments and differs only in interpolating cubically; otherwise
    `class_name` itself.
    """
    return _SAME_CONSTRUCTOR.get(class_name, class_name)


def from_dict(rft, d):
    """The element `d` (from `to_dict()` or
    `lume_rftrack.archive.read_archive()`)
    describes, built by `_construct()` and set from its `attributes`."""
    e = _construct(rft, d)
    for k in sorted(d.attributes) + list(_SET_LAST.get(d["class"], ())):
        _set_attribute(e, d["class"], k, d.attributes[k])
    e.set_name(str(d.name))
    if d.aperture_set:
        e.set_aperture(float(d.aperture_x), float(d.aperture_y), str(d.aperture_shape))
    if "t0" in d:
        e.set_t0(float(d.t0))
    return e


def rebuild_error(rft, e, d):
    """Rebuild `e` from `d` (its `to_dict()`) and compare its
    field, on and off axis, from before its entrance to past its exit --
    what tracking actually sees, and what catches constructor state no
    getter exposes (e.g. a solenoid's radius).

    Returns:
        float: 0.0 if the fields agree, else the largest difference over
            the largest field

    Raises:
        NotImplementedError: it can't be rebuilt at all
    """
    n = from_dict(rft, d)
    z = numpy.linspace(-0.1, 1.1, 25) * max(e.get_length(), 0.01) * 1000.0
    a, b = [], []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (-2.0, 1.5)):
        a.append(numpy.hstack(e.get_field(x, y, z, 0.0)))
        b.append(numpy.hstack(n.get_field(x, y, z, 0.0)))
    a = numpy.array(a)
    b = numpy.array(b)
    m = numpy.max(numpy.abs(a))
    # atol: a 1D field map's off-axis field comes from its samples' z
    # derivatives, which amplify _unsmoothed()'s round-off (RF-Track 2.7.4+;
    # up to ~2e-10 of the peak for a 3001-sample gun map)
    if numpy.allclose(b, a, rtol=1e-9, atol=1e-9 * m):
        return 0.0
    # no field at all in the original: the rebuilt element's own, absolute
    return float(numpy.max(numpy.abs(b - a)) / (m or 1.0))


def to_dict(rft, e, beamline_class="Lattice"):
    """`e`'s state, as plain Python and numpy values: `class`, `name`,
    `attributes` (its storable `attributes()`, plus an SBend's `fint`),
    `unarchived_attributes` (the names of those HDF5 can't hold, if any),
    `constructor` and `constructor_args` (if its class needs them), the
    aperture, `t0` if set and, if `beamline_class` is "Volume", its
    position `x`/`y`/`z`.

    Raises:
        NotImplementedError: an element whose state can't be read (e.g. a
            Solenoid with a thick winding, or a rotated or misaligned
            Volume element)
    """
    cls = type(e).__name__
    a = attributes(e)
    d = PKDict(
        name=e.get_name(),
        attributes=PKDict((k, v) for k, v in a.items() if _is_storable(v)),
    )
    if len(d.attributes) < len(a):
        d.unarchived_attributes = sorted(set(a) - set(d.attributes))
    if isinstance(e, rft.SBend):
        d.attributes.fint = _fint(rft, e)
    d["class"] = cls
    f = _constructor_function(cls)
    if f is not None:
        d.constructor = f(e)
        # in the constructor's argument order, which HDF5 doesn't keep
        d.constructor_args = list(d.constructor)
    d.aperture_x = e.get_aperture_x()
    d.aperture_y = e.get_aperture_y()
    d.aperture_shape = e.get_aperture_shape()
    d.aperture_set = _aperture_is_set(rft, e, d)
    if beamline_class == "Volume":
        d.update(_position_to_attrs(e))
    # Only RF fields and Screen have t0_is_set()/get_t0() at all.
    # get_phid() alone never reflects autophase(): it always reads back
    # the phid the element was constructed with. t0 is the actual
    # correction autophase() applies and .track() uses as-is, so archiving
    # it directly lets load_archive() reproduce the autophased physics
    # exactly (a fresh beamline with these set_t0() values tracks to the same
    # mean kinetic energy), whether or not the caller re-runs autophase().
    if hasattr(e, "t0_is_set") and e.t0_is_set():
        d.t0 = e.get_t0()
    return d


def _aperture_is_set(rft, e, d):
    """Whether `set_aperture()` was ever called on `e`.

    RF-Track has no getter for that, so this compares with the default
    of the same class, built the same way (a `Pillbox_Cavity`'s default
    depends on its geometry).
    """
    r = _construct(rft, d)
    return (e.get_aperture_x(), e.get_aperture_y()) != (
        r.get_aperture_x(),
        r.get_aperture_y(),
    )


def _construct(rft, d):
    """`d`'s class, built from its `constructor` arguments if it has
    them, else with none."""
    try:
        return getattr(rft, d["class"])(
            *(d.constructor[k] for k in d.constructor_args)
            if "constructor" in d
            else ()
        )
    except (AttributeError, TypeError) as e:
        # TypeError: required arguments; AttributeError: "No constructor
        # defined" (an abstract class)
        raise NotImplementedError(
            f"can't build a {d['class']!r} element; archiving it needs a"
            + f" _constructor_args_{constructor_class(d['class']).lower()}()"
        ) from e


def _constructor_args_coil(e):
    # the constructor gives a coil's field outside its length; built
    # without arguments and set, it has none there
    return _getters(e, "length", "B0", "R")


def _constructor_args_pillbox_cavity(e):
    # the number of cells has no getter or setter
    return PKDict(
        _getters(e, "coefficients", "frequency", "cell_length"),
        n_cells=e.get_length() / e.get_cell_length(),
    )


def _constructor_args_rf_fieldmap_1d(e):
    # get_Ez() is the samples the element tracks with, which RF-Track
    # 2.7.4+ smooths when it's built (see _unsmoothed())
    rv = _getters(e, "Ez", "hz", "length", "frequency", "direction")
    rv.Ez = _unsmoothed(
        numpy.asarray(rv.Ez),
        lambda x: numpy.asarray(
            type(e)(x, rv.hz, rv.length, rv.frequency, rv.direction).get_Ez()
        ),
    )
    return rv


def _constructor_args_rf_fieldmap_2d(e):
    # Unlike RF_FieldMap_1d's get_Ez(), there's no direct 2D array getter,
    # so this samples get_field_complex(x, y, z, t) point by point over
    # the full (nz, nr) grid -- cylindrical symmetry means Er is just the
    # Cartesian Ex component at (x=r, y=0) (sampling at (x=0, y=r) gives the
    # same value in Ey, not Ex). The
    # magnetic field is derived from the electric (Bt and Bz 0); a map
    # with its own would fail `rebuild_error()`'s check.
    hr = e.get_hr()
    hz = e.get_hz()
    er = numpy.zeros((int(e.get_nz()), int(e.get_nr())), dtype=complex)
    ez = numpy.zeros_like(er)
    with _unscaled(e):
        for zi, ri in numpy.ndindex(er.shape):
            f = e.get_field_complex(ri * hr * 1000.0, 0.0, zi * hz * 1000.0, 0.0)[0]
            er[zi, ri] = f[0]
            ez[zi, ri] = f[2]
    return PKDict(
        Er=er,
        Ez=ez,
        Bt=0.0,
        Bz=0.0,
        hr=hr,
        hz=hz,
        length=-1.0,
        **_getters(e, "frequency", "direction"),
    )


def _constructor_args_sbend(e):
    # an SBend built without arguments and set has no field (get_field()
    # is 0, so it doesn't bend in a Volume), though its getters agree
    return _getters(e, "length", "angle", "P_over_Q", "E1", "E2", "K1")


def _constructor_args_solenoid(e):
    """A Solenoid built without arguments aborts the process in
    get_field(), and B0 and the radius have no getter/setter pair. B0 is
    mu0 times the total current per length, with RF-Track's own mu0 (not
    exported): the current of a 1 m, 1 T solenoid is 1/mu0. A thick
    winding's number of current sheets has no getter at all."""
    if e.get_inner_R() != e.get_outer_R():
        raise NotImplementedError(
            "archiving a Solenoid with a thick winding (inner_R != outer_R) isn't"
            + " supported: its number of current sheets has no getter"
        )
    return PKDict(
        length=e.get_length(),
        B0=e.get_max_current()
        / type(e)(1.0, 1.0, e.get_inner_R()).get_max_current()
        / e.get_length(),
        R=e.get_inner_R(),
    )


def _constructor_args_static_magnetic_fieldmap_1d(e):
    # There's no direct Bz-array getter for this class (unlike
    # RF_FieldMap_1d's get_Ez()), so this samples get_field() along the
    # element's own z range instead. get_field(x, y, z, t)'s coordinate
    # arguments are documented (RF-Track reference manual, section 5.1.2)
    # as millimeters, not meters like get_length()/get_z0()/get_z1(). Sampling
    # with z in mm reproduces a real solenoid field map's peak field exactly;
    # the returned B itself is already in Tesla, no further scaling needed.
    # Those are the samples RF-Track 2.7.4+ smoothed (see _unsmoothed()).
    h = e.get_hz()
    with _unscaled(e):
        b = _sampled_bz(e)
    return PKDict(Bz=_unsmoothed(b, lambda x: _sampled_bz(type(e)(x, h))), hz=h)


def _constructor_args_static_magnetic_fieldmap_2d(e):
    # No array getters either: samples get_field() over the (nz, nr) grid,
    # Br being Bx at (x=r, y=0), as for RF_FieldMap_2d
    hr = e.get_hr()
    z = numpy.arange(int(e.get_nz())) * e.get_hz() * 1000.0
    br = numpy.zeros((len(z), int(e.get_nr())))
    bz = numpy.zeros_like(br)
    with _unscaled(e):
        for ri in range(br.shape[1]):
            b = numpy.asarray(e.get_field(ri * hr * 1000.0, 0.0, z, 0.0)[1])
            br[:, ri] = b[:, 0]
            bz[:, ri] = b[:, 2]
    return PKDict(Br=br, Bz=bz, hr=hr, hz=e.get_hz())


def _constructor_function(class_name):
    """This module's `_constructor_args_<class>` function for
    `class_name`, or `None`."""
    return getattr(
        pykern.pkinspect.this_module(),
        "_constructor_args_" + constructor_class(class_name).lower(),
        None,
    )


def _fint(rft, e):
    """`e`'s (an SBend's) fringe field integral, which has a setter but no
    getter, measured instead: in a Lattice, the vertical edge focusing
    depends on `hgap * fint`, so a particle off axis in y leaves a clone
    of `e` (which keeps its fint) with a y' that rises smoothly with the
    clone's fint, solved for by the secant method. With `hgap` 0, fint
    has no effect, and 0 is returned. A Volume ignores fint: its 3D
    fringe field depends on `hgap` alone.
    """
    if e.get_hgap() == 0:
        return 0.0
    p = e.get_P_over_Q()

    def _yp(fint=None):
        c = e.clone()
        if fint is not None:
            c.set_fint(fint)
        lattice = rft.Lattice()
        lattice.append(c)
        # 1 um, so a small aperture doesn't stop it
        return lattice.track(
            rft.Bunch6d(
                rft.electronmass,
                1.0,
                numpy.sign(p),
                numpy.array([0.0, 0.0, 1e-3, 0.0, 0.0, abs(p)]),
            )
        ).get_phase_space("%yp")[0, 0]

    target = _yp()
    f0, f1 = 0.0, 1.0
    y0, y1 = _yp(f0), _yp(f1)
    for _ in range(20):
        if y1 == y0:
            break
        f0, f1 = f1, f1 - (y1 - target) * (f1 - f0) / (y1 - y0)
        y0, y1 = y1, _yp(f1)
        if abs(f1 - f0) <= 1e-12 * max(1.0, abs(f1)):
            break
    return f1


def _getters(e, *names):
    """`{name: e.get_<name>()}` for `names`, in that order."""
    return PKDict((n, getattr(e, "get_" + n)()) for n in names)


def _is_storable(value):
    """Whether HDF5 can hold `value`: a number, string or numeric array --
    not, e.g., an RF-Track `Frame` or `None`."""
    return numpy.asarray(value).dtype.kind in "biufcSU"


def _position_to_attrs(e):
    """Return `{x, y, z}` for a Volume-mode element's absolute placement.

    Raises:
        NotImplementedError: the element has a non-identity rotation --
            decoding RF-Track's (origin, nx, ny, nz) Frame back into the
            roll/pitch/yaw angles Volume.add() takes isn't implemented yet
            (every element archived so far happens to have no rotation,
            so this hasn't been needed, and guessing at the angle
            convention risks silently wrong physics), or (RF-Track
            2.7.4+) the element is misaligned
    """

    def _is_rotated(frame):
        return not all(
            numpy.allclose(numpy.asarray(f()).flatten(), v)
            for f, v in (
                (frame.get_nx, [1, 0, 0]),
                (frame.get_ny, [0, 1, 0]),
                (frame.get_nz, [0, 0, 1]),
            )
        )

    def _origin(frame):
        # Frame.get_origin() is in mm (e.g. a screen added at z=0.3 m reads
        # back as origin z=300), but Volume.add()'s own x/y/z args are in
        # meters, same as every other RF-Track setter in this module, so
        # this must convert back.
        return numpy.asarray(frame.get_origin()).flatten() / 1000.0

    # RF-Track 2.7.4 separated an element's placement (get_position(), new)
    # from its misalignment (get_offsets(), which had been both)
    if hasattr(e, "get_position"):
        if _is_rotated(e.get_offsets()) or numpy.any(_origin(e.get_offsets())):
            raise NotImplementedError(
                "archiving a misaligned Volume element isn't supported yet"
            )
        f = e.get_position()
    else:
        f = e.get_offsets()
    if _is_rotated(f):
        raise NotImplementedError(
            "archiving a Volume element with a non-identity rotation isn't supported yet"
        )
    o = _origin(f)
    return PKDict(x=o[0], y=o[1], z=o[2])


def _sampled_bz(e):
    """A Static_Magnetic_FieldMap_1d's on-axis Bz [T] at each of its own
    samples (get_field() takes z in mm)."""
    _, b = e.get_field(
        0.0,
        0.0,
        numpy.linspace(e.get_z0(), e.get_z1(), int(e.get_nz())) * 1000.0,
        0.0,
    )
    return numpy.asarray(b)[:, 2]


def _set_attribute(e, class_name, name, value):
    """`e.set_<name>(value)`, or with `value`'s elements as the
    arguments: a vector attribute's setter can take them separately, e.g.
    `set_static_Bfield(Bx, By, Bz)`, `Corrector.set_strength(Sx, Sy)`.

    Raises:
        NotImplementedError: the setter takes neither
    """
    s = getattr(e, "set_" + name)
    try:
        s(value)
        return
    except TypeError:
        pass
    if isinstance(value, numpy.ndarray):
        try:
            s(*value.ravel())
            return
        except TypeError:
            pass
    raise NotImplementedError(
        f"can't restore {class_name}.{name}: set_{name}() doesn't take"
        + f" what get_{name}() returns ({value!r})"
    )


@contextlib.contextmanager
def _unscaled(e):
    """`e` with no phase offset (`phid`), power scaling (`P_actual`),
    `scaling_factor` or synchronization time (`t0`) while sampling its
    field: those are restored by their setters, so a sampled field map
    must not include them already (for phid: sampling with it set, then
    setting it again, double-applies the rotation; likewise for an
    autophased element's t0, which an unset t0 samples like).
    """
    s = PKDict(
        {
            a: getattr(e, "get_" + a)()
            for a in ("phid", "P_actual", "scaling_factor")
            if hasattr(e, "get_" + a)
        }
    )
    if hasattr(e, "t0_is_set") and e.t0_is_set():
        s.t0 = e.get_t0()
    if "phid" in s:
        e.set_phid(0.0)
    if "P_actual" in s:
        e.set_P_actual(e.get_P_map())
    if "scaling_factor" in s:
        e.set_scaling_factor(1.0)
    if "t0" in s:
        e.unset_t0()
    try:
        yield
    finally:
        for a, v in s.items():
            getattr(e, "set_" + a)(v)


def _unsmoothed(samples, smooth):
    """The samples to build a 1D field map from so that it holds `samples`.

    RF-Track 2.7.4+ smooths a 1D field map's samples (Savitzky-Golay) when
    it's built, then tracks with, and returns from `get_Ez()`, the
    smoothed samples, so building it again from those smooths them twice.
    The smoothing is linear and banded: this measures it on `smooth()` and
    solves for what smooths to `samples`. Older RF-Track doesn't smooth, so
    `samples` comes back as is. `rebuild_error()` checks the result.

    Args:
        samples (numpy.ndarray): what the built field map must hold
        smooth (callable): samples -> what a field map built from them holds
    Returns:
        numpy.ndarray: what to build the field map from
    """

    def _banded(width):
        # scipy.linalg.solve_banded()'s layout: row width + i - k of column k
        # is the matrix's (i, k). Impulses (2 * width + 1) apart can't
        # overlap, so one smooth() call measures every such column at once.
        rv = numpy.zeros((2 * width + 1, len(samples)))
        for j in range(rv.shape[0]):
            r = numpy.real(smooth(_comb(j, rv.shape[0])))
            for k in range(j, len(samples), rv.shape[0]):
                i = numpy.arange(max(k - width, 0), min(k + width + 1, len(samples)))
                rv[width + i - k, k] = r[i]
        return rv

    def _comb(start, step):
        rv = numpy.zeros(len(samples), dtype=samples.dtype)
        rv[start::step] = 1.0
        return rv

    def _width():
        # POSIT: Savitzky-Golay: an interior sample's window is +/- w, and one
        # within w of an end shares that end's window, so no sample moves
        # another more than 2 * w away
        c = len(samples) // 2
        r = numpy.flatnonzero(smooth(_comb(c, len(samples))))
        return min(2 * int(numpy.abs(r - c).max()), len(samples) - 1)

    if numpy.array_equal(smooth(samples), samples):
        return samples
    w = _width()
    return scipy.linalg.solve_banded((w, w), _banded(w), samples)
