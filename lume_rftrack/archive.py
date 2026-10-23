"""HDF5 archive format for lume_rftrack.simulator.RFTrackSimulator

`RFTrackSimulator.archive()`/`load_archive()` (`lume_rftrack.simulator`)
delegate to :func:`archive`/:func:`load_archive` here, and
:func:`read_archive` reads an archive without RF-Track (e.g. for another
program to import it). This module holds the actual beamline + bunch <->
HDF5 serialization -- not a bespoke format of its own, just RF-Track's
own native class/attribute names, written as plain, self-describing HDF5
any tool can read.

RF-Track elements can't be pickled and have no save/load API, so an
element is archived generically, as RF-Track itself exposes it: every
`get_X()`/`set_X()` pair whose getter takes no arguments, restored by
building the class with no arguments and calling each `set_X()`. A class
whose state isn't all settable that way (a field map's data, a pillbox's
cell count, ...) also has a `_constructor_args_<class>()`, giving the
arguments to build it with, in order.
`archive()` rebuilds every element in memory and compares its field with
the original's, so an element that wouldn't come back the same is
refused instead of archived wrong.

Like lume-impact's and lume-genesis' archives, it holds the last run's
output as well as its input: the tracked bunch, if there is one, so a
loaded simulator has its final particles without tracking again.
"""

import contextlib
import importlib.metadata

import beamphysics
import h5py
import numpy
import pykern.pkinspect
from beamphysics.units import pmd_unit, write_unit_h5
from pykern.pkcollections import PKDict

import lume_rftrack.units
from lume_rftrack.utils import (
    bunch6d_to_particlegroup,
    bunch6dt_to_particlegroup,
    particlegroup_to_bunch6d,
    particlegroup_to_bunch6dt,
)

# Element accessors archived by name rather than generically: `name` and
# `t0` (only when `t0_is_set()`), the aperture (whose setter is
# `set_aperture(x, y, shape)`), and `offsets` (the Lattice's
# `elements_offsets` or the Volume position instead)
_NOT_GENERIC = frozenset(
    ("aperture_shape", "aperture_x", "aperture_y", "name", "offsets", "t0")
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

# Volume tracking settings archive() records and load_archive() restores
_VOLUME_INTEGRATOR_ATTRS = (
    "dt_mm",
    "odeint_algorithm",
    "odeint_epsabs",
    "odeint_epsrel",
    "t_max_mm",
    "tt_dt_mm",
)


def archive(simulator, path, autophase=None, space_charge=None):
    """Write `simulator.beamline` and `simulator.initial_bunch` to `path`
    as HDF5 (called as `RFTrackSimulator.archive()`), and
    `simulator.tracked_bunch` if there is one.

    `path` is also a valid standalone openPMD particle file (readable
    with `beamphysics.ParticleGroup(h5=path)`). The bunch comes back as
    the same macroparticles, not a new realization; the random seed is
    not recorded. Each element's aperture and settable attributes
    (including its step counts), a Lattice's element offsets and a
    Volume's integrator settings are recorded too, and an autophased RF
    element keeps its phase correction, so a reloaded beamline tracks
    identically without calling `autophase()` again. Space charge is
    recorded as settings only: after loading, call
    `RFTrackSimulator.apply_space_charge()` to use it.

    Args:
        simulator (RFTrackSimulator): the `Lattice` or `Volume` and the
            `Bunch6d` or `Bunch6dT` to write
        path (str): file to write
        autophase (bool): whether autophase was used; `None` infers it
            from the RF elements' `t0` [None]
        space_charge (dict): `{"kind": "p2p"}` or `{"kind": "pic",
            "dt_mm": float, "grid": (nx, ny, nz), "mirror_m": float,
            "smooth": float}`; `None` uses `simulator.space_charge`
            [None]. RF-Track cannot report its active space-charge
            engine, so this cannot be inferred. To archive a loaded
            simulator without space charge, set
            `simulator.space_charge = None` first.

    Raises:
        NotImplementedError: an element that can't be rebuilt with the
            same field (see the module docstring), or a Volume element
            with a non-identity rotation

    The file is plain HDF5, readable without this module:

    - root attrs: `software`, `lume_rftrack_version`, `rftrack_version`
      (RF-Track's own), `bunch_type`, `bunch_t_mm` (a `Bunch6dT`'s clock,
      which its particles don't carry), `beamline_class`, `autophase`, and
      for a Volume `s0`/`s1` and its integrator settings (`dt_mm`,
      `tt_dt_mm`, `odeint_algorithm`, `odeint_epsabs`, `odeint_epsrel`,
      `t_max_mm`)
    - `elements_offsets`: for a Lattice, one row per element, as
      `Lattice.set_elements_offsets()` takes them
    - `elements/<i>`: one group per element, with attrs `class`, `name`,
      the aperture, `t0` if set and, in a Volume, its position `x`/`y`/
      `z`; an `attributes` group of its `get_X()`/`set_X()` values; and,
      for a class that needs them, a `constructor` group of its
      constructor arguments, named in order by the `constructor_args` attr
    - `field_maps`: each unique constructor array (a field map) once,
      keyed by md5 and referenced from a `constructor` as `"@<md5>"`
    - `space_charge`: the settings above, if any
    - `units`: the unit of each element value, laid out like `elements`
    - `particles`: the initial bunch, as openPMD
    - `output`: only if the simulator has tracked, the tracked bunch:
      attrs `bunch_type` and `bunch_t_mm`, as at the root, and its
      `particles`, as openPMD
    """
    import RF_Track as rft

    kind = type(simulator.beamline).__name__
    elements = []
    for i in range(simulator.beamline.size()):
        e = simulator.beamline[i]
        d = _element_to_attrs(rft, e, kind)
        _assert_restorable(rft, e, d)
        elements.append(d)
    elements, field_maps = _dedupe_field_maps(elements)
    particle_group, bunch_type = _bunch_to_particlegroup(simulator.initial_bunch)
    tracked = (
        None
        if simulator.tracked_bunch is None
        else _bunch_to_particlegroup(simulator.tracked_bunch)
    )
    if autophase is None:
        autophase = _infer_autophase(elements)
    if space_charge is None:
        space_charge = simulator.space_charge
    if space_charge is not None:
        space_charge = PKDict(space_charge)
    with h5py.File(path, "w") as f:
        # no dataType attr of our own: ParticleGroup.write() sets (and
        # would overwrite) openPMD's
        f.attrs["software"] = "lume_rftrack"
        f.attrs["lume_rftrack_version"] = importlib.metadata.version("lume-rftrack")
        f.attrs["rftrack_version"] = rft.RF_TRACK_VERSION
        f.attrs["bunch_type"] = bunch_type
        _write_bunch_time(f, simulator.initial_bunch)
        f.attrs["beamline_class"] = kind
        f.attrs["autophase"] = int(autophase)
        if space_charge is not None:
            sc = f.create_group("space_charge")
            sc.attrs["kind"] = space_charge.kind
            if space_charge.kind == "pic":
                sc.attrs["dt_mm"] = float(space_charge.dt_mm)
                sc.attrs["grid"] = numpy.asarray(space_charge.grid, dtype=int)
                sc.attrs["mirror_m"] = float(space_charge.mirror_m)
                sc.attrs["smooth"] = float(space_charge.smooth)
        if kind == "Volume":
            # get_s0()/get_s1() return a Frame (like an element's
            # get_offsets()), not a plain float -- its origin's z
            # component (in mm, same as get_offsets(), hence /1000) is
            # the tracking-range value set_s0()/set_s1() itself takes
            # (in meters).
            f.attrs["s0"] = (
                float(
                    numpy.asarray(simulator.beamline.get_s0().get_origin()).flatten()[2]
                )
                / 1000.0
            )
            f.attrs["s1"] = (
                float(
                    numpy.asarray(simulator.beamline.get_s1().get_origin()).flatten()[2]
                )
                / 1000.0
            )
            for k in _VOLUME_INTEGRATOR_ATTRS:
                f.attrs[k] = getattr(simulator.beamline, k)
        else:
            # one row per element, as Lattice.set_elements_offsets() takes it
            f.create_dataset(
                "elements_offsets",
                data=numpy.asarray(simulator.beamline.get_elements_offsets()),
            )
        _write_indexed_group(f, "elements", elements)
        if field_maps:
            fm = f.create_group("field_maps")
            for h, arr in field_maps.items():
                fm.create_dataset(h, data=arr)
        units_elements = f.create_group("units").create_group("elements")
        for i, element in enumerate(elements):
            _write_field_units_h5(
                units_elements.create_group(str(i)),
                element,
                getattr(rft, element["class"]),
            )
        particle_group.write(f)
        if tracked is not None:
            o = f.create_group("output")
            o.attrs["bunch_type"] = tracked[1]
            _write_bunch_time(o, simulator.tracked_bunch)
            tracked[0].write(o.create_group("particles"))


def load_archive(simulator, path):
    """Replace `simulator.beamline` and `simulator.initial_bunch` with the
    contents of `path` (as written by `archive()`; called as
    `RFTrackSimulator.load_archive()`), rebuilding a new `rft.Lattice`
    or `rft.Volume` and the exact same macroparticles `archive()`
    wrote. `simulator.tracked_bunch` becomes the archived tracked bunch,
    or `None` if there was none; the rebuilt beamline itself hasn't
    tracked (its screens and transport table are empty).

    Also sets `simulator.autophase`/`simulator.space_charge` from
    whatever `archive()` was given (`None` for either if it wasn't
    passed at archive time) -- purely informational, same as
    `simulator.beamline`/`simulator.initial_bunch`; this never acts on
    them itself (e.g. it doesn't call `autophase()` or build an
    `SC_engine`; call `simulator.apply_space_charge()` for that).
    """
    import RF_Track as rft

    a = read_archive(path)
    simulator.autophase = a.autophase
    simulator.space_charge = a.space_charge
    if a.beamline_class == "Lattice":
        beamline = rft.Lattice()
        for e in a.elements:
            beamline.append(_element_from_attrs(rft, e))
        beamline.set_elements_offsets(a.elements_offsets)
    elif a.beamline_class == "Volume":
        beamline = rft.Volume()
        for e in a.elements:
            beamline.add(
                _element_from_attrs(rft, e),
                float(e.x),
                float(e.y),
                float(e.z),
                0.0,
                0.0,
                0.0,
            )
        beamline.set_s0(a.s0)
        beamline.set_s1(a.s1)
        for k, v in a.integrator.items():
            setattr(beamline, k, v)
    else:
        raise NotImplementedError(
            f"don't know how to restore a beamline_class={a.beamline_class!r}"
        )
    simulator.beamline = beamline
    simulator.initial_bunch = _restored_bunch(a)
    simulator.tracked_bunch = None if a.output is None else _restored_bunch(a.output)


def read_archive(path):
    """Read everything `archive()` wrote to `path`, without RF-Track.

    Returns:
        PKDict:
            autophase (bool): `None` if not recorded
            beamline_class (str): "Lattice" or "Volume"
            bunch_t_mm (float): a `Bunch6dT`'s clock [mm/c], else `None`
            bunch_type (str): "Bunch6d" or "Bunch6dT"
            elements (list): one `PKDict` per element, keyed by `class`,
                `name`, the aperture, `t0`, `x`/`y`/`z` (see `archive()`),
                with RF-Track's own attribute names in `attributes` and,
                if any, its constructor arguments in `constructor`, field
                maps resolved to arrays
            elements_offsets (numpy.ndarray): a Lattice's element offsets,
                as `Lattice.set_elements_offsets()` takes them, else `None`
            integrator (PKDict): a Volume's `dt_mm`, `tt_dt_mm`,
                `odeint_algorithm`, `odeint_epsabs`, `odeint_epsrel` and
                `t_max_mm`, else `None`
            lume_rftrack_version (str): the lume-rftrack that wrote it,
                `None` if not recorded
            output (PKDict): the tracked bunch, with `bunch_t_mm`,
                `bunch_type` and `particle_group` like the initial
                bunch's, or `None` if the simulator hadn't tracked
            particle_group (beamphysics.ParticleGroup): the initial bunch
            rftrack_version (str): the RF-Track that wrote it, `None` if not
                recorded
            s0 (float): a Volume's tracking start [m], else `None`
            s1 (float): a Volume's tracking end [m], else `None`
            space_charge (PKDict): see `archive()`, or `None`
    """
    with h5py.File(path, "r") as f:
        elements = _read_indexed_group(f, "elements")
        if "field_maps" in f:
            elements = [_resolve_field_map_refs(e, f["field_maps"]) for e in elements]
        space_charge = None
        if "space_charge" in f:
            space_charge = PKDict(f["space_charge"].attrs)
            if "grid" in space_charge:
                space_charge.grid = tuple(int(n) for n in space_charge.grid)
        volume = str(f.attrs["beamline_class"]) == "Volume"
        output = None
        if "output" in f:
            o = f["output"]
            output = PKDict(
                bunch_t_mm=_bunch_time_attr(o),
                bunch_type=str(o.attrs["bunch_type"]),
                particle_group=beamphysics.ParticleGroup(h5=o["particles"]),
            )
        return PKDict(
            autophase=bool(f.attrs["autophase"]) if "autophase" in f.attrs else None,
            beamline_class=str(f.attrs["beamline_class"]),
            bunch_t_mm=_bunch_time_attr(f),
            bunch_type=str(f.attrs["bunch_type"]),
            elements=elements,
            elements_offsets=None if volume else f["elements_offsets"][()],
            integrator=(
                PKDict({k: _attr_value(f.attrs[k]) for k in _VOLUME_INTEGRATOR_ATTRS})
                if volume
                else None
            ),
            lume_rftrack_version=(
                str(f.attrs["lume_rftrack_version"])
                if "lume_rftrack_version" in f.attrs
                else None
            ),
            output=output,
            particle_group=beamphysics.ParticleGroup(h5=f),
            rftrack_version=(
                str(f.attrs["rftrack_version"])
                if "rftrack_version" in f.attrs
                else None
            ),
            s0=float(f.attrs["s0"]) if "s0" in f.attrs else None,
            s1=float(f.attrs["s1"]) if "s1" in f.attrs else None,
            space_charge=space_charge,
        )


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


def _assert_restorable(rft, e, d):
    """Rebuild `e` from `d` (its `_element_to_attrs()`) and require the
    same field, on and off axis, from before its entrance to past its
    exit -- what tracking actually sees, and what catches constructor
    state no getter exposes (e.g. a solenoid's radius).

    Raises:
        NotImplementedError: the rebuilt element's field differs, or it
            can't be rebuilt at all
    """
    n = _element_from_attrs(rft, d)
    z = numpy.linspace(-0.1, 1.1, 25) * max(e.get_length(), 0.01) * 1000.0
    a, b = [], []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (-2.0, 1.5)):
        a.append(numpy.hstack(e.get_field(x, y, z, 0.0)))
        b.append(numpy.hstack(n.get_field(x, y, z, 0.0)))
    a = numpy.array(a)
    if not numpy.allclose(
        numpy.array(b), a, rtol=1e-9, atol=1e-12 * numpy.max(numpy.abs(a))
    ):
        raise NotImplementedError(
            f"a {d['class']!r} element rebuilt from its attributes has a"
            + " different field; archiving it needs a"
            + f" _constructor_args_{_constructor_class(d['class']).lower()}()"
        )


def _attr_value(v):
    """An HDF5 attr as a plain Python value: h5py returns numpy scalars."""
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if isinstance(v, numpy.generic) else v


def _attributes(e):
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


def _bunch_to_particlegroup(bunch):
    """Convert `bunch` (a Bunch6d or Bunch6dT) to a `ParticleGroup`,
    returning `(particle_group, bunch_type)`. `bunch_type` records
    which conversion was used -- `particlegroup_to_bunch6d()`/
    `particlegroup_to_bunch6dt()` are not interchangeable (opposite
    z/t conventions), and nothing about the
    resulting `ParticleGroup` itself says which one made it.
    """
    import RF_Track as rft

    if isinstance(bunch, rft.Bunch6dT):
        return bunch6dt_to_particlegroup(bunch, snapshot_time=False), "Bunch6dT"
    return bunch6d_to_particlegroup(bunch), "Bunch6d"


def _bunch_time_attr(group):
    """The `bunch_t_mm` attr `_write_bunch_time()` wrote on `group`, or
    `None`."""
    return float(group.attrs["bunch_t_mm"]) if "bunch_t_mm" in group.attrs else None


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
            + f" _constructor_args_{_constructor_class(d['class']).lower()}()"
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
    # get_Ez() is the exact array the field map file was loaded into --
    # confirmed empirically this round-trips a reconstructed element's
    # get_field() output exactly (max diff 0.0) against the original.
    return _getters(e, "Ez", "hz", "length", "frequency", "direction")


def _constructor_args_rf_fieldmap_2d(e):
    # Unlike RF_FieldMap_1d's get_Ez(), there's no direct 2D array getter,
    # so this samples get_field_complex(x, y, z, t) point by point over
    # the full (nz, nr) grid -- cylindrical symmetry means Er is just the
    # Cartesian Ex component at (x=r, y=0) (confirmed empirically: sampling
    # at (x=0, y=r) instead shows the identical value in Ey, not Ex). The
    # magnetic field is derived from the electric (Bt and Bz 0); a map
    # with its own would fail `_assert_restorable()`.
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
    # as millimeters, not meters like get_length()/get_z0()/get_z1() --
    # confirmed empirically: sampling with z in mm reproduces a real
    # solenoid field map's documented peak field exactly, matching its
    # source field-map file; the returned B itself is already in Tesla,
    # no further scaling needed.
    z = numpy.linspace(e.get_z0(), e.get_z1(), int(e.get_nz())) * 1000.0
    with _unscaled(e):
        _, b = e.get_field(0.0, 0.0, z, 0.0)
    return PKDict(Bz=numpy.asarray(b)[:, 2], hz=e.get_hz())


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


def _constructor_class(class_name):
    """The class whose constructor functions `class_name` uses."""
    return _SAME_CONSTRUCTOR.get(class_name, class_name)


def _constructor_function(class_name):
    """This module's `_constructor_args_<class>` function for
    `class_name`, or `None`."""
    return getattr(
        pykern.pkinspect.this_module(),
        "_constructor_args_" + _constructor_class(class_name).lower(),
        None,
    )


def _dedupe_field_maps(elements):
    """Replace each array in an element's `constructor` (a field map,
    e.g. RF_FieldMap_1d's "Ez") with a short reference ("@<md5>") to one
    shared copy, keyed by the array's own content hash -- several
    elements built from the same field-map file (e.g. a linac's
    identical accelerating cavities) are otherwise archived with a
    byte-for-byte duplicate of the same array once per element.

    Returns:
        (list, dict): the deduped element dicts (copies -- the originals
            are untouched), and `{md5: array}` for every unique array
            found, to write once under a shared top-level group (see
            `archive()`). A reference is plain text, written as a normal
            attr like any other string.
    """
    import hashlib

    field_maps = {}
    deduped = []
    for e in elements:
        d = PKDict(e)
        if "constructor" in e:
            d.constructor = PKDict(e.constructor)
            for k, v in e.constructor.items():
                if isinstance(v, numpy.ndarray):
                    h = hashlib.md5(v.tobytes()).hexdigest()
                    field_maps[h] = v
                    d.constructor[k] = f"@{h}"
        deduped.append(d)
    return deduped, field_maps


def _element_from_attrs(rft, d):
    """The element `d` (from `_element_to_attrs()` or `read_archive()`)
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


def _element_to_attrs(rft, e, beamline_class="Lattice"):
    cls = type(e).__name__
    d = PKDict(name=e.get_name(), attributes=_attributes(e))
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
    # the phid the element was constructed with (confirmed empirically:
    # get_phid() stays unchanged across autophase()) -- t0 is the actual
    # correction autophase() applies and .track() uses as-is. Archiving
    # it directly (confirmed empirically: a fresh beamline with set_t0()
    # from these captured values, autophase() never called again,
    # reproduces the autophased beamline's own tracked mean kinetic
    # energy exactly) means load_archive() reproduces the real physics
    # regardless of whether the caller re-runs autophase() or even knows
    # it was ever used, rather than depending on that recomputation
    # happening to be reproducible.
    if hasattr(e, "t0_is_set") and e.t0_is_set():
        d.t0 = e.get_t0()
    return d


def _getters(e, *names):
    """`{name: e.get_<name>()}` for `names`, in that order."""
    return PKDict((n, getattr(e, "get_" + n)()) for n in names)


def _infer_autophase(elements):
    """Infer whether `autophase()` was used, from the already-serialized
    `elements` themselves, for `archive()`'s own `autophase=None` case.

    An RF element (one with a `phid`) `autophase()`'d before archiving
    has a real, nonzero `t0` (see `_element_to_attrs()`'s own comment --
    `get_phid()` never reflects it, `t0` does); one that was never
    autophased either has no `"t0"` key at all (`t0_is_set()` was
    `False`) or, if the caller deliberately pinned it to suppress a
    `Bunch6dT` beamline's own unwanted rescue autophase on `.track()`,
    exactly `0.0`. Either way, "any RF element with a nonzero `t0`" is
    the signal.
    """
    return any("phid" in d.attributes and d.get("t0", 0.0) != 0.0 for d in elements)


def _particlegroup_to_bunch(particle_group, bunch_type):
    """The exact inverse of `_bunch_to_particlegroup()`."""
    if bunch_type == "Bunch6dT":
        return particlegroup_to_bunch6dt(particle_group)
    return particlegroup_to_bunch6d(particle_group)


def _position_to_attrs(e):
    """Return `{x, y, z}` for a Volume-mode element's absolute placement.

    Raises:
        NotImplementedError: the element has a non-identity rotation --
            decoding RF-Track's (origin, nx, ny, nz) Frame back into the
            roll/pitch/yaw angles Volume.add() takes isn't implemented yet
            (every element archived so far happens to have no rotation,
            so this hasn't been needed, and guessing at the angle
            convention risks silently wrong physics).
    """
    fr = e.get_offsets()
    origin = numpy.asarray(fr.get_origin()).flatten()
    nx = numpy.asarray(fr.get_nx()).flatten()
    ny = numpy.asarray(fr.get_ny()).flatten()
    nz = numpy.asarray(fr.get_nz()).flatten()
    if not (
        numpy.allclose(nx, [1, 0, 0])
        and numpy.allclose(ny, [0, 1, 0])
        and numpy.allclose(nz, [0, 0, 1])
    ):
        raise NotImplementedError(
            "archiving a Volume element with a non-identity rotation isn't supported yet"
        )
    # Frame.get_origin() is in mm, confirmed empirically (e.g. a screen
    # added at z=0.3 m reads back as origin z=300) -- but Volume.add()'s
    # own x/y/z args are in meters, same as every other RF-Track setter in
    # this module, so this must convert back.
    return PKDict(x=origin[0] / 1000.0, y=origin[1] / 1000.0, z=origin[2] / 1000.0)


def _read_attrs_h5(group):
    """The inverse of `_write_attrs_h5()`."""
    d = PKDict({k: _attr_value(v) for k, v in group.attrs.items()})
    for k, v in group.items():
        d[k] = _read_attrs_h5(v) if isinstance(v, h5py.Group) else v[()]
    return d


def _read_indexed_group(parent, name):
    g = parent[name]
    return [_read_attrs_h5(g[str(i)]) for i in range(len(g))]


def _resolve_field_map_refs(element, field_maps_group):
    """The exact inverse of `_dedupe_field_maps()` for one element:
    replace every "@<md5>" reference in its `constructor` with the real
    array read from `field_maps_group[md5]`.
    """
    if "constructor" not in element:
        return element
    d = PKDict(element)
    d.constructor = PKDict(
        {
            k: (
                field_maps_group[v[1:]][()]
                if isinstance(v, str) and v.startswith("@")
                else v
            )
            for k, v in element.constructor.items()
        }
    )
    return d


def _restored_bunch(data):
    """The bunch `data` (`read_archive()`'s, or its `output`) holds, with a
    `Bunch6dT`'s clock set back: its particles alone leave it at their
    creation time."""
    res = _particlegroup_to_bunch(data.particle_group, data.bunch_type)
    if data.bunch_t_mm is not None:
        res.t = data.bunch_t_mm
    return res


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
    """`e` with no phase offset (`phid`), power scaling (`P_actual`) or
    `scaling_factor` while sampling its field: those are restored by
    their setters, so a sampled field map must not include them already
    (confirmed empirically for phid: sampling with it set, then setting
    it again, double-applies the rotation).
    """
    s = PKDict(
        {
            a: getattr(e, "get_" + a)()
            for a in ("phid", "P_actual", "scaling_factor")
            if hasattr(e, "get_" + a)
        }
    )
    if "phid" in s:
        e.set_phid(0.0)
    if "P_actual" in s:
        e.set_P_actual(e.get_P_map())
    if "scaling_factor" in s:
        e.set_scaling_factor(1.0)
    try:
        yield
    finally:
        for a, v in s.items():
            getattr(e, "set_" + a)(v)


def _write_attrs_h5(group, data):
    """Write a flat dict of scalars/small arrays as attrs on `group`,
    except a numpy array (e.g. a field map), which goes in as its own
    dataset instead -- an attr isn't meant to hold something that size --
    and a dict, which goes in as a child group, written the same way.

    Mirrors lume-impact's `write_attrs_h5()`/`write_fieldmap_h5()`, the
    simplest pattern in the LUME ecosystem for this.
    """
    for k, v in data.items():
        if isinstance(v, dict):
            _write_attrs_h5(group.create_group(k), v)
        elif isinstance(v, numpy.ndarray):
            group.create_dataset(k, data=v)
        else:
            group.attrs[k] = v


def _write_bunch_time(group, bunch):
    """Record a `Bunch6dT`'s clock `t` [mm/c] as `group`'s `bunch_t_mm`
    attr; a `Bunch6d` has none."""
    import RF_Track as rft

    if isinstance(bunch, rft.Bunch6dT):
        group.attrs["bunch_t_mm"] = float(bunch.t)


def _write_field_units_h5(group, data, element_class):
    """Write one child group per key in `data` that has a unit
    (`lume_rftrack.units.unit()` for `element_class`), each holding
    openPMD-style `unitSI`/`unitDimension`/`unitSymbol` attrs
    (`beamphysics.units.write_unit_h5()`, given a `pmd_unit` built from
    that symbol on the spot), and one child group per dict in `data`,
    written the same way, so `units` mirrors `elements`.

    Lives under a top-level `units` group parallel to `elements`/`bunch`,
    not nested inside them, so `elements` holds only RF-Track's values.
    Purely additive documentation; `load_archive()` never reads this
    back. A key with no unit (`None`, or not in the tables, like `name`)
    is left out.
    """
    for k, v in data.items():
        if isinstance(v, dict):
            _write_field_units_h5(group.create_group(k), v, element_class)
            continue
        try:
            s = lume_rftrack.units.unit(element_class, k)
        except KeyError:
            continue
        if s is not None:
            write_unit_h5(group.create_group(k), pmd_unit(s))


def _write_indexed_group(parent, name, items):
    """Write `items` (each a dict) as `parent/name/0`, `.../1`, ..."""
    g = parent.create_group(name)
    for i, item in enumerate(items):
        _write_attrs_h5(g.create_group(str(i)), item)
