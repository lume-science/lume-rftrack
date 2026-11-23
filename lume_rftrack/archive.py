"""HDF5 archive format for lume_rftrack.simulator.RFTrackSimulator

`RFTrackSimulator.archive()`/`load_archive()` (`lume_rftrack.simulator`)
delegate to :func:`archive`/:func:`load_archive` here, and
:func:`read_archive` reads an archive without RF-Track (e.g. for another
program to import it). This module holds the actual beamline + bunch <->
HDF5 serialization -- not a bespoke format of its own, just RF-Track's
own native class/attribute names, written as plain, self-describing HDF5
any tool can read.

Reading an element's state from RF-Track, and rebuilding it, is
`lume_rftrack.archive_elements`'s: everything RF-Track doesn't report
directly, and its version differences, are handled there. `archive()`
rebuilds every element in memory and compares its field with the
original's, so an element that wouldn't come back the same is refused
(or, with `strict=False`, archived with its error recorded) instead of
archived wrong.

Like lume-impact's and lume-genesis' archives, it holds the last run's
output as well as its input: the tracked bunch, if there is one, so a
loaded simulator has its final particles without tracking again.
"""

import importlib.metadata
import warnings

import beamphysics
import h5py
import numpy
from beamphysics.units import pmd_unit, write_unit_h5
from pykern.pkcollections import PKDict

import lume_rftrack.archive_elements
import lume_rftrack.units
from lume_rftrack.utils import (
    bunch6d_to_particlegroup,
    bunch6dt_to_particlegroup,
    particlegroup_to_bunch6d,
    particlegroup_to_bunch6dt,
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


def archive(simulator, path, autophase=None, space_charge=None, strict=True):
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
    `RFTrackSimulator.apply_space_charge()` to use it. Collective effects
    (e.g. a `ShortRangeWakefield`), on an element or the beamline, are
    not archived: RF-Track has no getters for their parameters, so they
    are dropped with a `UserWarning`, and a reloaded beamline tracks
    without them. So is a settable attribute whose value HDF5 can't hold
    (e.g. an RF-Track `Frame` from a getter a newer RF-Track adds).

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
        strict (bool): refuse an element rebuilt with a different field;
            `False` archives it anyway, with a `UserWarning`, and records
            its field error (`restore_field_error`) for `load_archive()`
            to warn about again, e.g. to archive with an RF-Track release
            this module doesn't support yet [True]

    Raises:
        NotImplementedError: an element that can't be rebuilt at all, or
            (if `strict`) not with the same field (see
            `lume_rftrack.archive_elements`), or a Volume element with a
            non-identity rotation

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
      the aperture, `t0` if set, in a Volume its position `x`/`y`/`z`,
      `unarchived_attributes` (the names of attributes HDF5 couldn't
      hold, if any) and `restore_field_error` (if archived with
      `strict=False` and rebuilt with a different field: the largest
      field difference over the largest field); an `attributes` group of
      its `get_X()`/`set_X()` values; and,
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
    w = PKDict(dropped=_collective_effects(simulator.beamline, kind), inexact=[])
    for i in range(simulator.beamline.size()):
        e = simulator.beamline[i]
        d = lume_rftrack.archive_elements.to_dict(rft, e, kind)
        r = lume_rftrack.archive_elements.rebuild_error(rft, e, d)
        if r:
            if strict:
                raise NotImplementedError(
                    f"a {d['class']!r} element rebuilt from its attributes has a"
                    + " different field; archiving it needs a"
                    + " _constructor_args_"
                    + lume_rftrack.archive_elements.constructor_class(
                        d["class"]
                    ).lower()
                    + "() in lume_rftrack.archive_elements"
                    + " (or archive(strict=False))"
                )
            d.restore_field_error = r
            w.inexact.append(f"element {i} {d.name!r}: {r:.2g}")
        elements.append(d)
        w.dropped += _collective_effects(e, f"element {i} {d.name!r}")
        w.dropped += [
            f"element {i} {d.name!r}: {a}" for a in d.get("unarchived_attributes", ())
        ]
    if w.dropped:
        warnings.warn(
            "collective effects and attributes HDF5 can't hold are not archived,"
            + " dropped: "
            + "; ".join(w.dropped),
            stacklevel=3,
        )
    if w.inexact:
        warnings.warn(
            "archived with a different field when rebuilt (strict=False),"
            + " largest error over largest field: "
            + "; ".join(w.inexact),
            stacklevel=3,
        )
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

    Warns (`UserWarning`) if `path` was written by a different RF-Track,
    which may track the same beamline differently (e.g. 2.7.4 smooths 1D
    field maps 2.7.3 doesn't), or holds elements `archive(strict=False)`
    couldn't rebuild with the same field.
    """
    import RF_Track as rft

    a = read_archive(path)
    if a.rftrack_version not in (None, str(rft.RF_TRACK_VERSION)):
        warnings.warn(
            f"{path} was written by RF-Track {a.rftrack_version}, loaded with"
            + f" {rft.RF_TRACK_VERSION}: it may track differently",
            stacklevel=3,
        )
    r = [
        f"element {i} {e.name!r}: {e.restore_field_error:.2g}"
        for i, e in enumerate(a.elements)
        if "restore_field_error" in e
    ]
    if r:
        warnings.warn(
            f"{path} was archived with strict=False, rebuilt with a different"
            + " field, largest error over largest field: "
            + "; ".join(r),
            stacklevel=3,
        )
    simulator.autophase = a.autophase
    simulator.space_charge = a.space_charge
    if a.beamline_class == "Lattice":
        beamline = rft.Lattice()
        for e in a.elements:
            beamline.append(lume_rftrack.archive_elements.from_dict(rft, e))
        beamline.set_elements_offsets(a.elements_offsets)
    elif a.beamline_class == "Volume":
        beamline = rft.Volume()
        for e in a.elements:
            beamline.add(
                lume_rftrack.archive_elements.from_dict(rft, e),
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
                `name`, the aperture, `t0`, `x`/`y`/`z`,
                `unarchived_attributes`, `restore_field_error` (see
                `archive()`),
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


def _attr_value(v):
    """An HDF5 attr as a plain Python value: h5py returns numpy scalars."""
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if isinstance(v, numpy.generic) else v


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


def _collective_effects(e, label):
    """`e`'s collective effects, as `["<label>: <class>"]`. RF-Track's
    `get_collective_effects()` returns `[]`, a single effect, or a list
    of them."""
    c = e.get_collective_effects()
    if not isinstance(c, list):
        c = [c]
    return [f"{label}: {type(x).__name__}" for x in c]


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


def _infer_autophase(elements):
    """Infer whether `autophase()` was used, from the already-serialized
    `elements` themselves, for `archive()`'s own `autophase=None` case.

    An RF element (one with a `phid`) `autophase()`'d before archiving
    has a real, nonzero `t0` (see `lume_rftrack.archive_elements.to_dict()`'s comment --
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
