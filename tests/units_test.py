"""lume_rftrack.units: every symbol parses to a real unit, every attribute of
every RF-Track element is in the table, and every numeric field an archive
writes has a unit unless the table says it has none.
"""


def test_every_element_attribute_is_in_the_table():
    """Including classes nothing here builds yet: the archive writes every
    get_X()/set_X() pair, so a new RF-Track attribute must be added (with
    `None` if its unit is unknown)."""
    import RF_Track as rft
    from pykern import pkunit

    from lume_rftrack.archive import _attributes
    from lume_rftrack.units import unit

    missing = set()
    for n in dir(rft):
        c = getattr(rft, n)
        if (
            not isinstance(c, type)
            or not issubclass(c, rft.Element)
            # beamlines, not elements
            or issubclass(c, rft.Lattice | rft.Volume)
        ):
            continue
        try:
            e = c()
        except (AttributeError, TypeError):
            # no constructor without arguments: archived only through a
            # _constructor_args_<class>()
            continue
        for a in _attributes(e):
            try:
                unit(c, a)
            except KeyError:
                missing.add(f"{n}.{a}")
    pkunit.pkeq(set(), missing)


def test_every_symbol_is_a_real_unit():
    from beamphysics.units import pmd_unit
    from pykern import pkunit

    from lume_rftrack.units import CLASS_UNITS, UNITS

    for k, v in list(UNITS.items()) + [
        (f"{c}.{a}", v) for c, u in CLASS_UNITS.items() for a, v in u.items()
    ]:
        if v is None:
            continue
        pkunit.pkok(isinstance(v, str) and v, "{!r}: {!r} is not a unit symbol", k, v)
        pkunit.pkok(pmd_unit(v).unitSI != 0, "{!r}: {!r} has unitSI 0", k, v)


def test_every_archived_numeric_field_has_a_unit():
    import math

    import h5py
    import numpy
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator
    from lume_rftrack.units import unit

    elements = [
        rft.Drift(0.1),
        rft.Quadrupole(0.1, math.nan, 0.0),
        rft.Sextupole(0.1, 1.0),
        rft.Corrector(0.1, 1.0, 2.0),
        rft.SBend(0.2, 0.01, 10.0),
        rft.Screen(),
        rft.Bpm(0.1, 0.0, 0.0),
        rft.Pillbox_Cavity(numpy.array([1e6]), 3e9, 0.05, 2),
        rft.RF_FieldMap_1d(
            (1e6 * numpy.sin(numpy.linspace(0, numpy.pi, 21))).astype(complex),
            0.01,
            0.2,
            1.3e9,
            1.0,
            1.0,
            1.0,
        ),
    ]
    elements[1].set_gradient(1.0)
    lattice = rft.Lattice()
    for i, e in enumerate(elements):
        e.set_name(f"E{i}")
        lattice.append(e)
    path = str(pkunit.empty_work_dir().join("a.h5"))
    RFTrackSimulator(
        lattice,
        rftrack_util.twiss_bunch(rft, 10.0, 10),
    ).archive(path)

    def _needs_unit(cls, k):
        try:
            return unit(cls, k) is not None
        except KeyError:
            return True

    def _numeric(v):
        # numbers, numeric arrays and "@<md5>" references to a
        # deduplicated field map (e.g. RF_FieldMap_1d Ez)
        if isinstance(v, bool | numpy.bool_):
            return False
        if isinstance(v, str):
            return v.startswith("@")
        return numpy.issubdtype(numpy.asarray(v).dtype, numpy.number)

    def _fields(group, units, cls, prefix=""):
        """`group`'s numeric fields that have a unit but no `units` entry"""
        res = set()
        for k, v in group.attrs.items():
            if _numeric(v) and _needs_unit(cls, k) and k not in units:
                res.add(prefix + k)
        for k in group:
            if isinstance(group[k], h5py.Group):
                res |= _fields(group[k], units[k], cls, f"{prefix}{k}/")
            elif _needs_unit(cls, k) and k not in units:
                res.add(prefix + k)
        return res

    with h5py.File(path, "r") as f:
        for i in f["elements"]:
            e = f["elements"][i]
            pkunit.pkeq(
                set(),
                _fields(
                    e,
                    f["units"]["elements"][i],
                    getattr(rft, e.attrs["class"]),
                ),
                "element {} ({}) has fields without a unit",
                i,
                e.attrs["class"],
            )
