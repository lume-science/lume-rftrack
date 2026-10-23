"""lume_rftrack.model.build_model: one variable per CLASS_ATTRIBUTES attribute
of each named element, found by its RF-Track class, with its default read
from the element and its unit from lume_rftrack.units
"""


def test_duplicate_names():
    import RF_Track as rft
    from pykern import pkunit

    import lume_rftrack.model

    lattice, bunch = _beamline(
        rft, [("Q1", rft.Quadrupole(0.1)), ("Q1", rft.Quadrupole(0.1))]
    )
    with pkunit.pkexcept("more than one element named 'Q1'"):
        lume_rftrack.model.build_model(lattice, bunch)


def test_variables():
    import numpy
    import RF_Track as rft
    from pykern import pkunit

    import lume_rftrack.model

    q = rft.Quadrupole(0.1)
    q.set_gradient(3.0)
    sx = rft.Sextupole(0.1)
    sx.set_gradient(40.0)
    cor = rft.Corrector(0.1)
    cor.set_strength(1.5, -0.5)
    fm = rft.RF_FieldMap_1d(
        (1e6 * numpy.sin(numpy.linspace(0, numpy.pi, 21))).astype(complex),
        0.01,
        0.2,
        1.3e9,
        1.0,
        1.0,
        1.0,
    )
    fm.set_phid(7.0)
    # an RF_Field without P_actual: phid only
    pb = rft.Pillbox_Cavity(numpy.array([1e6]), 3e9, 0.05, 2)
    pb.set_phid(12.0)
    sol = rft.Solenoid(0.1, 0.5, 0.05)
    lattice, bunch = _beamline(
        rft,
        [
            ("D1", rft.Drift(0.1)),
            ("Q", q),
            ("SX", sx),
            ("COR", cor),
            ("FM", fm),
            ("PB", pb),
            ("SOL", sol),
            ("B1", rft.SBend(0.2, 0.01, 100.0)),
            ("SCR", rft.Screen()),
            # no name: no variables
            ("", rft.Quadrupole(0.1)),
        ],
    )
    variables = lume_rftrack.model.build_model(
        lattice, bunch
    ).supported_variables.items()
    expect = [
        ("Q_gradient", "gradient", 3.0, "T/m"),
        ("SX_gradient", "gradient", 40.0, "T/m^2"),
        ("COR_strength_x", "strength", 1.5, "T*mm"),
        ("COR_strength_y", "strength", -0.5, "T*mm"),
        ("FM_phid", "phid", 7.0, "deg"),
        ("FM_P_actual", "P_actual", 1.0, "W"),
        ("PB_phid", "phid", 12.0, "deg"),
        ("SOL_max_current", "max_current", sol.get_max_current(), "A"),
        ("B1_angle", "angle", 0.01, "rad"),
    ]
    pkunit.pkeq(
        [(n, a, u) for n, a, _, u in expect],
        [(n, v.attribute, v.unit) for n, v in variables],
    )
    # RF-Track's unit conversions are not exact, e.g. the corrector's
    # vertical strength reads back as -0.4999999999999999
    numpy.testing.assert_allclose(
        [d for _, _, d, _ in expect], [v.default_value for _, v in variables]
    )


def test_vector_component_set():
    """Setting one component of a corrector's strength keeps the other."""
    import numpy
    import RF_Track as rft

    import lume_rftrack.model

    lattice, bunch = _beamline(rft, [("COR", rft.Corrector(0.1, 1.5, -0.5))])
    m = lume_rftrack.model.build_model(lattice, bunch)
    m.set({"COR_strength_y": 2.0})
    numpy.testing.assert_allclose(
        m.simulator.find_element("COR").get_strength(), [1.5, 2.0]
    )


def _beamline(rft, name_element):
    """A Lattice of the elements, and a one-particle bunch."""
    import numpy

    lattice = rft.Lattice()
    for name, e in name_element:
        e.set_name(name)
        lattice.append(e)
    return (
        lattice,
        rft.Bunch6d(
            numpy.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 100.0, rft.electronmass, -1.0, 1.0, 1]]
            )
        ),
    )
