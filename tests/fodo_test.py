"""lume_rftrack.rftrack test against a realistic lattice: an X-band linac FODO
cell, built directly from RF_Track's native API
(one FODO period: D0, C1, C2, QF, C3, C4, QD; analytic standing-wave
`RF_FieldMap_1d` cavities, space charge off, so this has no external
data-file dependency).
"""

_FIELDS = "%x %xp %y %yp %t %P"

_MODEL = None


def test_cavity_phase_scan():
    import math

    import numpy
    from pykern import pkunit

    m = _model()
    k0 = m.simulator.initial_bunch.get_phase_space("%K").mean()

    def _gain():
        return m.simulator.tracked_bunch.get_phase_space("%K").mean() - k0

    crest = _gain()
    pkunit.pkok(crest > 1.0, "expected energy gain on crest, got {} MeV", crest)
    for phid in (60.0, 90.0, 180.0):
        for c in ("C1", "C2", "C3", "C4"):
            m.set({c + "_phid": phid})
        numpy.testing.assert_allclose(
            _gain(), crest * math.cos(math.radians(phid)), atol=0.02 * crest
        )


def test_reset_restores_fodo_focusing():
    from pykern import pkunit

    m = _model()
    m.set({"QF_gradient": 5.0, "QD_gradient": -5.0})
    sigma_x_fodo = m.simulator.tracked_bunch.get_phase_space("%x").std()

    m.set({"QF_gradient": 0.0, "QD_gradient": 0.0})
    m.reset()

    pkunit.pkeq(5.0, m.get("QF_gradient"))
    pkunit.pkeq(-5.0, m.get("QD_gradient"))
    pkunit.pkeq(sigma_x_fodo, m.simulator.tracked_bunch.get_phase_space("%x").std())


def test_set_matches_direct_change():
    """set() re-tracks the beamline: the result is the same as changing the
    elements directly and tracking again, and the change does something."""
    import numpy
    from pykern import pkunit

    for values, change in (
        (
            {"QF_gradient": 0.0, "QD_gradient": 0.0},
            lambda s: (
                s.find_element("QF").set_gradient(0.0),
                s.find_element("QD").set_gradient(0.0),
            ),
        ),
        ({"C1_phid": 90.0}, lambda s: s.find_element("C1").set_phid(90.0)),
    ):
        m = _model()
        default = m.simulator.tracked_bunch.get_phase_space(_FIELDS)
        m.set(values)
        actual = m.simulator.tracked_bunch.get_phase_space(_FIELDS)
        numpy.testing.assert_array_equal(actual, _changed(change))
        pkunit.pkok(not numpy.array_equal(default, actual), "{} had no effect", values)


def _build_model():
    from lume_rftrack.rftrack import build_model

    s = _simulator()
    return build_model(
        [
            dict(kind="quadrupole_gradient", name="QF"),
            dict(kind="quadrupole_gradient", name="QD"),
        ]
        + [dict(kind="cavity_fieldmap", name=c) for c in ("C1", "C2", "C3", "C4")],
        s.beamline,
        s.initial_bunch,
    )


def _changed(change):
    """A fresh simulator's tracked phase space after `change(simulator)`,
    made the way the model makes it: track once with the defaults, change
    the elements directly, track again."""
    s = _simulator()
    s.track()
    change(s)
    s.track()
    return s.tracked_bunch.get_phase_space(_FIELDS)


def _model():
    """One model shared by every test here, reset to its defaults first.
    Building the model once keeps the tests quick."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    _MODEL.reset()
    return _MODEL


def _simulator():
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.simulator import RFTrackSimulator

    return RFTrackSimulator(
        rftrack_util.fodo_lattice(rft),
        rftrack_util.twiss_bunch(rft, 100.0, 100, emitt=5.0),
    )
