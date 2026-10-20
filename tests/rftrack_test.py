"""lume_rftrack.rftrack tests"""


def _build_model():
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.actions import (
        RFTrackReadOnlyScalarVariable,
        RFTrackWritableScalarVariable,
    )
    from lume_rftrack.rftrack import LUMERFTrackModel
    from lume_rftrack.simulator import RFTrackSimulator

    q = rft.Quadrupole(0.1, 5.0)
    q.set_name("Q1")
    d = rft.Drift(0.2)
    d.set_name("D1")
    beamline = rft.Lattice()
    beamline.append(q)
    beamline.append(d)

    return LUMERFTrackModel(
        simulator=RFTrackSimulator(
            beamline,
            rftrack_util.twiss_bunch(rft, 100.0, 100, charge_nc=-1, beta=5.0),
        ),
        action_variables=[
            RFTrackWritableScalarVariable(
                name="Q1_strength",
                element_name="Q1",
                attribute="strength",
                default_value=5.0,
            ),
            RFTrackReadOnlyScalarVariable(
                name="Q1_strength_ro",
                element_name="Q1",
                attribute="strength",
                read_only=True,
            ),
        ],
    )


def test_set_and_get():
    from pykern import pkunit

    m = _build_model()
    pkunit.pkeq(5.0, m.get("Q1_strength"))
    m.set({"Q1_strength": 500.0})
    pkunit.pkeq(500.0, m.get("Q1_strength"))
    pkunit.pkeq(500.0, m.get("Q1_strength_ro"))


def test_read_only_raises():
    from lume.exceptions import ReadOnlyError
    from pykern import pkunit

    with pkunit.pkexcept(ReadOnlyError):
        _build_model().set({"Q1_strength_ro": 1.0})


def _build_corrector_model():
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.actions import RFTrackWritableVectorComponentVariable
    from lume_rftrack.rftrack import LUMERFTrackModel
    from lume_rftrack.simulator import RFTrackSimulator

    c = rft.Corrector(0.1)
    c.set_name("C1")
    beamline = rft.Lattice()
    beamline.append(c)

    return LUMERFTrackModel(
        simulator=RFTrackSimulator(
            beamline,
            rftrack_util.twiss_bunch(rft, 100.0, 100, charge_nc=-1, beta=5.0),
        ),
        action_variables=[
            RFTrackWritableVectorComponentVariable(
                name="C1_hkick",
                element_name="C1",
                attribute="strength",
                index=0,
                default_value=0.0,
            ),
            RFTrackWritableVectorComponentVariable(
                name="C1_vkick",
                element_name="C1",
                attribute="strength",
                index=1,
                default_value=0.0,
            ),
        ],
    )


def _pkeq_approx(expect, actual):
    # RF-Track's internal unit conversions round-trip a kick through
    # floating point, e.g. 1.0 -> 0.9999999999999998 -- not a bug, just
    # not exact equality.
    import math

    from pykern import pkunit

    pkunit.pkok(
        math.isclose(expect, actual, rel_tol=1e-9, abs_tol=1e-9),
        "expect={} != actual={}",
        expect,
        actual,
    )


def test_vector_component_preserves_other_index():
    m = _build_corrector_model()
    m.set({"C1_hkick": 1.0})
    _pkeq_approx(1.0, m.get("C1_hkick"))
    _pkeq_approx(0.0, m.get("C1_vkick"))
    strength = list(m.simulator.find_element("C1").get_strength())
    _pkeq_approx(1.0, strength[0])
    _pkeq_approx(0.0, strength[1])

    m.set({"C1_vkick": 2.0})
    _pkeq_approx(1.0, m.get("C1_hkick"))
    _pkeq_approx(2.0, m.get("C1_vkick"))
    strength = list(m.simulator.find_element("C1").get_strength())
    _pkeq_approx(1.0, strength[0])
    _pkeq_approx(2.0, strength[1])


def test_vector_component_set_and_get():
    from pykern import pkunit

    m = _build_corrector_model()
    pkunit.pkeq(0.0, m.get("C1_hkick"))
    pkunit.pkeq(0.0, m.get("C1_vkick"))
    m.set({"C1_hkick": 3.0, "C1_vkick": 4.0})
    _pkeq_approx(3.0, m.get("C1_hkick"))
    _pkeq_approx(4.0, m.get("C1_vkick"))
