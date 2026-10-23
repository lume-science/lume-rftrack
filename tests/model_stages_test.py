"""lume_rftrack.model.LUMERFTrackModel as a lume.staged_model.StagedModel
stage: a beamline split into stages tracks like the whole one, a stage
with no bunch waits for one, and a Volume stage takes a Lattice's beam
at its entrance.
"""


def test_lattice_stages_match_one_lattice():
    import numpy
    import RF_Track as rft
    import rftrack_util
    from lume.staged_model import StagedModel

    import lume_rftrack.model
    import lume_rftrack.utils

    bunch = rftrack_util.twiss_bunch(rft, 100.0, 500)
    staged = StagedModel(
        [
            lume_rftrack.model.build_model(_lattice(rft, "Q1", 5.0, "D1"), bunch),
            lume_rftrack.model.build_model(_lattice(rft, "Q2", -5.0, "D2"), None),
        ]
    )
    whole = rft.Lattice()
    for n, g, d in (("Q1", 5.0, "D1"), ("Q2", -5.0, "D2")):
        whole.append(_lattice(rft, n, g, d))
    values = {"Q1_gradient": 8.0, "Q2_gradient": -3.0}
    staged.set(values)
    for n, v in values.items():
        whole.get_elements_by_name(n[:2]).set_gradient(v)
    f = "%x %xp %y %yp %t %P"
    numpy.testing.assert_allclose(
        lume_rftrack.utils.particlegroup_to_bunch6d(
            staged.lume_model_instances[1].final_particles
        ).get_phase_space(f),
        whole.track(bunch).get_phase_space(f),
        rtol=1e-9,
        atol=1e-9,
    )


def test_stage_without_bunch_waits():
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    import lume_rftrack.model
    import lume_rftrack.utils

    m = lume_rftrack.model.build_model(_lattice(rft, "Q1", 5.0, "D1"), None)
    m.reset()
    pkunit.pkeq(None, m.initial_particles)
    pkunit.pkeq(None, m.final_particles)
    m.initial_particles = lume_rftrack.utils.bunch6d_to_particlegroup(
        rftrack_util.twiss_bunch(rft, 100.0, 200)
    )
    pkunit.pkeq(200, len(m.final_particles))


def test_volume_stage_starts_at_its_entrance():
    import numpy
    import RF_Track as rft
    import rftrack_util
    from lume.staged_model import StagedModel
    from pykern import pkunit

    import lume_rftrack.model

    v = rftrack_util.volume_with_cavity(rft)
    staged = StagedModel(
        [
            lume_rftrack.model.build_model(
                _lattice(rft, "Q1", 5.0, "D1"),
                rftrack_util.twiss_bunch(rft, 100.0, 200),
            ),
            lume_rftrack.model.build_model(v, None),
        ]
    )
    staged.set({"Q1_gradient": 4.0})
    s = staged.lume_model_instances[1]
    pkunit.pkok(
        isinstance(s.simulator.initial_bunch, rft.Bunch6dT),
        "Volume stage got {}",
        type(s.simulator.initial_bunch),
    )
    numpy.testing.assert_array_equal(s.initial_particles.z, 0.0)
    p = s.final_particles
    pkunit.pkeq(200, len(p))
    pkunit.pkok(
        numpy.all(p.z > v.get_elements_by_name("C1").get_length()),
        "particles did not get through the cavity: min z {}",
        p.z.min(),
    )


def _lattice(rft, quadrupole, gradient, drift):
    """A Lattice of a 0.1 m quadrupole and a 1 m drift"""
    q = rft.Quadrupole(0.1)
    q.set_gradient(gradient)
    q.set_name(quadrupole)
    d = rft.Drift(1.0)
    d.set_name(drift)
    res = rft.Lattice()
    res.append(q)
    res.append(d)
    return res
