"""lume_rftrack.archive's `output` group: a tracked simulator's bunch
comes back as its `tracked_bunch` (a Volume's `Bunch6dT` with its clock),
and an untracked simulator's archive has no output.
"""


def test_lattice_tracked_bunch_round_trip():
    import numpy
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("archive.h5"))
    original = RFTrackSimulator(
        rftrack_util.fodo_lattice(rft), rftrack_util.twiss_bunch(rft, 100.0, 200)
    )
    original.track()
    original.archive(path)
    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)
    f = "%x %xp %y %yp %t %P"
    numpy.testing.assert_allclose(
        reloaded.tracked_bunch.get_phase_space(f),
        original.tracked_bunch.get_phase_space(f),
        rtol=1e-12,
        atol=1e-12,
    )


def test_untracked_has_no_output():
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.archive import read_archive
    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("archive.h5"))
    RFTrackSimulator(
        rftrack_util.fodo_lattice(rft), rftrack_util.twiss_bunch(rft, 100.0, 20)
    ).archive(path)
    pkunit.pkeq(None, read_archive(path).output)
    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)
    pkunit.pkeq(None, reloaded.tracked_bunch)


def test_volume_tracked_bunch_keeps_its_clock():
    """A tracked `Bunch6dT`'s particles don't carry its clock `t`, which
    `final_particles` reads; the archive records it."""
    import numpy
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    import lume_rftrack.model
    import lume_rftrack.utils
    from lume_rftrack.archive import read_archive

    path = str(pkunit.empty_work_dir().join("archive.h5"))
    original = lume_rftrack.model.build_model(
        rftrack_util.volume_with_cavity(rft),
        lume_rftrack.utils.particlegroup_to_bunch6dt(
            lume_rftrack.utils.bunch6d_to_particlegroup(
                rftrack_util.twiss_bunch(rft, 10.0, 200)
            )
        ),
    )
    original.reset()
    original.simulator.archive(path)
    pkunit.pkeq(
        original.simulator.tracked_bunch.t, read_archive(path).output.bunch_t_mm
    )
    reloaded = lume_rftrack.model.build_model(rft.Volume(), None)
    reloaded.simulator.load_archive(path)
    f = "%X %Y %Z %Px %Py %Pz %t0 %t"
    numpy.testing.assert_allclose(
        reloaded.simulator.tracked_bunch.get_phase_space(f),
        original.simulator.tracked_bunch.get_phase_space(f),
        rtol=1e-12,
        atol=1e-12,
    )
    a = original.final_particles
    b = reloaded.final_particles
    for k in ("x", "y", "z", "px", "py", "pz", "t", "weight"):
        numpy.testing.assert_allclose(getattr(b, k), getattr(a, k), rtol=1e-12)
