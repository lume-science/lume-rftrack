"""lume_rftrack.simulator.RFTrackSimulator.archive()/load_archive() round-trip test
for a Bunch6dT initial bunch (e.g. a Cathode Emission distribution),
distinct from fodo_lattice_test.py's Bunch6d coverage.
"""


def _build_bunch(rft):
    t = rft.Bunch6dT_twiss()
    t.beta_x = 0.01
    t.alpha_x = 0.0
    t.emitt_x = 1e-6
    t.beta_y = 0.01
    t.alpha_y = 0.0
    t.emitt_y = 1e-6
    t.beta_z = 0.01
    t.alpha_z = 0.0
    t.emitt_z = 1e-6
    t.sigma_z = 0.1
    t.sigma_pz = 1.0
    # the precise electron mass (not a truncated 0.511): archive() now
    # converts through a beamphysics ParticleGroup, which needs this to
    # match beamphysics' own electron mass constant closely enough for
    # species inference (lume_rftrack.utils._infer_particlegroup_species).
    return rft.Bunch6dT(0.51099895069, 1e9, -1, 0.5, t, 10000)


def _build_volume(rft):
    d = rft.Drift(0.1)
    d.set_name("D0")
    v = rft.Volume()
    v.add(d, 0, 0, 0, 0, 0, 0)
    v.set_s0(0)
    v.set_s1(0.1)
    return v


def test_bunch6dt_archive_round_trip():
    import numpy
    import RF_Track as rft
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("archive.h5"))

    original = RFTrackSimulator(_build_volume(rft), _build_bunch(rft))
    original.archive(path)

    # the archive is plain, self-describing HDF5 -- not something that
    # requires importing this module to read back. It's also a real
    # standalone openPMD particle file (archive()'s own docstring).
    import h5py

    with h5py.File(path, "r") as f:
        pkunit.pkeq("lume_rftrack", f.attrs["software"])
        pkunit.pkeq("Bunch6dT", f.attrs["bunch_type"])
        pkunit.pkeq(b"openPMD", f.attrs["dataType"])
        pkunit.pkok("particles" in f, "expected a particles group")

    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)
    pkunit.pkeq("Bunch6dT", type(reloaded.initial_bunch).__name__)

    # The exact same macroparticles come back (see archive()'s own
    # docstring) -- compare the real phase space, not just statistics.
    f = "%X %Y %Z %Px %Py %Pz %t0"
    numpy.testing.assert_allclose(
        reloaded.initial_bunch.get_phase_space(f),
        original.initial_bunch.get_phase_space(f),
        rtol=1e-12,
        atol=1e-12,
    )
