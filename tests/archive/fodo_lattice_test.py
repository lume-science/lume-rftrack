"""lume_rftrack.simulator.RFTrackSimulator.archive()/load_archive() round-trip test,
against the FODO lattice used by tests/fodo_test.py.
"""


def test_archive_round_trip():
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("archive.h5"))

    original = RFTrackSimulator(
        rftrack_util.fodo_lattice(rft),
        rftrack_util.twiss_bunch(rft, 100.0, 100, emitt=5.0),
    )
    original.track()
    original.archive(path)

    # the archive is plain, self-describing HDF5 -- not something that
    # requires importing this module to read back (that's the whole point:
    # it's the interchange contract with anything that reads it)
    import importlib.metadata

    import h5py

    with h5py.File(path, "r") as f:
        # also a real standalone openPMD particle file (archive()'s own
        # docstring) -- "dataType" is openPMD's own root attr, not ours.
        pkunit.pkeq(b"openPMD", f.attrs["dataType"])
        pkunit.pkeq("lume_rftrack", f.attrs["software"])
        pkunit.pkeq(rft.RF_TRACK_VERSION, f.attrs["rftrack_version"])
        pkunit.pkeq(
            importlib.metadata.version("lume-rftrack"), f.attrs["lume_rftrack_version"]
        )
        pkunit.pkeq("Bunch6d", f.attrs["bunch_type"])
        pkunit.pkeq("Drift", f["elements"]["0"].attrs["class"])
        pkunit.pkeq("D0", f["elements"]["0"].attrs["name"])
        pkunit.pkeq("RF_FieldMap_1d", f["elements"]["1"].attrs["class"])
        pkunit.pkok("particles" in f, "expected a particles group")

    # load_archive() fills in an otherwise-empty simulator, same calling
    # convention as lume.base.Base's in general
    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)

    pkunit.pkeq(
        original.beamline.size(), reloaded.beamline.size(), "element count differs"
    )
    for i in range(original.beamline.size()):
        pkunit.pkeq(
            original.beamline[i].get_name(),
            reloaded.beamline[i].get_name(),
            "element {} name differs",
            i,
        )

    rftrack_util.assert_same_tracking(original, reloaded)


def test_versions():
    import importlib.metadata

    import h5py
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.archive import read_archive
    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("version.h5"))
    RFTrackSimulator(
        rftrack_util.fodo_lattice(rft), rftrack_util.twiss_bunch(rft, 100.0, 10)
    ).archive(path)
    a = read_archive(path)
    pkunit.pkeq(rft.RF_TRACK_VERSION, a.rftrack_version)
    pkunit.pkeq(importlib.metadata.version("lume-rftrack"), a.lume_rftrack_version)
    # an archive from before the versions were recorded
    with h5py.File(path, "a") as f:
        del f.attrs["rftrack_version"]
        del f.attrs["lume_rftrack_version"]
    a = read_archive(path)
    pkunit.pkeq(None, a.rftrack_version)
    pkunit.pkeq(None, a.lume_rftrack_version)
