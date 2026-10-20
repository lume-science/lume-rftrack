"""lume_rftrack.simulator.RFTrackSimulator.archive()/load_archive() round-trip tests
for a Volume-mode beamline with an RF_FieldMap_1d cavity and a Screen --
the element classes and absolute (not sequential) positioning a Volume
beamline of field-map cavities/solenoids, correctors, and screens needs.
Static_Magnetic_FieldMap_1d (field-map SOLENOID) is covered too (see
static_magnetic_fieldmap_1d_test.py); so is Corrector (see
corrector_test.py).

The cavity is archived two ways. Before track() it keeps its construction
phid and Volume.track() autophases the copy, the same way for both. After
autophase() it has a t0, which get_phid() never reflects (confirmed
empirically: it always reads back the phid the element was constructed
with), so t0 is the only way its actual effective phase survives the
round trip.
"""


def test_t0_archive_round_trip():
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("archive.h5"))
    volume = rftrack_util.volume_with_cavity(rft)
    cavity = volume[0]
    # autophase for the bunch itself, as Volume.track() does: a single
    # electron-like reference particle built by hand was autophased to a
    # decelerating phase here
    bunch = rftrack_util.twiss_bunch(rft, 10.0, 200)
    volume.unset_t0()
    volume.autophase(bunch)
    pkunit.pkok(cavity.t0_is_set(), "expected autophase() to set t0")
    original_t0 = cavity.get_t0()
    # get_phid() never reflects autophase() -- still the construction
    # value, not the real effective phase
    pkunit.pkeq(10.0, cavity.get_phid())

    original = RFTrackSimulator(volume, bunch)
    original.archive(path)
    original.track()

    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)
    pkunit.pkok(reloaded.beamline[0].t0_is_set(), "expected t0 to survive the archive")
    pkunit.pkeq(original_t0, reloaded.beamline[0].get_t0())
    # archive() was never told autophase=True explicitly -- it's
    # inferred from the cavity's own nonzero archived t0.
    pkunit.pkeq(True, reloaded.autophase)
    # no autophase() was called on the reloaded copy at all -- t0 alone
    # reproduces the same physics
    rftrack_util.assert_same_tracking(original, reloaded)
    _assert_near_best_gain(original, bunch)


def test_volume_archive_round_trip():
    import h5py
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("volume.h5"))

    original = RFTrackSimulator(
        rftrack_util.volume_with_cavity(rft), rftrack_util.twiss_bunch(rft, 10.0, 200)
    )
    # archive before track(): the cavity has no t0 yet, so the archive holds
    # only its construction phid, and both copies autophase on their first
    # track(). Archiving after track() would also store the autophased t0
    # (see test_t0_archive_round_trip).
    original.archive(path)

    with h5py.File(path, "r") as f:
        pkunit.pkeq("Volume", f.attrs["beamline_class"])
        pkunit.pkeq("RF_FieldMap_1d", f["elements"]["0"].attrs["class"])
        pkunit.pkeq("Screen", f["elements"]["1"].attrs["class"])
        # the field map array went in as a real dataset under the
        # shared "field_maps" group, referenced by the element's own
        # "ez" attr ("@<md5>") rather than duplicated inline -- see
        # lume_rftrack.archive.archive()'s own docstring (_dedupe_field_maps)
        pkunit.pkeq(21, len(f["field_maps"][f["elements"]["0"].attrs["ez"][1:]]))

    pkunit.pkeq(
        10.0, original.beamline[0].get_phid(), "archive() must not mutate beamline"
    )

    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)

    pkunit.pkeq("Volume", type(reloaded.beamline).__name__)
    pkunit.pkeq(
        original.beamline[0].get_length(),
        reloaded.beamline[0].get_length(),
    )
    pkunit.pkeq(10.0, reloaded.beamline[0].get_phid())

    rftrack_util.assert_same_tracking(original, reloaded)
    _assert_near_best_gain(original, original.initial_bunch)


def _assert_near_best_gain(simulator, bunch):
    """The tracked cavity is near its best phase: the gain is at least 90%
    of the largest gain a scan of the cavity's t0 over one RF period finds
    (1.3 GHz: 230.6 mm/c, in 10 mm/c steps), and less than the on-axis
    integral of the field (2 / pi * 1 MV/m * 0.2 m = 0.127 MeV)."""
    from pykern import pkunit

    gain = _gain(simulator)
    best = _best_gain(bunch)
    pkunit.pkok(
        0.9 * best <= gain < 0.127,
        "gain {} MeV, expected at least 90% of the scanned best {}",
        gain,
        best,
    )


def _best_gain(bunch):
    """The largest mean kinetic energy gain over a scan of the cavity's t0
    across one RF period (1.3 GHz: 230.6 mm/c), in 10 mm/c steps."""
    import numpy
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.simulator import RFTrackSimulator

    def _scan(t0):
        v = rftrack_util.volume_with_cavity(rft)
        v[0].set_t0(t0)
        v[1].set_t0(0.0)
        s = RFTrackSimulator(v, bunch)
        s.track()
        return _gain(s)

    return max(_scan(t0) for t0 in numpy.arange(0.0, 230.6, 10.0))


def _gain(simulator):
    k = "%K"
    return (
        simulator.tracked_bunch.get_phase_space(k).mean()
        - simulator.initial_bunch.get_phase_space(k).mean()
    )
