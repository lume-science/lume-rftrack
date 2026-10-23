"""lume_rftrack.archive archive()/load_archive() coverage for Corrector.

Corrector(L, Sx, Sy)'s Sx/Sy are an integrated strength in T.mm, not a
kick angle, and get_strength()/set_strength() is the pair that reads and
writes them, so the generic archive needs nothing Corrector-specific:
get_kick(P_over_Q), the equivalent kick angle, takes an argument and is
not archived.
"""


def test_strength_is_archived_directly():
    import numpy
    import RF_Track as rft
    from pykern import pkunit

    from lume_rftrack.archive import _element_to_attrs

    d = _element_to_attrs(rft, rft.Corrector(0.1, 2.0, 3.0))
    numpy.testing.assert_allclose(d.attributes.strength, [2.0, 3.0], rtol=1e-12)
    pkunit.pkeq(0.1, d.attributes.length)
    pkunit.pkok("kick" not in d.attributes, "get_kick() takes an argument")
    pkunit.pkok("constructor" not in d, "a Corrector needs no constructor args")


def test_archive_round_trip():
    import numpy
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("corrector.h5"))

    beamline = rft.Lattice()
    corr = rft.Corrector(0.1, 2.0, 3.0)
    corr.set_name("DC1")
    beamline.append(corr)
    beamline.append(rft.Drift(0.1))

    original = RFTrackSimulator(
        beamline,
        rftrack_util.twiss_bunch(rft, 100.0, 50),
    )
    original.archive(path)

    reloaded = RFTrackSimulator()
    reloaded.load_archive(path)
    numpy.testing.assert_allclose(
        reloaded.beamline[0].get_kick(1.0),
        original.beamline[0].get_kick(1.0),
        rtol=1e-12,
    )

    rftrack_util.assert_same_tracking(original, reloaded)
