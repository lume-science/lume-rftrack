"""archive() of collective effects, which RF-Track can't report the
parameters of: dropped, with a warning naming each one.
"""


def test_dropped_with_warning():
    import warnings

    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    def _wake():
        return rft.ShortRangeWakefield(0.00275, 0.00696, 0.0083)

    lattice = rft.Lattice()
    for n, effects in (("D1", 0), ("D2", 1), ("D3", 2)):
        d = rft.Drift(0.5)
        d.set_name(n)
        for _ in range(effects):
            d.add_collective_effect(_wake())
        lattice.append(d)
    volume = rft.Volume()
    volume.add_collective_effect(_wake())
    d = rft.Drift(0.5)
    d.set_name("V1")
    volume.add(d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    volume.set_s0(0.0)
    volume.set_s1(0.5)
    path = str(pkunit.empty_work_dir().join("a.h5"))
    for beamline, bunch, expect in (
        (
            lattice,
            rftrack_util.twiss_bunch(rft, 100.0, 10),
            [
                "element 1 'D2': ShortRangeWakefield",
                "element 2 'D3': ShortRangeWakefield; element 2 'D3': ShortRangeWakefield",
            ],
        ),
        (
            volume,
            rft.Bunch6dT(rftrack_util.twiss_bunch(rft, 100.0, 10)),
            ["Volume: ShortRangeWakefield"],
        ),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RFTrackSimulator(beamline, bunch).archive(path)
        w = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        pkunit.pkeq(1, len(w), "warnings={}", w)
        for e in expect:
            pkunit.pkok(e in w[0], "expect={} in warning={}", e, w[0])
        reloaded = RFTrackSimulator()
        reloaded.load_archive(path)
        pkunit.pkeq(
            [[]] * reloaded.beamline.size(),
            [
                reloaded.beamline[i].get_collective_effects()
                for i in range(reloaded.beamline.size())
            ],
        )


def test_none_no_warning():
    import warnings

    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    lattice = rft.Lattice()
    d = rft.Drift(0.5)
    d.set_name("D1")
    lattice.append(d)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        RFTrackSimulator(lattice, rftrack_util.twiss_bunch(rft, 100.0, 10)).archive(
            str(pkunit.empty_work_dir().join("a.h5"))
        )
