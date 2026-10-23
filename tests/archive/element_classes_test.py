"""archive()/load_archive() of RF-Track element classes with no
lume_rftrack-specific code (Sextupole, Multipole, the _CINT field maps) or
only their constructor arguments (SBend, Solenoid, Coil,
Static_Magnetic_FieldMap_2d), and refusal of one that wouldn't come back
the same.
"""


def test_lattice_round_trip():
    import numpy
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.simulator import RFTrackSimulator

    s = rft.Sextupole(0.1)
    s.set_gradient(40.0)
    m = rft.Multipole(0.1)
    m.set_Bn(numpy.array([0, 0.5, 3.0], dtype=complex))
    z = numpy.linspace(0, 0.1, 41)
    c = rft.RF_FieldMap_1d_CINT(
        (20e6 * numpy.sin(numpy.pi * z / 0.1)).astype(complex),
        z[1] - z[0],
        0.1,
        3e9,
        1,
    )
    c.set_phid(-5.0)
    _round_trip(
        _lattice(
            rft,
            [
                s,
                m,
                rft.SBend(0.3, 0.05, -100.0, 0.01, 0.02),
                rft.Solenoid(0.2, 0.3, 0.05),
                c,
            ],
        ),
        rftrack_util.twiss_bunch(rft, 100.0, 50),
        RFTrackSimulator,
    )


def test_refuses_what_it_cannot_rebuild():
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    for e, expect in (
        # its number of current sheets has no getter
        (rft.Solenoid(0.2, 0.3, 0.04, 0.06, 3), "thick winding"),
        # its material has no getter, and set_shape() takes more than
        # get_shape() returns
        (rft.Absorber(0.01, "water"), "can't restore Absorber.shape"),
    ):
        with pkunit.pkexcept(expect):
            RFTrackSimulator(
                _lattice(rft, [e]), rftrack_util.twiss_bunch(rft, 100.0, 10)
            ).archive(str(pkunit.empty_work_dir().join("a.h5")))


def test_volume_round_trip():
    import numpy
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.simulator import RFTrackSimulator

    m = rft.Static_Magnetic_FieldMap_2d(
        numpy.outer(numpy.linspace(-1, 1, 26), numpy.linspace(0, 0.01, 5)),
        numpy.outer(0.1 * numpy.exp(-(numpy.linspace(-2, 2, 26) ** 2)), numpy.ones(5)),
        0.002,
        0.004,
    )
    m.set_scaling_factor(1.5)
    v = rft.Volume()
    z = 0.0
    for i, e in enumerate(
        (
            rft.Coil(0.2, 0.1, 0.05),
            rft.Solenoid(0.2, 0.3, 0.05),
            m,
            rft.Sextupole(0.1, 30.0),
        )
    ):
        e.set_name(f"E{i}")
        v.add(e, 0.0, 0.0, z, 0.0, 0.0, 0.0)
        z += e.get_length()
    v.set_s0(0.0)
    v.set_s1(z)
    _round_trip(
        v,
        rft.Bunch6dT(rftrack_util.twiss_bunch(rft, 10.0, 50)),
        RFTrackSimulator,
        fields="%X %xp %Y %yp %P",
    )


def _lattice(rft, elements):
    res = rft.Lattice()
    for i, e in enumerate(elements):
        e.set_name(f"E{i}")
        res.append(e)
    return res


def _round_trip(beamline, bunch, simulator_class, fields="%x %xp %y %yp %t %P"):
    """Archive, reload, and require the same classes, names and tracking"""
    import numpy
    from pykern import pkunit

    path = str(pkunit.empty_work_dir().join("a.h5"))
    original = simulator_class(beamline, bunch)
    original.archive(path)
    reloaded = simulator_class()
    reloaded.load_archive(path)
    pkunit.pkeq(
        [
            (type(original.beamline[i]).__name__, original.beamline[i].get_name())
            for i in range(original.beamline.size())
        ],
        [
            (type(reloaded.beamline[i]).__name__, reloaded.beamline[i].get_name())
            for i in range(reloaded.beamline.size())
        ],
    )
    original.track()
    reloaded.track()
    pkunit.pkok(original.tracked_bunch.get_ngood() > 0, "every particle was lost")
    numpy.testing.assert_allclose(
        reloaded.tracked_bunch.get_phase_space(fields),
        original.tracked_bunch.get_phase_space(fields),
        rtol=1e-8,
        atol=1e-8,
    )
