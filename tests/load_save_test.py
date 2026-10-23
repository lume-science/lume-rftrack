"""lume_rftrack.utils field-map readers and result writers"""


def test_load_field_map():
    import numpy
    from pykern import pkunit

    from lume_rftrack.utils import load_field_map

    f = str(pkunit.empty_work_dir().join("map.txt"))
    numpy.savetxt(f, numpy.column_stack([[0.0, 0.5, 1.0], [1.0, -4.0, 2.0]]))
    field, step, length = load_field_map(
        dict(file=f, rescale_mode="factor", scale_factor=2.0)
    )
    numpy.testing.assert_allclose(field, [2.0, -8.0, 4.0])
    pkunit.pkeq(0.5, step)
    pkunit.pkeq(1.0, length)
    field, _, _ = load_field_map(dict(file=f, rescale_mode="peak", target_field=1.0))
    numpy.testing.assert_allclose(field, [0.25, -1.0, 0.5])
    with pkunit.pkexcept("unknown rescale_mode"):
        load_field_map(dict(file=f, rescale_mode="target", target_field=1.0))


def test_load_field_map_2d():
    import numpy
    from pykern import pkunit

    from lume_rftrack.utils import load_field_map_2d

    f = str(pkunit.empty_work_dir().join("map2d.txt"))
    # Z [cm], R [cm], Ez, Er, |E|, H -- Z varies fastest
    rows = [
        [z, r, 10.0 * z + r, 100.0 * z + r, 0.0, 0.0]
        for r in (0.0, 1.0)
        for z in (0.0, 1.0, 2.0)
    ]
    numpy.savetxt(f, numpy.array(rows))
    er, ez, hr, hz = load_field_map_2d(dict(file=f))
    pkunit.pkeq((3, 2), ez.shape)
    pkunit.pkeq(0.01, hz)
    pkunit.pkeq(0.01, hr)
    numpy.testing.assert_allclose(ez[2, 1], 21.0e6)
    numpy.testing.assert_allclose(er[1, 0], 100.0e6)


def test_load_field_map_2d_written():
    """write_field_map_2d() is the inverse of load_field_map_2d(), and the
    map it writes builds an RF_FieldMap_2d with the same field."""
    import numpy
    import RF_Track as rft
    from pykern import pkunit

    from lume_rftrack.utils import load_field_map_2d, write_field_map_2d

    nz, nr, hz, hr = 21, 5, 0.005, 0.002
    z = numpy.arange(nz) * hz
    r = numpy.arange(nr) * hr
    ez = numpy.outer(20e6 * numpy.sin(numpy.pi * z / z[-1]), numpy.ones(nr))
    er = numpy.outer(1e6 * numpy.cos(numpy.pi * z / z[-1]), r / r[-1])
    f = str(pkunit.empty_work_dir().join("map2d.txt"))
    write_field_map_2d(f, er, ez, hr, hz)
    a = load_field_map_2d(dict(file=f))
    for e, x in zip((er, ez, hr, hz), a):
        numpy.testing.assert_allclose(x, e, rtol=1e-12)
    p = (1.0, 0.5, 40.0, 0.0)
    numpy.testing.assert_allclose(
        rft.RF_FieldMap_2d(*a[:2], 0, 0, *a[2:], -1, 3e9, +1).get_field(*p)[0],
        rft.RF_FieldMap_2d(er, ez, 0, 0, hr, hz, -1, 3e9, +1).get_field(*p)[0],
        rtol=1e-12,
    )


def test_save_particles():
    import numpy
    import RF_Track as rft
    import scipy.constants
    from pykern import pkunit

    from lume_rftrack.utils import save_particles

    with pkunit.save_chdir_work():
        save_particles(
            "p.npy",
            rft.Bunch6d(
                numpy.array(
                    [[1.0, 2.0, 3.0, 4.0, 0.0, 5.0, rft.electronmass, -1.0, 1.0, 1]]
                )
            ),
        )
        a = numpy.load("p.npy")
        pkunit.pkeq((1, 13), a.shape)
        numpy.testing.assert_allclose(a[0, 0], 0.001)
        numpy.testing.assert_allclose(a[0, 2], 0.003)
        numpy.testing.assert_allclose(a[0, 9], 5.0e6, rtol=1e-6)
        # one macroparticle of one electron
        numpy.testing.assert_allclose(a[0, 12], scipy.constants.e)


def test_save_particles_bunch6dt():
    import numpy
    import RF_Track as rft
    import scipy.constants
    from pykern import pkunit

    from lume_rftrack.utils import save_particles

    with pkunit.save_chdir_work():
        save_particles(
            "p.npy",
            rft.Bunch6dT(
                numpy.array(
                    [[1.0, 2.0, 3.0, 4.0, 0.0, 5.0, rft.electronmass, -1.0, 1.0, 0.0]]
                )
            ),
        )
        a = numpy.load("p.npy")
        pkunit.pkeq((1, 13), a.shape)
        numpy.testing.assert_allclose(a[0, 0], 0.001)
        numpy.testing.assert_allclose(a[0, 2], 0.003)
        numpy.testing.assert_allclose(a[0, 12], scipy.constants.e)


def test_save_stats_lattice_and_volume():
    import numpy
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.utils import save_stats

    def _bunch():
        return rftrack_util.twiss_bunch(rft, 100.0, 100)

    with pkunit.save_chdir_work():
        lattice = rft.Lattice()
        lattice.append(rft.Drift(1.0))
        lattice.track(_bunch())
        save_stats(lattice, "stats.npy")
        a = numpy.load("stats.npy")
        # transport table, then on-axis Ez and Bz; s is in meters
        pkunit.pkeq(34, a.shape[1])
        numpy.testing.assert_allclose(a[-1, 0], 1.0)

        volume = rft.Volume()
        volume.add(rft.Drift(1.0), 0, 0, 0, 0, 0, 0)
        volume.set_s0(0)
        volume.set_s1(1.0)
        volume.tt_dt_mm = 10.0
        volume.track(_bunch())
        save_stats(volume, "stats.npy")
        b = numpy.load("stats.npy")
        pkunit.pkeq(34, b.shape[1])
        numpy.testing.assert_allclose(b[-1, 0], 1.0, atol=0.011)
