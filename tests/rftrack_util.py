"""Bunches and beamlines shared by the lume_rftrack tests

Kept out of the ``*_test.py`` files so no test imports another's helpers.
"""


def assert_same_tracking(original, reloaded):
    """Track both simulators and require the same result, particle by
    particle, to float rounding: archive() stores the bunch's own
    macroparticles, so a reloaded copy tracks like the original."""
    import numpy

    f = "%x %xp %y %yp %t %P"
    original.track()
    reloaded.track()
    numpy.testing.assert_allclose(
        reloaded.tracked_bunch.get_phase_space(f),
        original.tracked_bunch.get_phase_space(f),
        rtol=1e-12,
        atol=1e-12,
    )


def fodo_lattice(rft):
    """One period of an X-band linac FODO cell: D0, C1, C2, QF, C3, C4, QD.
    The cavities are identical `RF_FieldMap_1d` standing-wave pi-mode
    structures (11.9942 GHz, 80 MV/m, twelve half-wavelength cells, phid 0
    which is on crest), sampled analytically so there are no data files.
    The quadrupoles are +5 and -5 T/m; a positive gradient defocuses an
    electron in x (and focuses it in y), so for this electron beam QF is
    the horizontally defocusing one.
    """
    import math

    import numpy

    def _cavity(name):
        frequency_hz = 11994200000.0
        n_cells = 12
        length = n_cells * rft.clight / (2.0 * frequency_hz)
        z = numpy.linspace(0.0, length, 40 * n_cells + 1)
        e = rft.RF_FieldMap_1d(
            (80e6 * numpy.cos(n_cells * numpy.pi * z / length)).astype(complex),
            z[1] - z[0],
            length,
            frequency_hz,
            +1,
        )
        e.set_phid(0.0)
        e.set_name(name)
        return e

    def _quadrupole(name, gradient):
        e = rft.Quadrupole(0.1, math.nan, 0.0)
        e.set_gradient(gradient)
        e.set_name(name)
        return e

    d0 = rft.Drift(0.05)
    d0.set_name("D0")
    res = rft.Lattice()
    for e in (
        d0,
        _cavity("C1"),
        _cavity("C2"),
        _quadrupole("QF", 5.0),
        _cavity("C3"),
        _cavity("C4"),
        _quadrupole("QD", -5.0),
    ):
        res.append(e)
    return res


def twiss_bunch(rft, pc_mev, n, charge_nc=0.05, beta=1.0, emitt=1.0):
    """An electron `Bunch6d_QR` with `n` macroparticles at `pc_mev`: a round
    Twiss beam (alpha 0, `beta` [m], normalized `emitt` [mm mrad]) with
    sigma_t and sigma_pt of 1. `Bunch6d_QR` is quasi-random, so the same
    arguments always give the same bunch, whatever the random seed.
    """
    t = rft.Bunch6d_twiss()
    t.beta_x = t.beta_y = beta
    t.alpha_x = t.alpha_y = 0.0
    t.emitt_x = t.emitt_y = emitt
    t.sigma_t = t.sigma_pt = 1.0
    return rft.Bunch6d_QR(rft.electronmass, charge_nc * rft.nC, -1, pc_mev, t, n)


def volume_with_cavity(rft):
    """A Volume with one 1.3 GHz `RF_FieldMap_1d` cavity ("C1", phid 10)
    and a screen ("SCR1") 0.1 m after it."""
    import numpy

    nz = 21
    hz = 0.01
    # a smooth half-sine bump, not a single spike -- RF_Track's Volume.track()
    # autophases the cavity, which fails ("no on-crest accelerating phase")
    # against an unrealistic field profile
    cavity = rft.RF_FieldMap_1d(
        (1e6 * numpy.sin(numpy.linspace(0, numpy.pi, nz))).astype(complex),
        hz,
        (nz - 1) * hz,
        1.3e9,
        1.0,
        1.0,
        1.0,
    )
    cavity.set_name("C1")
    cavity.set_phid(10.0)
    screen = rft.Screen()
    screen.set_name("SCR1")
    v = rft.Volume()
    v.add(cavity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v.add(screen, 0.0, 0.0, cavity.get_length() + 0.1, 0.0, 0.0, 0.0)
    v.set_s0(0.0)
    v.set_s1(cavity.get_length() + 0.2)
    return v
