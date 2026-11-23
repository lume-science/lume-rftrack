"""Tracking physics of single elements: each is checked against its textbook
(thin-lens, kick or free-flight) value, not just against itself. The beam is
a 100 MeV/c electron bunch, so the rigidity is B*rho = pc/c = 0.3336 T m.
"""

_PC_MEV = 100.0

_FIELDS = "%x %xp %y %yp"


def test_corrector_kick():
    """A corrector's strength is an integrated field [T mm]: every particle
    is kicked by strength / (B rho), and an electron goes the opposite way in
    x and y (Kx and Ky are the integrated By and Bx, so the signs differ)."""
    import numpy
    import RF_Track as rft

    x, y = _kicks(rft.Corrector(0.1, 1.0, 2.0))
    # T mm / T m = mrad, which is what xp and yp are in
    numpy.testing.assert_allclose(x, -1.0 / _brho_t_m(), rtol=1e-3)
    numpy.testing.assert_allclose(y, 2.0 / _brho_t_m(), rtol=1e-3)


def test_drift_propagation():
    """After a drift of length L, x = x0 + L x0' (xp is in mrad, x in mm)."""
    import numpy
    import RF_Track as rft

    i, o = _track(rft.Drift(1.0))
    numpy.testing.assert_allclose(
        o[:, 0], i[:, 0] + 1.0 * i[:, 1], rtol=1e-6, atol=1e-9
    )
    numpy.testing.assert_allclose(o[:, 1], i[:, 1], rtol=1e-6, atol=1e-9)


def test_quadrupole_gradient():
    """A thin quadrupole of gradient g [T/m] and length L changes a
    particle's angle by -+ (g L / B rho) x, and the sign in y is the opposite
    of the sign in x. A positive gradient defocuses an electron in x
    (RF-Track's sign follows the particle's charge)."""
    import math

    import numpy
    import RF_Track as rft

    for g in (5.0, -5.0):
        q = rft.Quadrupole(0.1, math.nan, 0.0)
        q.set_gradient(g)
        i, o = _track(q)
        kl = g * 0.1 / _brho_t_m()
        # 5%: the quadrupole is 0.1 m long, not thin, and its two planes
        # differ by about +-2% (cosh vs cos)
        numpy.testing.assert_allclose(_slope(i, o, 0, 1), kl, rtol=0.05)
        numpy.testing.assert_allclose(_slope(i, o, 2, 3), -kl, rtol=0.05)


def test_screen_size():
    """A Screen's width and height are in meters (not mm, as the RF-Track
    reference manual says): it records the particles with
    |x| <= width / 2, with x in mm."""
    import numpy
    import RF_Track as rft

    xs = [-6.0, -4.0, 4.0, 6.0]
    b = rft.Bunch6d(
        numpy.array(
            [
                [x, 0.0, 0.0, 0.0, 0.0, _PC_MEV, rft.electronmass, -1.0, 1.0, i]
                for i, x in enumerate(xs)
            ]
        )
    )
    # 10.5 mm wide keeps |x| <= 5.25 mm; 12.5 mm keeps |x| <= 6.25 mm
    for width, expect in ((0.0105, xs[1:3]), (0.0125, xs)):
        v = rft.Volume()
        v.add(rft.Drift(0.1), 0, 0, 0, 0, 0, 0)
        v.add(rft.Screen(), 0, 0, 0.05, 0, 0, 0)
        v.set_s0(0.0)
        v.set_s1(0.1)
        v[1].set_width(width)
        v.track(b)
        numpy.testing.assert_array_equal(
            sorted(v.get_bunch_at_screens()[0].get_phase_space("%x").ravel()),
            expect,
            err_msg=f"width={width}",
        )


def test_solenoid_focuses():
    """A weak solenoid whose field starts and ends at zero rotates the beam
    by integral(Bz dz) / (2 B rho) and focuses it in both planes with
    1/f = integral(Bz^2 dz) / (4 B rho^2): its transfer matrix is that
    rotation times the focusing (R21 = R43 = -cos(rotation) / f). elegant
    and MAD-X give the same matrix for this field to 0.5%.

    The Volume's step is set: its default adaptive integrator
    (`odeint_epsabs` 1e-3, `dt_mm` inf) steps across the whole solenoid
    and finds little of either (RF-Track 2.7.4; 2.7.3 too, less so).
    """
    import math

    import numpy
    import RF_Track as rft

    bz = 0.04 * numpy.exp(-(numpy.linspace(-1, 1, 101) ** 2) * 8)
    sol = rft.Static_Magnetic_FieldMap_1d(bz, 0.01)
    v = rft.Volume()
    v.add(sol, 0, 0, 0, 0, 0, 0)
    v.set_s0(0.0)
    v.set_s1(sol.get_length())
    v.dt_mm = 1.0
    m = _transfer_matrix(v)
    # an electron in a positive Bz rotates from x toward -y
    numpy.testing.assert_allclose(
        math.atan2(-m[0, 2], m[0, 0]),
        numpy.trapezoid(bz, dx=0.01) / (2 * _brho_t_m()),
        rtol=0.01,
    )
    numpy.testing.assert_allclose(
        [m[1, 0], m[3, 2]],
        -numpy.trapezoid(bz**2, dx=0.01) / (4 * _brho_t_m() ** 2),
        rtol=0.01,
    )


def _brho_t_m():
    import scipy.constants

    return _PC_MEV * 1e6 / scipy.constants.c


def _kicks(element):
    i, o = _track(element)
    return (o[:, 1] - i[:, 1]).mean(), (o[:, 3] - i[:, 3]).mean()


def _slope(i, o, position, angle):
    """Change in the angle [mrad] per position [mm]: 1/m"""
    import numpy

    return numpy.polyfit(i[:, position], o[:, angle] - i[:, angle], 1)[0]


def _track(element):
    import RF_Track as rft
    import rftrack_util

    from lume_rftrack.simulator import RFTrackSimulator

    if isinstance(element, rft.Volume):
        b = element
    else:
        b = rft.Lattice()
        b.append(element)
    s = RFTrackSimulator(b, rftrack_util.twiss_bunch(rft, _PC_MEV, 200))
    i = s.initial_bunch.get_phase_space(_FIELDS)
    s.track()
    return i, s.tracked_bunch.get_phase_space(_FIELDS)


def _transfer_matrix(beamline):
    """`beamline`'s 4x4 (x [mm], xp [mrad], y, yp) transfer matrix: a
    reference particle on axis and one displaced 1 mm or 1 mrad in each
    coordinate, differenced. Column k is the response to coordinate k."""
    import numpy
    import RF_Track as rft

    from lume_rftrack.simulator import RFTrackSimulator

    p = numpy.zeros((5, 10))
    p[:, 5:9] = (_PC_MEV, rft.electronmass, -1.0, 1.0)
    p[:, 9] = range(5)
    p[1:, :4] = numpy.eye(4)
    s = RFTrackSimulator(beamline, rft.Bunch6d(p))
    s.track()
    o = s.tracked_bunch.get_phase_space(_FIELDS)
    return (o[1:] - o[0]).T
