"""Tests for `twin/zone_model.py`: a zone1 model's `reset()`,
independence of separate zone1 models, the zone1 -> zone2 handoff
(charge, energy and bunch length all survive it), and the chain's
`reset()` handing zone1's beam on to zone2.
"""

import sys

import pykern.pkio

_TWIN_DIR = pykern.pkio.py_path(__file__).dirpath().dirpath().join("twin")
_N_PARTICLES = 300
# Tests run single-threaded so they never take every core on a big machine.
_THREADS = 1
_SOLENOID = "SolBF_current"


def test_reset_restores_solenoid_current():
    sys.path.insert(0, str(_TWIN_DIR))
    import RF_Track as rft
    from pykern import pkunit
    from zone_model import zone1_model

    rft.cvar.number_of_threads = _THREADS
    z = zone1_model(n_particles=_N_PARTICLES)
    nominal = z.get([_SOLENOID])[_SOLENOID]
    z.set({_SOLENOID: nominal * 2})
    pkunit.pkeq(nominal * 2, z.get([_SOLENOID])[_SOLENOID])
    z.reset()
    pkunit.pkeq(nominal, z.get([_SOLENOID])[_SOLENOID])


def test_zone1_models_do_not_share_state():
    sys.path.insert(0, str(_TWIN_DIR))
    import RF_Track as rft
    from pykern import pkunit
    from zone_model import zone1_model

    rft.cvar.number_of_threads = _THREADS
    a = zone1_model(n_particles=_N_PARTICLES)
    nominal = a.get([_SOLENOID])[_SOLENOID]
    a.set({_SOLENOID: nominal * 2})
    pkunit.pkeq(
        nominal, zone1_model(n_particles=_N_PARTICLES).get([_SOLENOID])[_SOLENOID]
    )


def test_chain_reset_hands_zone1_beam_to_zone2():
    """After `reset()`, zone2 tracks zone1's reset beam, not the one
    zone1 had before."""
    sys.path.insert(0, str(_TWIN_DIR))
    import numpy
    from pykern import pkunit
    from zone_model import build_chain

    import lume_rftrack.utils

    chain = build_chain(n_particles=_N_PARTICLES, threads=_THREADS)
    zone1, zone2 = chain.lume_model_instances
    chain.set({_SOLENOID: zone1.get([_SOLENOID])[_SOLENOID] * 1.5})
    changed = zone2.final_particles.x.std()
    chain.reset()
    numpy.testing.assert_array_equal(
        zone2.simulator.initial_bunch.get_phase_space("%x %xp %y %yp %t %P"),
        lume_rftrack.utils.particlegroup_to_bunch6d(
            zone1.final_particles
        ).get_phase_space("%x %xp %y %yp %t %P"),
    )
    pkunit.pkok(
        zone2.final_particles.x.std() != changed, "zone2 still has the changed beam"
    )


def test_handoff_keeps_charge_energy_and_bunch_length():
    sys.path.insert(0, str(_TWIN_DIR))
    import actions
    from pykern import pkunit
    from zone_model import build_chain

    chain = build_chain(n_particles=_N_PARTICLES, threads=_THREADS)
    zone1, zone2 = chain.lume_model_instances
    chain.set({})
    # zone2 serves no energy of its own (magnetic transport doesn't change
    # it), so measure its bunch the way zone1's are
    for v1, v2 in zip(
        actions.energy_diagnostic_variables("zone1_").values(),
        actions.energy_diagnostic_variables("zone2_").values(),
    ):
        a = v1._get(zone1.simulator)
        b = v2._get(zone2.simulator)
        pkunit.pkok(
            abs(a - b) < 1e-3 * abs(a),
            "{} changed in handoff: zone1={} zone2={}",
            v2.name,
            a,
            b,
        )
    sigma_z = zone1.simulator.tracked_bunch.get_phase_space("%Z")[:, 0].std()
    sigma_t = zone2.simulator.initial_bunch.get_phase_space("%t")[:, 0].std()
    # zone1 is relativistic, so sigma_t [mm/c] ~ sigma_z [mm] / beta ~ sigma_z
    pkunit.pkok(
        abs(sigma_t - sigma_z) < 0.02 * sigma_z,
        "bunch length lost in handoff: sigma_t={} sigma_z={}",
        sigma_t,
        sigma_z,
    )


def test_zone1_solenoid_current_changes_the_beam():
    """`<solenoid>_current` sets RF-Track's own live scale of the field
    (current / reference current * 1e-4): setting it re-tracks the same
    beamline, and focuses differently."""
    sys.path.insert(0, str(_TWIN_DIR))
    import RF_Track as rft
    from pykern import pkunit
    from zone_model import zone1_model

    rft.cvar.number_of_threads = _THREADS
    z = zone1_model(n_particles=_N_PARTICLES)
    beamline = z.simulator.beamline

    def _sigma_x():
        return z.simulator.tracked_bunch.get_phase_space("%X")[:, 0].std()

    nominal = z.get([_SOLENOID])[_SOLENOID]
    before = _sigma_x()
    z.set({_SOLENOID: nominal * 1.5})
    pkunit.pkok(z.simulator.beamline is beamline, "beamline was rebuilt")
    pkunit.pkeq(550.0, nominal)
    pkunit.pkeq(
        nominal * 1.5 / 550.0 * 1e-4,
        z.simulator.find_element("SolBF").get_scaling_factor(),
    )
    pkunit.pkok(
        abs(_sigma_x() - before) > 0.01 * before,
        "sigma_x didn't change: {} -> {}",
        before,
        _sigma_x(),
    )


def test_zone2_k1_sets_the_quadrupole():
    """`<quad>_K1` (`actions.QuadrupoleK1Variable`) is K1 against zone2's
    design P/Q: setting it reads back, and sets the element's own
    gradient to match, so `<quad>_gradient` follows."""
    sys.path.insert(0, str(_TWIN_DIR))
    import numpy
    import zone_model
    from pykern import pkunit

    z = zone_model.zone2_model()
    g = z.get(["DQ1_gradient"])["DQ1_gradient"]
    z.set({"DQ1_K1": 3.0})
    numpy.testing.assert_allclose(z.get(["DQ1_K1"])["DQ1_K1"], 3.0)
    b = zone_model.ZONE2_MODULE.BUNCH
    numpy.testing.assert_allclose(
        z.simulator.find_element("DQ1").get_K1(b.pc_mev / b.charge), 3.0
    )
    pkunit.pkok(
        z.get(["DQ1_gradient"])["DQ1_gradient"] != g, "gradient didn't follow K1"
    )
