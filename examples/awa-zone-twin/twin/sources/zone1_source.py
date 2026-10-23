#!/usr/bin/env python
"""
Simulation notes:
    https://git.cels.anl.gov/awa/awa-rftrack-lattices

"""

import pathlib

import numpy
import RF_Track as rft
from pykern.pkcollections import PKDict

import lume_rftrack.model
import lume_rftrack.utils

BUNCH = PKDict(
    charge=-1.0,
    mass_mev=0.51099895069,
    pc_mev=100.0,
    random_seed=5489,
    c_sig_t=3.0,
    c_sig_x=3.0,
    c_sig_y=3.0,
    cathode=True,
    dist_pz="fd_300",
    dist_x="r",
    dist_y="g",
    dist_z="plateau",
    e_photon=4.73,
    lt=0.006,
    noise_reduc=True,
    np=10000,
    phi_eff=3.5,
    q_total_nc=1.0,
    ref_clock=0,
    ref_ekin=0,
    ref_zpos=0,
    rt=0.0003,
    sig_x=1.67,
    species="electrons",
)
SETTINGS = PKDict(
    num_threads=1,
    space_charge=PKDict(
        kind="pic",
        dt_mm=3,
        grid=(
            8,
            8,
            16,
        ),
        mirror_m=0.0,
        smooth=0.5,
    ),
    tracking_range_m=(
        0.0,
        11.372,
    ),
    volume_dt_mm=0.5,
    volume_tt_dt_mm=10.0,
)


# Every element's aperture: a circular 15 mm half-width beam pipe
_APERTURE_M = 0.015

# The field-map files, next to this file
_LIB_DIR = pathlib.Path(__file__).resolve().parent / "zone1_lib"

# Every linac cavity's field map
_LINAC = "CAVITY-fieldMapFile.linac.dat"


def autophase(volume):
    # Autophase against a single on-axis reference particle, then track
    # the real bunch through this SAME (now-phased) volume; do not
    # rebuild it. Volume.autophase() sets each RF element's internal t0
    # directly and .track() uses that state as-is -- get_phid() does not
    # reflect it (it always reads back the phid the element was
    # constructed with), so reading it back to "convert" t0 into an
    # equivalent phid for a rebuilt volume silently produces the unphased
    # result instead of the true autophased state.
    # A Cathode or From File reference particle starts at rest
    # (X,Px,Y,Py,Z,Pz all 0), not at the design momentum, since both are
    # snapshots of a bunch that begins from (near) rest, not an
    # already-accelerated analytic (Twiss) distribution.
    p0 = rft.Bunch6dT(
        numpy.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, BUNCH.mass_mev, BUNCH.charge, 1.0, 0.0]]
        )
    )
    volume.unset_t0()
    volume.autophase(p0)


def build_beamline():
    # Volume (rather than Lattice) supports elements at explicit, possibly
    # overlapping absolute z positions -- e.g. a gun cavity's E field and
    # a bucking/focusing solenoid's B field genuinely occupy the same
    # physical space in a real photoinjector.
    v = rft.Volume()
    for z, e in (
        (
            -0.5,
            _solenoid(
                "SolBF", "SOLENOID-fieldMapFile.sol_bucking-focusing_pm550A.dat", 1e-4
            ),
        ),
        (0.0, _cavity("Gun", "CAVITY-fieldMapFile.rf_gunG4.dat", 80e6)),
        (0.0, _solenoid("SolM", "SOLENOID-fieldMapFile.sol_matching_m440A.dat", 1e-4)),
        (0.576433, _cavity("L1", _LINAC, 22e6)),
        (0.701802, _named("DT1", rft.Corrector(0.0381))),
        (1.544, _solenoid("SolL1", "SOLENOID-fieldMapFile.sol_linacLB_500A.dat", 2e-7)),
        (1.63525, _named("DT2", rft.Corrector(0.0381))),
        (2.9, _named("YAG1", rft.Screen())),
        (3.306433, _cavity("L2", _LINAC, 22e6)),
        (4.12, _solenoid("SolL2", "SOLENOID-fieldMapFile.sol_linac_500A.dat", 2e-7)),
        (4.3053, _named("DT3", rft.Corrector(0.0381))),
        (4.856433, _cavity("L3", _LINAC, 22e6)),
        (5.84835, _named("DT4", rft.Corrector(0.0381))),
        (5.881, _solenoid("SolL4", "SOLENOID-fieldMapFile.sol_linac_500A.dat", 2e-7)),
        (6.292, _named("YAG2", rft.Screen())),
        (6.986433, _cavity("L4", _LINAC, 22e6)),
        (7.0358, _named("DT5", rft.Corrector(0.0381))),
        (8.326433, _cavity("L5", _LINAC, 22e6)),
        (9.525, _named("YAG3", rft.Screen())),
        (9.736433, _cavity("L6", _LINAC, 22e6)),
        (9.7917, _named("DT6", rft.Corrector(0.0381))),
        (11.120374, _named("DT7", rft.Corrector(0.085852))),
        (11.372, _named("YAG4", rft.Screen())),
    ):
        e.set_aperture(_APERTURE_M, _APERTURE_M, "circular")
        v.add(e, 0.0, 0.0, z, 0.0, 0.0, 0.0)
    v.set_s0(SETTINGS.tracking_range_m[0])
    v.set_s1(SETTINGS.tracking_range_m[1])
    v.odeint_algorithm = "rkf45"
    # RF-Track's default, 1e-3, leaves the beam about 25% larger at YAG4
    # and the gun autophased about 0.4 deg off
    v.odeint_epsabs = 1e-6
    v.dt_mm = SETTINGS.volume_dt_mm
    v.tt_dt_mm = SETTINGS.volume_tt_dt_mm
    # nonzero: RF_Track prints tracking progress (e.g. "t = ... mm/c")
    # straight to stdout as track() advances, which is run.log's own
    # content with no further plumbing needed
    v.verbosity = 1
    return v


def build_bunch():
    # Cathode emission: particles start essentially at rest and are
    # generated directly from the photocathode's laser/emission model,
    # rather than from an already-moving analytic (Twiss) distribution.
    g = rft.Bunch6dT_Generator()
    g.species = BUNCH.species
    g.cathode = BUNCH.cathode
    g.noise_reduc = BUNCH.noise_reduc
    g.q_total = BUNCH.q_total_nc
    g.ref_ekin = BUNCH.ref_ekin
    g.ref_zpos = BUNCH.ref_zpos
    g.ref_clock = BUNCH.ref_clock
    g.phi_eff = BUNCH.phi_eff
    g.e_photon = BUNCH.e_photon
    g.dist_x = BUNCH.dist_x
    g.dist_y = BUNCH.dist_y
    g.sig_x = BUNCH.sig_x
    g.dist_z = BUNCH.dist_z
    g.rt = BUNCH.rt
    g.lt = BUNCH.lt
    g.c_sig_x = BUNCH.c_sig_x
    g.c_sig_y = BUNCH.c_sig_y
    g.c_sig_t = BUNCH.c_sig_t
    g.dist_pz = BUNCH.dist_pz
    return rft.Bunch6dT(g, BUNCH.np)


def main():
    rft.rng_set_seed(BUNCH.random_seed)
    rft.cvar.number_of_threads = SETTINGS.num_threads

    bunch = build_bunch()
    lume_rftrack.utils.save_particles("initial_particles.npy", bunch)

    volume = build_beamline()
    autophase(volume)
    lume_rftrack.utils.apply_space_charge(volume, SETTINGS.space_charge)

    model = lume_rftrack.model.build_model(volume, bunch)
    model.reset()
    lume_rftrack.utils.save_particles(
        "final_particles.npy", model.simulator.tracked_bunch
    )
    lume_rftrack.utils.save_stats(model.simulator.beamline, "stats.npy")
    lume_rftrack.utils.save_screens(model.simulator.beamline, bunch, "screen-{}.npy")
    lume_rftrack.utils.save_bpms(model.simulator.beamline, "bpms.npy")


def _cavity(name, file, peak_v_per_m):
    """A 1.3 GHz standing-wave cavity from `file`'s on-axis Ez, rescaled
    to `peak_v_per_m`."""
    field, step, length = lume_rftrack.utils.load_field_map(
        PKDict(file=_LIB_DIR / file, rescale_mode="peak", target_field=peak_v_per_m)
    )
    return _named(name, rft.RF_FieldMap_1d(field, step, length, 1.3e9, +1))


def _named(name, element):
    element.set_name(name)
    return element


def _solenoid(name, file, scaling_factor):
    """A solenoid from `file`'s on-axis Bz, as read, scaled by RF-Track's
    own `scaling_factor` -- settable live, unlike rescaling the map."""
    field, step, _ = lume_rftrack.utils.load_field_map(
        PKDict(file=_LIB_DIR / file, rescale_mode="factor", scale_factor=1.0)
    )
    e = rft.Static_Magnetic_FieldMap_1d(field, step)
    e.set_scaling_factor(scaling_factor)
    return _named(name, e)


# Last, and not under "if __name__ == '__main__'": twin/zone_model.py execs
# this file without this line, so it can build the beamline and bunch
# without running the simulation.
main()
