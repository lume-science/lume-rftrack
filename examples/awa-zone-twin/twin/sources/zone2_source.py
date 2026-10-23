#!/usr/bin/env python
"""
Simulation notes:
    https://git.cels.anl.gov/awa/awa-rftrack-lattices

"""

import beamphysics
import RF_Track as rft
from pykern.pkcollections import PKDict

import lume_rftrack.model
import lume_rftrack.utils

BUNCH = PKDict(
    charge=-1.0,
    mass_mev=0.51099895069,
    pc_mev=60.0,
    random_seed=5489,
    distribution_file="beam-distributionFile.awa_zone1_output.h5",
)
SETTINGS = PKDict(
    num_threads=1,
    space_charge=PKDict(),
)

# Every element's aperture: a circular 15 mm half-width beam pipe
_APERTURE_M = 0.015

# The design rigidity [MV/c] each quadrupole's K1 [1/m^2] is converted with
_P_OVER_Q = BUNCH.pc_mev / BUNCH.charge


def build_beamline():
    # Lattice appends elements sequentially with no absolute-position
    # concept at all: every element begins where the previous one ended,
    # with drifts filling the gaps. There is no Volume-style tracking range
    # to set (a Lattice is inherently bounded by its own appended
    # elements), and no continuous-ODE step size to tune (dt_mm/
    # odeint_algorithm/tt_dt_mm are Volume-only) -- each element instead
    # has its own default step count.
    v = rft.Lattice()
    for e in (
        rft.Drift(0.555448),
        _named("DT8", rft.Corrector(0.085852)),
        rft.Drift(0.147574),
        _quadrupole("DQ1", 0.1016, 0.0),
        rft.Drift(0.143),
        _quadrupole("DQ2", 0.1016, 0.0),
        rft.Drift(0.2824),
        _quadrupole("DQ3", 0.1016, 0.0),
        rft.Drift(0.134974),
        _named("DT9", rft.Corrector(0.085852)),
        rft.Drift(0.142748),
        _named("DT10", rft.Corrector(0.085852)),
        rft.Drift(0.123773),
        _quadrupole("DQ4", 0.168402, 3.0),
        rft.Drift(0.094998),
        _quadrupole("DQ5", 0.168402, 0.0),
        rft.Drift(0.085598),
        _quadrupole("DQ6", 0.168402, 0.0),
        rft.Drift(0.104698),
        _quadrupole("DQ7", 0.168402, -4.0),
        rft.Drift(1.638073),
        _named("DT12", rft.Corrector(0.085852)),
        rft.Drift(0.293674),
        _screen("YAG5"),
        rft.Drift(0.1841),
        _quadrupole("DQ8", 0.0762, 7.0),
        rft.Drift(0.2731),
        _quadrupole("DQ9", 0.0762, 3.0),
        rft.Drift(0.071374),
        _named("ET2", rft.Corrector(0.085852)),
        rft.Drift(0.074674),
        _quadrupole("DQ10", 0.0762, -8.0),
        rft.Drift(0.3904),
        _screen("YAG7"),
    ):
        e.set_aperture(_APERTURE_M, _APERTURE_M, "circular")
        # set_tt_nsteps() controls how finely get_transport_table()
        # samples within this element (the Lattice-only equivalent of
        # Volume's tt_dt_mm) -- without it, the transport table (and so
        # the Beam Variables/Lattice Overlay plots) only reports one
        # point per element.
        e.set_tt_nsteps(20)
        v.append(e)
    # nonzero: RF_Track prints tracking progress (e.g. "t = ... mm/c")
    # straight to stdout as track() advances, which is run.log's own
    # content with no further plumbing needed
    v.verbosity = 1
    return v


def build_bunch():
    pg = beamphysics.ParticleGroup(h5=BUNCH.distribution_file)
    # Already-accelerated bunch (e.g. re-injecting a previous sim's tracked
    # output): Bunch6d is the only bunch type Lattice.track() accepts, so
    # Sequential (Lattice) positioning requires this branch. The group is
    # drifted to a single z plane first, which keeps the bunch length.
    return lume_rftrack.utils.particlegroup_to_bunch6d(pg)


def main():
    rft.rng_set_seed(BUNCH.random_seed)
    rft.cvar.number_of_threads = SETTINGS.num_threads

    bunch = build_bunch()
    lume_rftrack.utils.save_particles("initial_particles.npy", bunch)

    volume = build_beamline()

    model = lume_rftrack.model.build_model(volume, bunch)
    model.reset()
    lume_rftrack.utils.save_particles(
        "final_particles.npy", model.simulator.tracked_bunch
    )
    lume_rftrack.utils.save_stats(model.simulator.beamline, "stats.npy")
    lume_rftrack.utils.save_screens(model.simulator.beamline, bunch, "screen-{}.npy")
    lume_rftrack.utils.save_bpms(model.simulator.beamline, "bpms.npy")


def _named(name, element):
    element.set_name(name)
    return element


def _quadrupole(name, length, k1):
    """A quadrupole of normalized strength `k1` [1/m^2] at the design
    rigidity; RF-Track keeps its gradient, which `build_model()` serves."""
    return _named(name, rft.Quadrupole(length, _P_OVER_Q, k1))


def _screen(name):
    """A screen with its t0 pinned to 0: RF-Track's track() re-autophases
    every RF element and Screen from scratch if any has t0_is_set() False,
    and this zone doesn't autophase."""
    e = _named(name, rft.Screen())
    e.set_t0(0.0)
    return e


# Last, and not under "if __name__ == '__main__'": twin/zone_model.py execs
# this file without this line, so it can build the beamline and bunch
# without running the simulation.
main()
