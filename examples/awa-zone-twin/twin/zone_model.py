"""LUME models for AWA zone1/zone2, chained with `lume.staged_model.StagedModel`.

Both zones' physics come from two scripts, the code Sirepo's rftrack app
generates for its "AWA Zone #1" and "AWA Zone #2" sims, kept in `sources/zone1_source.py`/`sources/zone2_source.py` (see
`_load_generated_module()` below), which build their beamlines with
RF-Track's own API. Reusing their `build_beamline()`/`build_bunch()`/
`autophase()` and `lume_rftrack`'s `apply_space_charge()`/`build_model()`
means this module doesn't re-derive the lattice, bunch, autophase, and
space-charge setup -- the two scripts are the single source of truth.

Each zone is the `lume_rftrack.model.LUMERFTrackModel` `build_model()`
returns, which is already a `StagedModel` stage: zone1's tracked bunch
becomes zone2's initial bunch on every `set()`, through
`ParticleGroup`. This module only registers each zone's own action
variables (`actions.py`) on it: zone1's field-map solenoids'
`<name>_current` (the coil current [A], set as RF-Track's own live scale of a
`Static_Magnetic_FieldMap_1d`'s field, converted with each map's
reference current, `actions.SOLENOID_REF_A`), zone2's quadrupoles' normalized `<name>_K1`
(`actions.QuadrupoleK1Variable`), and each zone's diagnostics and screen
images. Every one is set on, or read from, the live beamline; neither
zone ever rebuilds its beamline for a `set()`.

Zone1's track is slow: a space-charge-tracked run of 10,000
macroparticles (the sim's own `BUNCH.np`) takes real time, and building
it also reads ~10 field-map files and autophases. Zone2 has no field
maps, no autophase, and no space charge, and is fast.
"""

import pathlib

import actions
import RF_Track as rft
from lume.staged_model import StagedModel
from pykern.pkcollections import PKDict

import lume_rftrack.model
import lume_rftrack.utils

# Neither generated script's SETTINGS.num_threads is used (only its
# `main()` reads it), so `build_chain()` sets the thread count -- capped at
# rft.max_number_of_threads so it stays valid on a machine with fewer cores.
# 8 is within 8% of the fastest zone1 build at 1000, 2000 and 5000 particles
# (4-8 threads win; 12 or more is slower).
DEFAULT_THREADS = 8

# Each screen's image field of view [mm, full width] -- NOT the mechanical
# screen size (its aperture): at nominal settings the
# beam is only ~0.1-1.5 mm in sigma there (measured), tens of times
# smaller than the mechanical screen, so an image histogrammed over the
# full mechanical size is almost entirely empty. These are ~10 sigma at
# nominal settings (some headroom for retuning before the beam image
# clips); a large enough retune can still outgrow them -- regenerate with
# new numbers (print each screen's sigma from a tracked chain)
# if that happens often for your settings.
SCREEN_FOV_MM = {
    "YAG1": 10.0,
    "YAG2": 6.0,
    "YAG3": 6.0,
    "YAG4": 6.0,
    "YAG5": 6.0,
    "YAG7": 10.0,
}

# The last line of each zone's script, which runs the simulation
_MAIN_CALL = "\nmain()\n"

# Each zone's script
_SOURCES_DIR = pathlib.Path(__file__).resolve().parent / "sources"

# The field-map files zone1's script names
_LIB_DIR = _SOURCES_DIR / "zone1_lib"


def build_chain(n_particles=None, threads=None):
    """`zone1_model(n_particles)` staged into `zone2_model()`: zone1's
    tracked bunch becomes zone2's initial bunch on every `set()` and
    `reset()`.

    `threads` is RF-Track's thread count for every track (default
    `DEFAULT_THREADS`, capped at `rft.max_number_of_threads`).
    """
    rft.cvar.number_of_threads = min(
        DEFAULT_THREADS if threads is None else threads, rft.max_number_of_threads
    )
    return _Chain([zone1_model(n_particles), zone2_model()])


def screen_names(beamline):
    """The names of `beamline`'s screens, in order (as
    `get_bunch_at_screens()` returns their bunches)."""
    return [
        beamline[i].get_name()
        for i in range(beamline.size())
        if isinstance(beamline[i], rft.Screen)
    ]


def zone1_model(n_particles=None):
    """AWA zone 1, the chain's first stage: it generates its own cathode
    bunch, autophased and with space charge, as its script does.

    `n_particles`, if given, overrides the sim's own `BUNCH.np` (10,000)
    -- lower it for a faster interactive response at the cost of
    statistics.
    """
    m = _load_generated_module("zone1_source")
    rft.rng_set_seed(m.BUNCH.random_seed)
    if n_particles is not None:
        m.BUNCH.np = n_particles
    scales = _unscale_solenoids(m)
    b = m.build_bunch()
    v = m.build_beamline()
    _restore_solenoid_scaling(v, scales)
    m.autophase(v, b)
    lume_rftrack.utils.apply_space_charge(v, m.SETTINGS.space_charge)
    return _add_variables(
        lume_rftrack.model.build_model(v, b),
        "zone1_",
        actions.energy_diagnostic_variables,
        actions.solenoid_current_variables,
    )


def zone2_model():
    """AWA zone 2, the chain's second stage: it tracks nothing until zone1
    hands it a bunch. Its script's own `BUNCH` distribution (a saved
    snapshot of zone1's output) is unused; its design momentum is the
    rigidity each `<quad>_K1` converts with, as the script builds them.
    """
    return _add_variables(
        lume_rftrack.model.build_model(ZONE2_MODULE.build_beamline(), None),
        "zone2_",
        actions.beam_diagnostic_variables,
        lambda s: actions.quadrupole_k1_variables(
            s, ZONE2_MODULE.BUNCH.pc_mev / ZONE2_MODULE.BUNCH.charge
        ),
    )


class _Chain(StagedModel):
    """`StagedModel`, whose `reset()` resets each stage on its own, which
    would leave zone2 tracking the beam zone1 had before; this then hands
    zone1's reset beam on to zone2."""

    def reset(self):
        super().reset()
        self._set({})


def _add_variables(model, prefix, diagnostic_variables, element_variables):
    """Register the zone's own variables on `model`, then track it at its
    defaults: `element_variables(model.simulator)`, then the diagnostics
    `diagnostic_variables(prefix)` and the screen images.
    """
    for v in (
        *element_variables(model.simulator),
        *diagnostic_variables(prefix).values(),
        *actions.screen_image_variables(
            screen_names(model.simulator.beamline), SCREEN_FOV_MM
        ).values(),
    ):
        model.register_action_variable(v)
    model.reset()
    return model


def _load_generated_module(name):
    """Exec `sources/<name>.py`, without its trailing `main()` call, into a
    fresh namespace, returning that namespace as a `PKDict` of its
    top-level names (`BUNCH`, `SETTINGS`, `build_beamline`, `build_bunch`,
    `autophase` if the sim uses it, ...). A fresh namespace per call, so
    one zone model's edit of `BUNCH` (`n_particles`) isn't another's.
    """
    path = _SOURCES_DIR / f"{name}.py"
    text = path.read_text()
    if not text.endswith(_MAIN_CALL):
        raise AssertionError(f"{path} does not end with {_MAIN_CALL!r}")
    ns = PKDict(__file__=str(path), __name__=f"lume_rftrack_awa_zone_twin.{name}")
    exec(compile(text[: -len(_MAIN_CALL)], str(path), "exec"), ns)
    # the field maps are named relative to the sim's own run directory
    for s in ns.ELEMENTS:
        if "file" in s:
            s.file = str(_LIB_DIR / s.file)
    return ns


def _restore_solenoid_scaling(beamline, scales):
    """Set each field-map solenoid in `scales` (`_unscale_solenoids()`'s)
    to its scale factor, as RF-Track's own live `scaling_factor`."""
    for n, f in scales.items():
        beamline.get_elements_by_name(n).set_scaling_factor(f)


def _unscale_solenoids(module):
    """Zero out `module.ELEMENTS`' field-map solenoids' `scale_factor`
    (the generated script multiplies the map by it), returning
    `{name: scale_factor}`. The map is then built as read, and
    `_restore_solenoid_scaling()` applies the factor live instead, which
    is what makes `<name>_current` settable, from the sim's own nominal
    (e.g. 550 A).
    """
    res = PKDict()
    for s in module.ELEMENTS:
        if s.kind == "solenoid_fieldmap":
            res[s.name] = s.scale_factor
            s.scale_factor = 1.0
    return res


# Read-only views of each source (its beamline, settings) for consumers that
# only inspect it. `zone1_model()` execs its own copy, because it edits BUNCH.
# Assigned here, after `_load_generated_module()`, which they call at import.
ZONE1_MODULE = _load_generated_module("zone1_source")
ZONE2_MODULE = _load_generated_module("zone2_source")
