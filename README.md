# lume-rftrack
RF-Track-specific implementation of LUMEModel classes for virtual accelerators

## What this package provides

- `RFTrackSimulator`: wraps a native RF-Track `Lattice` or `Volume` and an initial bunch, tracks it, and archives it to or loads it from HDF5.
- `LUMERFTrackModel`: a LUME action model driven by simulator-backed variables. Every `set()` re-tracks the whole beamline.
- `build_model()`: wraps a beamline in a `LUMERFTrackModel` with one writable variable per adjustable attribute of each named element, found from its RF-Track class and named `<element>_<attribute>`.
- Action variables in `lume_rftrack.actions`: `RFTrackReadOnlyScalarVariable`, `RFTrackWritableScalarVariable` and `RFTrackWritableVectorComponentVariable`, for any element attribute with a `get_X()`/`set_X()` pair, for building a model with your own variables.
- `lume_rftrack.units`: the unit of every RF-Track element attribute, by class.
- `lume_rftrack.utils`: space-charge setup, conversion between RF-Track bunches and openPMD `ParticleGroup`s, and helpers that save particles, screens, BPMs and beam statistics.

## Install

```
pip install git+https://github.com/lume-science/lume-rftrack
```

RF-Track is a dependency, distributed under its own license.

## Variables

`build_model()` makes a variable for these RF-Track attributes of each named element, by its class or nearest base class, using RF-Track's own names:

| RF-Track class | Variables |
| --- | --- |
| `Corrector` | `strength_x`, `strength_y` [T*mm] |
| `Quadrupole` | `gradient` [T/m] |
| `RF_Field` (every RF element) | `phid` [deg], `P_actual` [W] (if the class has it) |
| `SBend` | `angle` [rad] |
| `Sextupole` | `gradient` [T/m^2] |
| `Solenoid` | `max_current` [A] |

Other elements (drifts, screens, field-map solenoids, ...) track and archive as usual but have no variables. For any other attribute, make an `RFTrackWritableScalarVariable` for it; `lume_rftrack.units.unit()` has its unit. A quantity RF-Track doesn't store directly, such as a quadrupole's normalized K1 (RF-Track's `get_K1(P_over_Q)` converts the gradient with the beam's rigidity), is an action of your own: see `QuadrupoleK1Variable` in `examples/awa-zone-twin/twin/actions.py`.

## Example: implementing a `LUMERFTrackModel` object

```python
import numpy
import RF_Track as rft

import lume_rftrack.model

# 1) Build a native RF-Track lattice: a quadrupole and a drift
quad = rft.Quadrupole(0.1)
quad.set_name("Q1")
quad.set_gradient(5.0)  # T/m
drift = rft.Drift(1.0)
drift.set_name("D1")
lattice = rft.Lattice()
lattice.append(quad)
lattice.append(drift)

# 2) Create an initial bunch: 1000 electrons at 100 MeV/c
rng = numpy.random.default_rng(1)
phase_space = numpy.zeros((1000, 6))
phase_space[:, [0, 2]] = rng.normal(0, 0.5, (1000, 2))  # x, y [mm]
phase_space[:, 5] = 100.0  # P [MeV/c]
bunch = rft.Bunch6d(rft.electronmass, 1e9, -1, phase_space)

# 3) Wrap them in a model, with one writable variable per adjustable
#    element attribute, named <element>_<attribute>
model = lume_rftrack.model.build_model(lattice, bunch)

# 4) Use the model API: every set() re-tracks the beamline
model.set({"Q1_gradient": 8.0})
print(model.get(["Q1_gradient"]))
print(model.simulator.tracked_bunch.get_phase_space("%x").std(), "mm")

# Optional: archive the beamline and bunch to HDF5 and load them back
model.simulator.archive("run.h5")
model.simulator.load_archive("run.h5")

# Optional: reset the model to its initial state
model.reset()
```

## Staged models

A `LUMERFTrackModel` is a `lume.staged_model.StagedModel` stage: its
tracked bunch goes to the next stage, and the stage before it supplies
its initial bunch, as openPMD `ParticleGroup`s, so it chains with other
RF-Track models or with any LUME model that has the same particle mixins
(e.g. lume-impact's). Build a stage that takes its beam from upstream
with no bunch:

```python
from lume.staged_model import StagedModel

chain = StagedModel(
    [
        lume_rftrack.model.build_model(linac, bunch),
        lume_rftrack.model.build_model(chicane, None),
    ]
)
chain.set({"Q1_gradient": 8.0})  # re-tracks this stage and every one after
```

An incoming beam enters each stage at its entrance (z = 0), each particle
at its own arrival time, as a `Bunch6d` for a `Lattice` or a `Bunch6dT`
for a `Volume`. Times carry through unchanged, so a downstream stage's RF
phases are against the same clock as the stage before.

## Examples

Two complete examples are in `examples/`:

- `xband-linac`: a 100-500 MeV/c X-band electron linac, one `Lattice` of
  traveling-wave structures, quadrupoles, correctors and BPMs, after
  RF-Track's tutorials 7 and 13.
- `awa-zone-twin`: the Argonne Wakefield Accelerator's first two zones, a
  `Volume` chained into a `Lattice`, served as EPICS PVs and viewed in
  Phoebus.

## Archives

`RFTrackSimulator.archive()` writes the beamline, the initial bunch (as openPMD particles), the field maps (deduplicated) and the units to an HDF5 file, and `load_archive()` restores them. Like lume-impact's and lume-genesis' archives, it also holds the last run's output: if the simulator has tracked, an `output` group with the tracked bunch, which `load_archive()` restores as `tracked_bunch`, so a loaded model's `final_particles` needs no re-track. The file records the RF-Track and lume-rftrack versions that wrote it, which `lume_rftrack.archive.read_archive()` returns.

An element is archived as RF-Track exposes it: its class and every `get_X()`/`set_X()` pair, under RF-Track's own names, so any element class that can be built without arguments and set that way needs no code here. A class with state only its constructor takes (field maps, `Pillbox_Cavity`, `SBend`, `Solenoid`, `Coil`) also records its constructor arguments. `archive()` rebuilds each element and compares its field with the original's, and raises `NotImplementedError` for one it can't reproduce (e.g. a thick-winding `Solenoid`, or an `Absorber`) rather than archive it wrong.

## Development

```
pip install -e .
pykern test tests
pre-commit run --all-files
```

The version comes from the git tag through `setuptools_scm`.
