# X-band linac

A 100-500 MeV/c electron linac, after RF-Track's tutorials 7 ("A
100-500 MeV X-band linac") and 13 ("Imperfections and beam-based
alignment") in `rf-track-jupyter-notebooks`, as a lume-rftrack model.

`linac.py` builds the linac with RF-Track's own API and hands it to
`lume_rftrack.model.build_model()`, which serves every element's
attributes as variables. Nothing else wraps it.

## The lattice

One `rft.Lattice`: a half quadrupole, then FODO cells of eight 2π/3
traveling-wave structures (`rft.TW_Structure`, 11.9942 GHz, 80 MV/m, with
a short-range wakefield) and two quadrupoles. Each quadrupole has a
corrector before it and a BPM after it, as in tutorial 13. Three cells
reach about 490 MeV/c.

It differs from the tutorials in two ways:

- **Flat.** The tutorials build one FODO `Lattice` and append it to the
  linac once per cell. RF-Track keeps an appended lattice as a single
  container element, so `build_model()` would see only the containers. Here
  every structure, quadrupole, corrector and BPM is its own element of
  the linac, named `TW01`..`TW24`, `Q01`..`Q07`, `C01`..`C07` and
  `BPM01`..`BPM07`.
- **Analytic structures.** The tutorials' field map (`data/TWS_Xband.dat`)
  is replaced by RF-Track's analytic `TW_Structure`, so the example needs
  no data file. Its structure is 0.208 m, the map's is 0.212 m.

Each quadrupole is built with an undefined rigidity and a fixed K1, as in
the tutorials, so `autophase()` sets it from the reference momentum where
it sits. `build_model()` then serves its gradient.

## Run it

```bash
python linac.py
```

It prints the model's 45 variables (each structure's `phid`, each
quadrupole's `gradient`, each corrector's `strength_x`/`strength_y`), then
uses `set()` twice, each time re-tracking the bunch:

```
45 variables, for example:
  TW01_phid = 0 deg
  Q01_gradient = -5.055 T/m
  C01_strength_x = 0 T*mm
  C01_strength_y = 0 T*mm
nominal: P = 492.5 MeV/c, sigma_P/P = 7.33 permille, emittance x/y = 5.16/5.13 mm mrad
C01_strength_x = 0.1 T*mm: P = 492.5 MeV/c, sigma_P/P = 7.33 permille, emittance x/y = 18.14/5.13 mm mrad
  BPM x change [mm]: [-0.015 -0.249 -0.649 -0.256 -0.258  0.065  0.379]
every structure 20 deg off crest: P = 468.4 MeV/c, sigma_P/P = 2.37 permille, emittance x/y = 5.02/5.02 mm mrad
saved linac.png and phase_space.png
```

- **A corrector kick** moves the orbit at every BPM after it. The
  horizontal emittance grows because the off-axis bunch drives the
  structures' transverse wakefields.
- **RF phase.** On crest, the longitudinal wakefield leaves a large energy
  spread. Running every structure 20° off crest gives up a little energy
  to cancel most of it, which is tutorial 12's optimization.

It also saves two plots (matplotlib, which `openpmd-beamphysics` already
installs) in the current directory:

- `linac.png`: each run's mean momentum, horizontal orbit and horizontal
  emittance along the linac, one panel each. The kick's orbit oscillates
  through the FODO lattice, and its emittance grows wherever the orbit
  is far off axis in the structures.
- `phase_space.png`: the final bunch's momentum offset against arrival
  time, on crest and 20° off crest. On crest, the longitudinal wakefield
  bends the bunch into a large S; off crest, the RF slope cancels most
  of it.

## Tests

```bash
pykern test tests
```
