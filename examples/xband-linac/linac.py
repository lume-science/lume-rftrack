#!/usr/bin/env python
"""A 100-500 MeV/c X-band electron linac as a LUME model

RF-Track's tutorials 7 and 13 (rf-track-jupyter-notebooks, "A 100-500
MeV X-band linac" and "Imperfections and beam-based alignment"), built
flat: every structure, quadrupole, corrector and BPM is its own named
element of one `rft.Lattice`, instead of a FODO `Lattice` appended to the
linac once per cell, so `lume_rftrack.model.build_model()` serves each
one's attributes. The tutorials' field map is replaced by RF-Track's
analytic `TW_Structure`, so this needs no data file.

Run it to see the model's variables, then a corrector kick and an RF
phase change, each applied with `set()` and read back from the BPMs and
the tracked bunch. It saves two plots of the three runs: `linac.png`
(momentum, orbit and emittance along the linac) and `phase_space.png`
(the final longitudinal phase space, on and off crest).
"""

import numpy
import RF_Track as rft
from pykern.pkcollections import PKDict

import lume_rftrack.model

BUNCH = PKDict(
    charge=-1.0,
    emitt_mm_mrad=5.0,
    n_particles=1000,
    population=rft.nC,
    sigma_pt_permille=0.1,
    sigma_t_mm=0.1,
)

LINAC = PKDict(
    final_momentum_mev=500.0,
    frequency_hz=11.9942e9,
    # on-axis amplitude of the synchronous space harmonic
    gradient_v_per_m=80e6,
    initial_momentum_mev=100.0,
    # FODO phase advance per cell
    mu_deg=90.0,
    quadrupole_length_m=0.1,
    # 2pi/3 cells per structure: 0.208 m, close to the tutorials' 0.212 m map
    structure_cells=25,
    # structures between quadrupoles
    structures_per_half_cell=4,
)

# Each run's line color and style, in run order: the chart palette's first
# three categorical slots, with a line style as well so a run is never
# told apart by color alone
_LINES = (
    PKDict(color="#2a78d6", linestyle="solid"),
    PKDict(color="#eb6834", linestyle="dashed"),
    PKDict(color="#1baf7a", linestyle="dashdot"),
)

# Recessive axes and grid, neutral text
_PLOT_STYLE = {
    "axes.edgecolor": "#52514e",
    "axes.facecolor": "#fcfcfb",
    "axes.grid": True,
    "axes.labelcolor": "#0b0b0b",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "#fcfcfb",
    "grid.alpha": 0.25,
    "grid.color": "#52514e",
    "legend.frameon": False,
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
}


def build_beamline():
    """The linac, autophased: a half quadrupole, then enough FODO cells
    to reach `LINAC.final_momentum_mev`. Each quadrupole is built with an
    undefined rigidity (`nan`) and fixed K1, so `autophase()` sets each one
    to the reference momentum where it sits.
    """
    k1 = _k1()
    c = rft.Lattice()
    _append_cell(c, PKDict(), k1)
    gain = c.autophase(_reference_particle()) - LINAC.initial_momentum_mev
    v = rft.Lattice()
    n = PKDict()
    _append_steering(v, n, LINAC.quadrupole_length_m / 2, k1)
    for _ in range(
        round((LINAC.final_momentum_mev - LINAC.initial_momentum_mev) / gain)
    ):
        _append_cell(v, n, k1)
    v.autophase(_reference_particle())
    return v


def build_bunch():
    """A quasi-random bunch matched to the FODO lattice."""
    mu = numpy.deg2rad(LINAC.mu_deg)
    t = rft.Bunch6d_twiss()
    t.emitt_x = BUNCH.emitt_mm_mrad
    t.emitt_y = BUNCH.emitt_mm_mrad
    t.beta_x = _cell_length() * (1 - numpy.sin(mu / 2)) / numpy.sin(mu)
    t.beta_y = _cell_length() * (1 + numpy.sin(mu / 2)) / numpy.sin(mu)
    t.sigma_t = BUNCH.sigma_t_mm
    t.sigma_pt = BUNCH.sigma_pt_permille
    return rft.Bunch6d_QR(
        rft.electronmass,
        BUNCH.population,
        BUNCH.charge,
        LINAC.initial_momentum_mev,
        t,
        BUNCH.n_particles,
    )


def main():
    m = lume_rftrack.model.build_model(build_beamline(), build_bunch())
    v = m.supported_variables
    print(f"{len(v)} variables, for example:")
    for n in ("TW01_phid", "Q01_gradient", "C01_strength_x", "C01_strength_y"):
        print(f"  {n} = {m.get([n])[n]:.4g} {v[n].unit}")
    m.reset()
    r = PKDict(nominal=_run("nominal", m))
    bpms = m.simulator.beamline.get_bpm_readings()[:, 0]
    m.set({"C01_strength_x": 0.1})
    r["C01 kick"] = _run("C01_strength_x = 0.1 T*mm", m)
    print(
        "  BPM x change [mm]:",
        numpy.round(m.simulator.beamline.get_bpm_readings()[:, 0] - bpms, 3),
    )
    m.reset()
    m.set({n: -20.0 for n in v if n.endswith("_phid")})
    r["20° off crest"] = _run("every structure 20 deg off crest", m)
    _plot_linac(r)
    _plot_phase_space(r)
    print("saved linac.png and phase_space.png")


def _append_cell(lattice, names, k1):
    """One FODO cell: structures, a defocusing quadrupole, structures, a
    focusing quadrupole. `names` counts each kind, for unique names."""
    for k in (-k1, k1):
        for _ in range(LINAC.structures_per_half_cell):
            lattice.append(_named(names, "TW", _structure()))
        _append_steering(lattice, names, LINAC.quadrupole_length_m, k)


def _append_steering(lattice, names, length, k1):
    """A corrector, a quadrupole and a BPM, as in tutorial 13."""
    lattice.append(_named(names, "C", rft.Corrector()))
    lattice.append(_named(names, "Q", rft.Quadrupole(length, numpy.nan, k1)))
    lattice.append(_named(names, "BPM", rft.Bpm()))


def _cell_length():
    return (
        2 * LINAC.structures_per_half_cell * LINAC.structure_cells * _rf_cell_length()
        + 2 * LINAC.quadrupole_length_m
    )


def _k1():
    """The K1 [1/m^2] giving `LINAC.mu_deg` per cell, thin-lens."""
    return (
        numpy.sin(numpy.deg2rad(LINAC.mu_deg) / 2)
        / (_cell_length() / 4)
        / LINAC.quadrupole_length_m
    )


def _named(names, prefix, element):
    names[prefix] = names.get(prefix, 0) + 1
    element.set_name(f"{prefix}{names[prefix]:02d}")
    return element


def _plot_linac(runs):
    """linac.png: each run's mean momentum, horizontal orbit and
    horizontal emittance along the linac, one panel each"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    with matplotlib.pyplot.rc_context(_PLOT_STYLE):
        f, axes = matplotlib.pyplot.subplots(
            3, 1, sharex=True, figsize=(8, 8), layout="constrained"
        )
        for (n, r), s in zip(runs.items(), _LINES):
            for i, a in enumerate(axes, start=1):
                a.plot(r.table[:, 0], r.table[:, i], linewidth=2, label=n, **s)
        for a, y in zip(axes, ("P [MeV/c]", "<x> [mm]", "emittance x [mm mrad]")):
            a.set_ylabel(y)
        axes[-1].set_xlabel("s [m]")
        axes[0].legend()
        f.suptitle("X-band linac: three runs")
        f.savefig("linac.png", dpi=150)


def _plot_phase_space(runs):
    """phase_space.png: the final bunch's momentum offset against its
    arrival time, on crest and off crest, side by side"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot

    with matplotlib.pyplot.rc_context(_PLOT_STYLE):
        f, axes = matplotlib.pyplot.subplots(
            1, 2, sharex=True, sharey=True, figsize=(9, 4), layout="constrained"
        )
        for a, (n, s) in zip(
            axes, (("nominal", _LINES[0]), ("20° off crest", _LINES[2]))
        ):
            p = runs[n].longitudinal
            a.scatter(
                p[:, 0],
                (p[:, 1] / p[:, 1].mean() - 1) * 1e3,
                s=6,
                alpha=0.5,
                color=s.color,
                linewidths=0,
            )
            a.set_title(n)
            a.set_xlabel("dt [mm/c]")
        axes[0].set_ylabel("dP/P [permille]")
        f.suptitle("Final longitudinal phase space")
        f.savefig("phase_space.png", dpi=150)


def _reference_particle():
    return rft.Bunch6d(
        rft.electronmass,
        BUNCH.population,
        BUNCH.charge,
        numpy.array([0, 0, 0, 0, 0, LINAC.initial_momentum_mev]),
    )


def _report(title, model):
    i = model.simulator.tracked_bunch.get_info()
    print(
        f"{title}: P = {i.mean_P:.1f} MeV/c, sigma_P/P = {i.sigma_pt:.2f} permille,"
        + f" emittance x/y = {i.emitt_x:.2f}/{i.emitt_y:.2f} mm mrad"
    )


def _rf_cell_length():
    """A 2pi/3 cell: a third of the RF wavelength [m]."""
    return rft.clight / LINAC.frequency_hz / 3


def _run(title, model):
    """Print the tracked bunch's summary, and return what the plots need:
    the transport table (s, P, <x>, emittance x) and the final
    longitudinal phase space (dt, P)."""
    _report(title, model)
    return PKDict(
        longitudinal=model.simulator.tracked_bunch.get_phase_space("%dt %P"),
        table=model.simulator.beamline.get_transport_table(
            "%S %mean_P %mean_x %emitt_x"
        ),
    )


def _structure():
    """A 2pi/3 traveling-wave structure with its short-range wakefield
    (tutorial 13's iris geometry)."""
    s = rft.TW_Structure(
        LINAC.gradient_v_per_m,
        0,
        LINAC.frequency_hz,
        2 * numpy.pi / 3,
        LINAC.structure_cells,
    )
    c = _rf_cell_length()
    # iris radius, gap, cell length [m]
    s.add_collective_effect(
        rft.ShortRangeWakefield(
            0.5 * (6.3 + 4.7) / 1e3 * 0.5, c - 0.5 * (1.67 + 1) / 1e3, c
        )
    )
    s.set_nsteps(100)
    s.set_cfx_nsteps(10)
    # ten transport-table rows per structure, for smooth plots
    s.set_tt_nsteps(10)
    return s


if __name__ == "__main__":
    main()
