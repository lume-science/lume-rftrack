"""Tests for `linac.py`: the flat linac's variables, its momentum, and
that a corrector and the RF phases do what `main()` shows."""

import sys

import pykern.pkio

_EXAMPLE_DIR = pykern.pkio.py_path(__file__).dirpath().dirpath()

_MODEL = None


def test_corrector_moves_only_downstream_bpms():
    import numpy
    from pykern import pkunit

    m = _model()
    b = m.simulator.beamline
    before = b.get_bpm_readings()[:, 0]
    m.set({"C03_strength_x": 0.1})
    d = b.get_bpm_readings()[:, 0] - before
    m.reset()
    # BPM03 is the first BPM after C03
    numpy.testing.assert_array_equal(d[:2], 0.0)
    pkunit.pkok(
        numpy.all(numpy.abs(d[3:]) > 1e-3), "downstream BPMs did not move: {}", d
    )


def test_off_crest_reduces_energy_spread():
    from pykern import pkunit

    m = _model()
    crest = m.simulator.tracked_bunch.get_info()
    m.set({n: -20.0 for n in m.supported_variables if n.endswith("_phid")})
    off = m.simulator.tracked_bunch.get_info()
    m.reset()
    pkunit.pkok(off.mean_P < crest.mean_P, "off crest gained energy: {}", off.mean_P)
    pkunit.pkok(
        off.sigma_pt < crest.sigma_pt / 2,
        "energy spread {} not well below {} on crest",
        off.sigma_pt,
        crest.sigma_pt,
    )


def test_reaches_final_momentum():
    import linac
    from pykern import pkunit

    p = _model().simulator.tracked_bunch.get_info().mean_P
    pkunit.pkok(
        abs(p - linac.LINAC.final_momentum_mev) < 0.05 * linac.LINAC.final_momentum_mev,
        "final momentum {} MeV/c",
        p,
    )


def test_variables():
    from pykern import pkunit

    v = _model().supported_variables
    pkunit.pkeq(
        [24, 7, 7, 7],
        [
            len([n for n in v if n.endswith(s)])
            for s in ("_phid", "_gradient", "_strength_x", "_strength_y")
        ],
    )
    pkunit.pkeq(45, len(v))
    pkunit.pkeq("deg", v["TW24_phid"].unit)
    pkunit.pkeq("T*mm", v["C07_strength_y"].unit)


def _model():
    """The example's model, built once and tracked at its defaults"""
    global _MODEL

    if _MODEL is None:
        sys.path.insert(0, str(_EXAMPLE_DIR))
        import linac

        import lume_rftrack.model

        _MODEL = lume_rftrack.model.build_model(
            linac.build_beamline(), linac.build_bunch()
        )
        _MODEL.reset()
    return _MODEL
