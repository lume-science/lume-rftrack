"""Action-variable implementations for the AWA zone twin's own extra PVs
-- the ones `lume_rftrack.model.build_model()` doesn't make: zone1's
solenoids' coil currents (RF-Track's live `scaling_factor` of the field
map, which `build_model()` has no default for), zone2's quadrupoles'
normalized K1 (a conversion of RF-Track's gradient with the beam's
rigidity), and each zone's diagnostics (zone1's charge, energy and energy
spread; zone2's charge, rms sizes and emittances) and screen images
(derived from the tracked bunch, not a single element attribute at all).

`lume_rftrack.actions`'s own action classes
(`RFTrackWritableScalarVariable` etc.) are the generic layer; these are
the AWA-twin-specific one. Each zone registers them on its own
`LUMERFTrackModel`, so like `lume_rftrack.actions`' they take its
`RFTrackSimulator`.
"""

import math
from typing import ClassVar

import numpy
import RF_Track as rft
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import NDVariable, ScalarVariable

import lume_rftrack.actions
import lume_rftrack.model
import lume_rftrack.units

# A field-map solenoid's field map is in gauss, at its reference current:
# `scaling_factor` = current / reference current * GAUSS_TO_TESLA.
GAUSS_TO_TESLA = 1e-4

# Each zone1 solenoid's reference current [A], the one its field map is
# named for (AWA's awa-rftrack-lattices `ZONE1_ELEMENTS`)
SOLENOID_REF_A = {
    "SolBF": 550.0,
    "SolL1": 500.0,
    "SolL2": 500.0,
    "SolL4": 500.0,
    "SolM": 440.0,
}

# Resolution of a screen's served image (AWA's zones have no modeled
# camera, so this is just a reasonable display resolution).
IMAGE_BINS = 64


class BeamInfoVariable(ScalarVariable, ReadOnlyActionMixin):
    """One field of a zone's tracked bunch's `get_info()` (e.g. `sigma_x`
    [mm], `emitt_x` [normalized, mm.mrad]), NaN before there's a bunch to
    measure.
    """

    # the `get_info()` field
    attribute: str
    label: str

    def _get(self, simulator):
        bunch = simulator.tracked_bunch
        if bunch is None:
            return math.nan
        return getattr(bunch.get_info(), self.attribute)


class ChargePcVariable(ScalarVariable, ReadOnlyActionMixin):
    """A zone's total tracked charge, in pC (NaN before there's a bunch
    to measure: zone2's, until zone1 hands it one).
    """

    label: ClassVar[str] = "Charge"

    def _get(self, simulator):
        bunch = simulator.tracked_bunch
        if bunch is None:
            return math.nan
        return abs(bunch.get_total_charge()) / rft.nC * 1000.0


class EnergyMeVVariable(ScalarVariable, ReadOnlyActionMixin):
    """A zone's mean kinetic energy, in MeV (NaN before there's a bunch
    to measure).
    """

    label: ClassVar[str] = "Energy"

    def _get(self, simulator):
        bunch = simulator.tracked_bunch
        if bunch is None:
            return math.nan
        return bunch.get_info().mean_E


class EnergySpreadPctVariable(ScalarVariable, ReadOnlyActionMixin):
    """A zone's energy spread, as a percent of its mean energy (NaN
    before there's a bunch to measure, or if the mean energy is exactly
    0).
    """

    label: ClassVar[str] = "Energy spread (rms)"

    def _get(self, simulator):
        bunch = simulator.tracked_bunch
        if bunch is None:
            return math.nan
        info = bunch.get_info()
        return 100.0 * info.sigma_E / info.mean_E if info.mean_E else math.nan


class QuadrupoleK1Variable(ScalarVariable, WritableActionMixin):
    """A quadrupole's normalized strength K1 [1/m^2] (positive focuses in
    x, whatever the particle's charge). RF-Track's own `gradient` doesn't
    depend on the beam; K1 does, so it is read and set against a fixed
    design rigidity `p_over_q` [MV/c] (`get_K1(P_over_Q)`/
    `set_K1(K1, P_over_Q)`), as a control system's K1 setpoint is.
    """

    element_name: str
    p_over_q: float

    def _get(self, simulator):
        return simulator.find_element(self.element_name).get_K1(self.p_over_q)

    def _set(self, simulator, value):
        simulator.find_element(self.element_name).set_K1(value, self.p_over_q)


class ScreenImageVariable(NDVariable, ReadOnlyActionMixin):
    """A screen's image: a 2D histogram of the beam's transverse position
    there, `fov_mm` [mm] across on each axis, computed when read (RF-Track
    has no screen-image accessor of its own, unlike Cheetah's
    `Screen.reading`). All zeros before there's a bunch.
    """

    fov_mm: float
    # the screen's position among the beamline's screens, as
    # `get_bunch_at_screens()` orders them
    screen_index: int

    def _get(self, simulator):
        if simulator.tracked_bunch is None:
            return numpy.zeros((IMAGE_BINS, IMAGE_BINS))
        return _histogram(
            simulator.beamline.get_bunch_at_screens()[self.screen_index], self.fov_mm
        )


class SolenoidCurrentVariable(ScalarVariable, WritableActionMixin):
    """A field-map solenoid's coil current [A], against the field map's
    reference current `ref_a` [A]: RF-Track's live `scaling_factor` of the
    field is `current / ref_a * GAUSS_TO_TESLA`. The sim carries no other
    current-to-field calibration.
    """

    element_name: str
    ref_a: float

    def _get(self, simulator):
        return _current(simulator.find_element(self.element_name), self.ref_a)

    def _set(self, simulator, value):
        simulator.find_element(self.element_name).set_scaling_factor(
            value / self.ref_a * GAUSS_TO_TESLA
        )


def beam_diagnostic_variables(prefix):
    """A magnetic-transport zone's diagnostics (its energy is fixed by its
    input bunch): `<prefix>charge_pC`, the rms sizes `<prefix>sigma_x_mm`/
    `<prefix>sigma_y_mm` and the normalized emittances
    `<prefix>emitt_x`/`<prefix>emitt_y`, by name. Each has a display
    `label`."""
    return {
        v.name: v
        for v in (
            ChargePcVariable(name=f"{prefix}charge_pC", read_only=True, unit="pC"),
            *(
                BeamInfoVariable(
                    name=f"{prefix}{a}{u}",
                    attribute=a,
                    label=f"{lbl} {a[-1]}",
                    read_only=True,
                    unit=un,
                )
                for lbl, a, u, un in (
                    ("RMS size", "sigma_x", "_mm", "mm"),
                    ("RMS size", "sigma_y", "_mm", "mm"),
                    ("Norm. emittance", "emitt_x", "", "mm.mrad"),
                    ("Norm. emittance", "emitt_y", "", "mm.mrad"),
                )
            ),
        )
    }


def energy_diagnostic_variables(prefix):
    """An accelerating zone's diagnostics, `<prefix>charge_pC`,
    `<prefix>energy_MeV` and `<prefix>energy_spread_pct`, by name. Each has
    a display `label`."""
    return {
        v.name: v
        for v in (
            ChargePcVariable(name=f"{prefix}charge_pC", read_only=True, unit="pC"),
            EnergyMeVVariable(name=f"{prefix}energy_MeV", read_only=True, unit="MeV"),
            EnergySpreadPctVariable(
                name=f"{prefix}energy_spread_pct", read_only=True, unit="%"
            ),
        )
    }


def quadrupole_k1_variables(simulator, p_over_q):
    """One `QuadrupoleK1Variable` per named Quadrupole in `simulator`'s
    beamline, named `<name>_K1` by `lume_rftrack.model.variable_name()`,
    converting with the design rigidity `p_over_q` [MV/c].
    """
    return [
        QuadrupoleK1Variable(
            name=lume_rftrack.model.variable_name(e.get_name(), "K1"),
            element_name=e.get_name(),
            p_over_q=p_over_q,
            default_value=e.get_K1(p_over_q),
            unit=lume_rftrack.units.UNITS.K1,
        )
        for e in _named_elements(simulator, rft.Quadrupole)
    ]


def screen_image_variables(screen_names, fov_mm):
    """One `<name>_image` variable per screen in `screen_names` (the
    beamline's screens, in order), `fov_mm[name]` [mm] across."""
    return {
        f"{n}_image": ScreenImageVariable(
            name=f"{n}_image",
            fov_mm=fov_mm[n],
            screen_index=i,
            shape=(IMAGE_BINS, IMAGE_BINS),
            unit="counts",
            read_only=True,
        )
        for i, n in enumerate(screen_names)
    }


def solenoid_current_variables(simulator):
    """One `<name>_current` variable [A] per named
    `Static_Magnetic_FieldMap_1d` (a field-map solenoid) in `simulator`'s
    beamline, in `SOLENOID_REF_A`."""
    return [
        SolenoidCurrentVariable(
            name=lume_rftrack.model.variable_name(e.get_name(), "current"),
            element_name=e.get_name(),
            ref_a=SOLENOID_REF_A[e.get_name()],
            default_value=_current(e, SOLENOID_REF_A[e.get_name()]),
            unit="A",
        )
        for e in _named_elements(simulator, rft.Static_Magnetic_FieldMap_1d)
    ]


def _current(element, ref_a):
    """`element`'s coil current [A] at its field map's reference `ref_a`."""
    return element.get_scaling_factor() * ref_a / GAUSS_TO_TESLA


def _histogram(bunch, size_mm):
    """2D histogram of `bunch`'s transverse position in
    `[-size_mm/2, size_mm/2]` on each axis, `IMAGE_BINS` square, as counts.
    A `Bunch6dT` (zone1's screens) needs the uppercase `%X %Y` field codes,
    a `Bunch6d` (zone2's) the lowercase `%x %y`.
    """
    xy = bunch.get_phase_space("%X %Y" if isinstance(bunch, rft.Bunch6dT) else "%x %y")
    h = size_mm / 2.0
    res, _, _ = numpy.histogram2d(
        xy[:, 0], xy[:, 1], bins=IMAGE_BINS, range=[[-h, h], [-h, h]]
    )
    return res.astype(numpy.float64)


def _named_elements(simulator, element_class):
    """`simulator`'s beamline's named elements of `element_class`."""
    for i in range(simulator.beamline.size()):
        e = simulator.beamline[i]
        if isinstance(e, element_class) and e.get_name():
            yield e
