"""Action-variable implementations for the AWA zone twin's own extra PVs
-- the ones `lume_rftrack.model.build_model()` doesn't make: zone1's
solenoids' `scaling_factor` (an RF-Track attribute `build_model()` has no
default for), zone2's quadrupoles' normalized K1 (a conversion of
RF-Track's gradient with the beam's rigidity), and each zone's charge/
energy/energy-spread diagnostics and screen images (derived from the
tracked bunch, not a single element attribute at all).

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

# Resolution of a screen's served image (AWA's zones have no modeled
# camera, so this is just a reasonable display resolution).
IMAGE_BINS = 64


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


def diagnostic_variables(prefix):
    """A zone's diagnostics, `<prefix>charge_pC`, `<prefix>energy_MeV` and
    `<prefix>energy_spread_pct`, by name. Each has a display `label`."""
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


def scaling_factor_variables(simulator):
    """One `<name>_scaling_factor` variable per named
    `Static_Magnetic_FieldMap_1d` (a field-map solenoid) in `simulator`'s
    beamline: RF-Track's own live scale of its field."""
    return [
        lume_rftrack.actions.RFTrackWritableScalarVariable(
            name=lume_rftrack.model.variable_name(e.get_name(), "scaling_factor"),
            element_name=e.get_name(),
            attribute="scaling_factor",
            default_value=e.get_scaling_factor(),
            unit=lume_rftrack.units.unit(type(e), "scaling_factor"),
        )
        for e in _named_elements(simulator, rft.Static_Magnetic_FieldMap_1d)
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
