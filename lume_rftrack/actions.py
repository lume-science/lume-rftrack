"""RF-Track element-attribute action variables for
`lume_rftrack.rftrack.LUMERFTrackModel`

Each class exposes one RF-Track element attribute (via its paired
``get_X()``/``set_X()`` methods) as a `lume.variables.Variable`/
`lume.actions.Action`, for use as a `LUMERFTrackModel` action variable --
generic, simulator-level plumbing with no unit conversion or PV-naming
convention of its own (a facility-specific control-system layer, if
one is ever needed, belongs on top of these).

`_RFTrackElementAccessMixin` comes first despite being private, since every
other class here builds on it; the rest are in alphabetical order.
"""

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable


class _RFTrackElementAccessMixin:
    """Shared element-attribute resolution for RF-Track action variables."""

    element_name: str
    attribute: str

    def _get_direct_attribute(self, simulator):
        e = simulator.find_element(self.element_name)
        return getattr(e, f"get_{self.attribute}")()

    def _set_direct_attribute(self, simulator, value):
        e = simulator.find_element(self.element_name)
        getattr(e, f"set_{self.attribute}")(value)


class RFTrackReadOnlyActionMixin(_RFTrackElementAccessMixin, ReadOnlyActionMixin):
    """Read-only action mixin for an RF-Track element attribute."""

    read_only = True

    def _get(self, simulator):
        return self._get_direct_attribute(simulator)


class RFTrackReadOnlyScalarVariable(ScalarVariable, RFTrackReadOnlyActionMixin):
    """Read-only scalar variable for an RF-Track element attribute."""


class RFTrackWritableActionMixin(_RFTrackElementAccessMixin, WritableActionMixin):
    """Writable action mixin for an RF-Track element attribute."""

    def _get(self, simulator):
        return self._get_direct_attribute(simulator)

    def _set(self, simulator, value):
        self._set_direct_attribute(simulator, value)


class RFTrackWritableK1Variable(ScalarVariable, RFTrackWritableActionMixin):
    """Writable scalar variable for a quadrupole's K1 [1/m^2], the
    normalized focusing strength (positive focuses in x, whatever the
    particle's charge). K1 is relative to the beam's rigidity, so it is
    read and set against the simulator's initial bunch (`p_over_q()`),
    not through a plain get_X()/set_X() pair like the other variables.
    """

    def _get(self, simulator):
        e = simulator.find_element(self.element_name)
        return e.get_K1(simulator.p_over_q())

    def _set(self, simulator, value):
        e = simulator.find_element(self.element_name)
        e.set_K1(value, simulator.p_over_q())


class RFTrackWritableScalarVariable(ScalarVariable, RFTrackWritableActionMixin):
    """Writable scalar variable for an RF-Track element attribute."""


class RFTrackWritableVectorComponentVariable(
    ScalarVariable, RFTrackWritableActionMixin
):
    """Writable scalar variable for one component of a vector-valued
    RF-Track element attribute (e.g. ``Corrector.strength`` ->
    ``[hkick, vkick]``).

    The underlying ``get_X()``/``set_X(...)`` pair takes/returns the whole
    vector at once, so a set here first reads the other component(s) back
    to avoid clobbering them.
    """

    index: int

    def _get(self, simulator):
        return self._get_direct_attribute(simulator)[self.index]

    def _set(self, simulator, value):
        e = simulator.find_element(self.element_name)
        current = list(getattr(e, f"get_{self.attribute}")())
        current[self.index] = value
        getattr(e, f"set_{self.attribute}")(*current)
