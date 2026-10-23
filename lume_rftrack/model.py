"""LUME interface for RF-Track

RF-Track is an in-process Python simulation library with no text-based
lattice file format, so this builds on ``lume.actions.ActionModel``
directly: the caller builds a real ``RF_Track.Lattice``/``RF_Track.Volume``
and initial bunch themselves via RF-Track's native API, then wraps them in
:class:`lume_rftrack.simulator.RFTrackSimulator` and a list of
:class:`lume_rftrack.actions.RFTrackWritableScalarVariable`/
:class:`lume_rftrack.actions.RFTrackReadOnlyScalarVariable` instances
naming the element attributes to expose -- or lets :func:`build_model`
make them from the beamline's own elements.

Element attribute access uses RF-Track's own paired ``get_X()``/``set_X()``
methods (e.g. ``get_gradient()``/``set_gradient()``), not plain mutable
attributes.
"""

import numpy
from lume.actions import ActionModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn
from pykern.pkcollections import PKDict

import lume_rftrack.units
import lume_rftrack.utils
from lume_rftrack.actions import (
    RFTrackWritableScalarVariable,
    RFTrackWritableVectorComponentVariable,
)
from lume_rftrack.simulator import RFTrackSimulator

# RF-Track class -> the attributes `build_model()` makes a variable for,
# the control-room knobs: a magnet's field, an RF element's phase and
# power. A class not listed here uses its nearest listed base class
# (`RF_Field` covers every RF element), and a listed attribute the element
# has no get_X()/set_X() pair for is skipped (e.g. a Pillbox_Cavity's
# P_actual). A vector attribute has one variable per component.
CLASS_ATTRIBUTES = PKDict(
    Corrector=("strength",),
    Quadrupole=("gradient",),
    RF_Field=("phid", "P_actual"),
    SBend=("angle",),
    Sextupole=("gradient",),
    Solenoid=("max_current",),
)

# Suffixes of a vector attribute's variables, by component, e.g. a
# Corrector's strength_x and strength_y
_COMPONENT_SUFFIXES = ("x", "y", "z")


class LUMERFTrackModel(
    InitialParticlesMixIn, FinalParticlesMixIn, ActionModel[RFTrackSimulator]
):
    """LUME model for RF-Track.

    Every `set()` re-tracks the whole beamline eagerly (RF-Track tracking
    is fast enough for this to be appropriate, unlike more expensive
    PIC-style codes).

    It is also a `lume.staged_model.StagedModel` stage, next to any other
    LUME model with the same particle mixins: it takes its initial bunch
    from the stage before it and hands its tracked bunch to the next, as
    openPMD `ParticleGroup`s. A stage built with no bunch
    (`build_model(beamline, None)`) tracks nothing until one arrives.
    """

    @property
    def final_particles(self):
        """The tracked bunch as a `ParticleGroup`, or `None` before the
        first track."""
        return _particlegroup(self.simulator.tracked_bunch)

    @property
    def initial_particles(self):
        """The initial bunch as a `ParticleGroup`, or `None` if there is
        none yet."""
        return _particlegroup(self.simulator.initial_bunch)

    @initial_particles.setter
    def initial_particles(self, particles):
        """Make `particles` the initial bunch, entering the beamline at
        its entrance (z = 0), then re-track.

        `particles` is first drifted to a plane, so each particle arrives
        there at its own `t`, which keeps the bunch length. A Lattice
        gets that plane as a `Bunch6d`; a Volume gets a `Bunch6dT` whose
        particles each start at z = 0 at their `t`. Times are kept as
        they are, so an RF element's phase is against the same clock as
        the stage before.
        """
        import RF_Track as rft

        if isinstance(self.simulator.beamline, rft.Volume):
            p = particles.copy()
            p.drift_to_z()
            p.z = numpy.zeros_like(p.z)
            b = lume_rftrack.utils.particlegroup_to_bunch6dt(p)
        else:
            b = lume_rftrack.utils.particlegroup_to_bunch6d(particles)
        self.simulator.initial_bunch = b
        self.simulator.track()

    def _set(self, values):
        super()._set(values)
        self.simulator.track()


def build_model(beamline, bunch):
    """`beamline`/`bunch` wrapped as a `LUMERFTrackModel`, with one
    writable action variable per `CLASS_ATTRIBUTES` attribute of each
    named element in `beamline`. Each variable's default_value is read
    directly off the element (RF_Track's own live get_X()), and its unit
    is `lume_rftrack.units.unit()`'s for the element's class.

    Args:
        beamline: the `RF_Track.Volume` or `RF_Track.Lattice` to track
        bunch: the initial `RF_Track` bunch, or `None` for a staged
            model's stage that gets its bunch from the stage before it

    Returns:
        LUMERFTrackModel: wrapping `beamline` and `bunch`

    Raises:
        ValueError: two elements with variables have the same name
    """
    variables = PKDict()
    for i in range(beamline.size()):
        e = beamline[i]
        n = e.get_name()
        if not n:
            continue
        for v in _variables(e, n):
            if v.name in variables:
                raise ValueError(f"more than one element named {n!r} has variables")
            variables[v.name] = v
    return LUMERFTrackModel(
        simulator=RFTrackSimulator(beamline, bunch),
        action_variables=list(variables.values()),
    )


def variable_name(element_name, attribute):
    """The name of the variable for `element_name`'s `attribute`, e.g.
    "Q1_gradient"."""
    return f"{element_name}_{attribute}"


def _attributes(element):
    """`element`'s `CLASS_ATTRIBUTES` that it has get_X()/set_X() for."""
    for c in type(element).__mro__:
        a = CLASS_ATTRIBUTES.get(c.__name__)
        if a is not None:
            return [
                x
                for x in a
                if hasattr(element, "get_" + x) and hasattr(element, "set_" + x)
            ]
    return []


def _particlegroup(bunch):
    """`bunch` as a `ParticleGroup` (`None` stays `None`): a `Bunch6dT` as
    a time snapshot, any other bunch as a fixed-z-plane snapshot."""
    import RF_Track as rft

    if bunch is None:
        return None
    if isinstance(bunch, rft.Bunch6dT):
        return lume_rftrack.utils.bunch6dt_to_particlegroup(bunch)
    return lume_rftrack.utils.bunch6d_to_particlegroup(bunch)


def _variables(element, element_name):
    res = []
    for a in _attributes(element):
        common = PKDict(
            element_name=element_name,
            attribute=a,
            unit=lume_rftrack.units.unit(type(element), a),
        )
        v = getattr(element, "get_" + a)()
        if not isinstance(v, numpy.ndarray):
            res.append(
                RFTrackWritableScalarVariable(
                    name=variable_name(element_name, a), default_value=v, **common
                )
            )
            continue
        for i, x in enumerate(v.ravel()):
            res.append(
                RFTrackWritableVectorComponentVariable(
                    name=variable_name(element_name, f"{a}_{_COMPONENT_SUFFIXES[i]}"),
                    default_value=float(x),
                    index=i,
                    **common,
                )
            )
    return res
