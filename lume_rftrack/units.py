"""The one canonical RF-Track attribute -> unit table

Covers every attribute `lume_rftrack.archive` writes (each element's
`get_X()`/`set_X()` pairs, plus its constructor arguments and Volume
position) and every action variable `lume_rftrack.model.build_model()`
makes, so the archive and the served variables can't disagree about a
unit. Follows `lume-impact`'s pattern (`impact/parsers.py`'s
`HEADER_UNITS`/`ELE_UNITS`): plain unit strings, used directly as a
variable's `unit=` and converted to a `pmd_unit` only where one is
needed (writing the archive, `archive._write_field_units_h5`).

Every value is a unit symbol `beamphysics.units.pmd_unit(symbol)`
resolves on its own, including RF-Track's non-SI "deg", "mm/c" and
"MV/c" (a momentum per charge, i.e. a rigidity). `None` means RF-Track
doesn't document the unit, the value has none (a string), or its
elements have different units (a multipole's per-order coefficients).

A name RF-Track uses with different units on different classes has a
`CLASS_UNITS` entry for each class that differs from `UNITS`.
"""

from pykern.pkcollections import PKDict

# Class name -> the attributes whose unit differs from `UNITS` for that
# class (and its subclasses). Quadrupole and Sextupole strengths are
# P_over_Q * k * L (their docstrings), so MV/c/m and MV/c/m^2.
CLASS_UNITS = PKDict(
    Corrector=PKDict(strength="T*mm"),
    Quadrupole=PKDict(strength="MV/c/m"),
    Sextupole=PKDict(gradient="T/m^2", strength="MV/c/m^2"),
)

UNITS = PKDict(
    B0="T",
    Bfield="T",
    # per-order coefficients: T/m^(n-1)
    Bn=None,
    Br="T",
    Brho="T*m",
    Bz="T",
    E1="rad",
    E1d="deg",
    E2="rad",
    E2d="deg",
    Er="V/m",
    Ez="V/m",
    K0="1/m",
    K1="1/m^2",
    K1L="1/m",
    M2="1",
    M2x="1",
    M2y="1",
    P_actual="W",
    P_map="W",
    P_over_Q="MV/c",
    # in units of e, which pmd_unit has no symbol for
    Q=None,
    R="m",
    R_max="m",
    R_min="m",
    angle="rad",
    angled="deg",
    aperture="m",
    aperture_x="m",
    aperture_y="m",
    cell_length="m",
    cfx_nsteps="1",
    coefficients="V/m",
    current_density="A/m^2",
    direction="1",
    entrance_aperture="m",
    exit_aperture="m",
    # an SBend's fringe field integral, measured by archive._fint()
    fint="1",
    frequency="Hz",
    gradient="T/m",
    h="1/m",
    height="m",
    hgap="m",
    hr="m",
    hx="m",
    hy="m",
    hz="m",
    length="m",
    mass="MeV/c^2",
    max_current="A",
    mu=None,
    n_cells="1",
    nsteps="1",
    odeint_algorithm=None,
    odeint_epsabs=None,
    odeint_epsrel="1",
    phi="rad",
    phid="deg",
    position=None,
    resolution="mm",
    # not in the reference manual -- inferred by analogy with every other
    # RF-Track time-like quantity (t0, set_time_window(), sigma_t), all
    # documented as "mm/c".
    resolution_t="mm/c",
    sc_nsteps="1",
    scaling_factor="1",
    shape=None,
    static_Bfield="T",
    static_Efield="V/m",
    # differs by class, see CLASS_UNITS
    strength=None,
    # per-order coefficients: MV/c/m^(n-1)
    strengths=None,
    t0="mm/c",
    temperature=None,
    tt_nsteps="1",
    width="m",
    x="m",
    x0="m",
    xa="m",
    y="m",
    y0="m",
    ya="m",
    z="m",
    z0="m",
    z1="m",
    zR=None,
    zRx=None,
    zRy=None,
)


def unit(element_class, attribute):
    """The unit symbol of `attribute` on an RF-Track `element_class` (a
    class, e.g. `type(element)`): the first `CLASS_UNITS` entry along its
    MRO, else `UNITS`.

    Raises:
        KeyError: `attribute` is in neither table
    """
    for c in element_class.__mro__:
        u = CLASS_UNITS.get(c.__name__)
        if u is not None and attribute in u:
            return u[attribute]
    return UNITS[attribute]
