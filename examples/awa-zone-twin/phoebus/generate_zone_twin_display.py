"""Generate `zone_twin.bob`, a to-scale, clickable synoptic of the AWA
zone1->zone2 twin served by `twin/serve.py`.

Each element is drawn as a colored button at its real position
along the beamline; clicking it selects that element in the editor panel
beside the screen images. The panel holds one `embedded` widget per
element, all at the same spot, each showing a small, reusable, generic
editor `.bob` for that device *kind* (`solenoid_controls.bob`,
`cavity_controls.bob`, `corrector_controls.bob`, `quad_controls.bob`),
parameterized by macros carrying that element's own served PV names --
the same file serves every solenoid, cavity, etc., just loaded with
different macro values. A button writes its element's name to the local
PV `loc://selected_element`, and a rule on each embedded widget shows it
only while that PV holds its element's name. A screen's button writes its name to
`loc://selected_screen` instead, and a rule on the `ScreenTabs` navtabs
selects that screen's tab, which shows `screen_view.bob`, an `image`
widget on its own `<name>_image` PV (a 2D histogram each zone computes
from `get_bunch_at_screens()`). A bare drift is not drawn at all.

Positions, widths and kinds come straight from the beamlines
`zone_model`'s own loaded `ZONE1_MODULE`/`ZONE2_MODULE` build -- the same
ones `zone_model.py` tracks -- each element's kind from its RF-Track
class. A zone1 element (in a Volume, placed at arbitrary, even
overlapping z) is drawn at its absolute position; a zone2 element (in a
Lattice, sequential) at its center. The macro values (which PV each
button's editor controls) are named by
`lume_rftrack.model.variable_name()`, as `build_model()` and the zone
models name their variables. So the display follows the code on both
counts. Run `python generate_zone_twin_display.py` to
regenerate; do not edit the `.bob` files by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

import RF_Track as rft
from pykern.pkcollections import PKDict

import lume_rftrack.model
import lume_rftrack.units

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twin"))
import actions
from zone_model import (
    SCREEN_FOV_MM,
    ZONE1_MODULE,
    ZONE2_MODULE,
    screen_names,
)

ZONE1_BEAMLINE = ZONE1_MODULE.build_beamline()
ZONE2_BEAMLINE = ZONE2_MODULE.build_beamline()

DEFAULT_PREFIX = (
    "AWA:SIM:"  # the PV prefix serve.py uses; the display macro P replaces it
)

X0, AXIS_Y, WIDTH_PX = 30, 160, 1200
# The floor plan's own left edge. The first elements are centered near
# s = 0, so they and their rotated name labels extend about 44 px left of
# the axis start: this shifts them right, off the window edge.
LATTICE_X0 = X0 + 36

# screen_view.bob's own canvas size and the ScreenTabs navtabs' tab-button
# column width -- the navtabs widget's own width/height must be at least
# tab_width plus this canvas size, or the embedded display is clipped to
# whatever smaller area navtabs actually gives it (confirmed live: it
# does not scale the embedded display down to fit, it crops it).
_SCREEN_VIEW_WIDTH = 500
_SCREEN_VIEW_HEIGHT = 540
_SCREEN_TAB_WIDTH = 140

COLORS = PKDict(
    cavity=(220, 60, 60),
    solenoid=(60, 120, 220),
    corrector=(200, 60, 200),
    quadrupole=(255, 140, 0),
    screen=(0, 190, 120),
)

# RF-Track class -> this display's drawing group (color/editor). Anything
# not listed (Drift, Bpm, SBend, ...) isn't drawn at all.
_CLASS_GROUP = {
    "Corrector": "corrector",
    "Quadrupole": "quadrupole",
    "RF_FieldMap_1d": "cavity",
    "Screen": "screen",
    "Static_Magnetic_FieldMap_1d": "solenoid",
}

# The PV the element buttons write and the embedded editors' rules watch.
_SELECTED_PV = 'loc://selected_element("")'

# The PV the screen buttons write and the ScreenTabs navtabs watches. Not
# _SELECTED_PV: a rule with no true expression resets its property, so
# selecting an element would reset the tab to the first screen.
_SELECTED_SCREEN_PV = 'loc://selected_screen("YAG1")'

_EDITOR_WIDTH = 370

# group -> (editor title, editor file, [(label, macro, parameter, format), ...]).
# Each macro is the element's variable for that parameter
# (_element_macros()).
_EDITORS = {
    "solenoid": (
        "Solenoid",
        "solenoid_controls.bob",
        [("Current", "CURRENT", "current", "decimal")],
    ),
    # POWER is RF_Track's own P_actual, in W: the field amplitude
    # scales as sqrt(P_actual / P_map). (build_model() has no live
    # setter for an absolute V/m value.)
    "cavity": (
        "Cavity",
        "cavity_controls.bob",
        [
            ("Power", "POWER", "P_actual", "decimal"),
            ("Phase", "PHASE", "phid", "decimal"),
        ],
    ),
    "corrector": (
        "Corrector",
        "corrector_controls.bob",
        [
            ("Horizontal", "H", "strength_x", "decimal"),
            ("Vertical", "V", "strength_y", "decimal"),
        ],
    ),
    "quadrupole": (
        "Quadrupole",
        "quad_controls.bob",
        [
            ("Strength K1", "K1", "K1", "decimal"),
            ("Gradient", "GRADIENT", "gradient", "decimal"),
        ],
    ),
}


# group -> the RF-Track class its editor's units are looked up for
# (lume_rftrack.units.unit())
_GROUP_CLASS = {g: c for c, g in _CLASS_GROUP.items()}


def _element_macros(group, name):
    """`{macro: variable name}` for element `name`'s own editor. `NAME` is
    the element's own name, shown in the editor's title. Both zones'
    cavities are field maps, so each has a `P_actual`.
    """
    res = {"NAME": name}
    for _, macro, parameter, _ in _EDITORS[group][2]:
        res[macro] = lume_rftrack.model.variable_name(name, parameter)
    return res


def _active_tab_rule(names):
    """A `rules` block that sets a `navtabs`' `active_tab` to the index of
    the tab in `names` that `_SELECTED_SCREEN_PV` equals.
    """
    exps = "".join(
        f'        <exp bool_exp="pvStr0 == {escape(chr(34) + n + chr(34), {chr(34): "&quot;"})}">\n'
        f"          <value>{i}</value>\n"
        f"        </exp>\n"
        for i, n in enumerate(names)
    )
    return (
        f"    <rules>\n"
        f'      <rule name="SelectedScreen" prop_id="active_tab" out_exp="false">\n'
        f"{exps}"
        f"        <pv_name>{escape(_SELECTED_SCREEN_PV)}</pv_name>\n"
        f"      </rule>\n"
        f"    </rules>\n"
    )


def _visible_rule(shown_for):
    """A `rules` block that shows its widget while `_SELECTED_PV` equals
    `shown_for` and hides it otherwise.
    """
    q = escape(f'"{shown_for}"', {'"': "&quot;"})
    return (
        f"    <rules>\n"
        f'      <rule name="Selected" prop_id="visible" out_exp="false">\n'
        f'        <exp bool_exp="pvStr0 == {q}">\n'
        f"          <value>true</value>\n"
        f"        </exp>\n"
        f'        <exp bool_exp="pvStr0 != {q}">\n'
        f"          <value>false</value>\n"
        f"        </exp>\n"
        f"        <pv_name>{escape(_SELECTED_PV)}</pv_name>\n"
        f"      </rule>\n"
        f"    </rules>\n"
    )


class _Display:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self._widgets: list[str] = []
        self._names: set[str] = set()

    def _unique(self, name: str) -> str:
        base, index = name, 2
        while name in self._names:
            name, index = f"{base}{index}", index + 1
        self._names.add(name)
        return name

    def label(
        self,
        name,
        text,
        x,
        y,
        w,
        h=22,
        bold=False,
        size=None,
        rotated=False,
        centered=False,
    ):
        font = ""
        if bold or size:
            font = (
                f'\n    <font><font family="Liberation Sans" style="{"BOLD" if bold else "REGULAR"}" '
                f'size="{size or 12.0}"/></font>'
            )
        rot = "\n    <rotation_step>1</rotation_step>" if rotated else ""
        if centered:
            rot += "\n    <horizontal_alignment>1</horizontal_alignment>"
        self._widgets.append(
            f'  <widget type="label" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <text>{escape(text)}</text>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>{font}{rot}\n"
            f"  </widget>"
        )

    def value(self, name, pv, x, y, w, h=22, precision=3, size=None):
        font = (
            f'    <font><font family="Liberation Sans" style="REGULAR" size="{size}"/></font>\n'
            if size
            else ""
        )
        self._widgets.append(
            f'  <widget type="textupdate" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <pv_name>pva://$(P={DEFAULT_PREFIX}){pv}</pv_name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"{font}"
            f"    <precision>{precision}</precision>\n"
            f"    <precision_from_pv>false</precision_from_pv>\n"
            f"    <show_units>false</show_units>\n"
            f"  </widget>"
        )

    def placeholder(self, name, text, x, y, w, h=22):
        """A label shown only while nothing is selected."""
        self._widgets.append(
            f'  <widget type="label" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <text>{escape(text)}</text>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"{_visible_rule('')}"
            f"  </widget>"
        )

    def led(self, name, pv, x, y, w=20, h=20):
        self._widgets.append(
            f'  <widget type="led" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <pv_name>pva://$(P={DEFAULT_PREFIX}){pv}</pv_name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f'    <off_color><color red="0" green="170" blue="0"/></off_color>\n'
            f'    <on_color><color red="255" green="150" blue="0"/></on_color>\n'
            f"  </widget>"
        )

    def rect(self, name, x, y, w, h, rgb):
        r, g, b = rgb
        self._widgets.append(
            f'  <widget type="rectangle" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f'    <background_color><color red="{r}" green="{g}" blue="{b}"/></background_color>\n'
            f"    <line_width>1</line_width>\n"
            f"  </widget>"
        )

    def embedded(self, name, x, y, w, h, file, macros, shown_for):
        """An `embedded` display of `file` with `macros`, visible only
        while `_SELECTED_PV` holds `shown_for`.
        """
        macro_xml = "".join(f"<{k}>{escape(v)}</{k}>" for k, v in macros.items())
        self._widgets.append(
            f'  <widget type="embedded" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <file>{escape(file)}</file>\n"
            f"    <macros>{macro_xml}</macros>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"    <visible>false</visible>\n"
            f"{_visible_rule(shown_for)}"
            f"  </widget>"
        )

    def select_button(self, name, x, y, w, h, rgb, element, tooltip, pv=_SELECTED_PV):
        """A button that writes `element` to `pv`."""
        r, g, b = rgb
        self._widgets.append(
            f'  <widget type="action_button" version="3.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <actions>\n"
            f'      <action type="write_pv">\n'
            f"        <pv_name>{escape(pv)}</pv_name>\n"
            f"        <value>{escape(element)}</value>\n"
            f"        <description>{escape(tooltip)}</description>\n"
            f"      </action>\n"
            f"    </actions>\n"
            f"    <text></text>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f'    <background_color><color red="{r}" green="{g}" blue="{b}"/></background_color>\n'
            f"    <tooltip>{escape(tooltip)}</tooltip>\n"
            f"  </widget>"
        )

    def navtabs(self, name, tabs, x, y, w, h, tab_width=140, tab_height=30):
        """A `navtabs` widget (one embedded display at a time, selected
        from a column of tab buttons -- `direction` defaults to
        `VERTICAL`, which is what makes that column, not a horizontal
        strip): `tabs` is `[(tab_name, file, macros_dict), ...]`. Writing
        a tab's name to `_SELECTED_SCREEN_PV` also selects that tab.
        """
        tab_xml = "".join(
            f"\n      <tab>\n"
            f"        <name>{escape(tab_name)}</name>\n"
            f"        <file>{escape(file)}</file>\n"
            f"        <macros>"
            + "".join(f"<{k}>{escape(v)}</{k}>" for k, v in macros.items())
            + "</macros>\n"
            "        <group_name></group_name>\n"
            "      </tab>"
            for tab_name, file, macros in tabs
        )
        self._widgets.append(
            f'  <widget type="navtabs" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <tabs>{tab_xml}\n    </tabs>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"    <tab_width>{tab_width}</tab_width>\n"
            f"    <tab_height>{tab_height}</tab_height>\n"
            f"{_active_tab_rule([n for n, _, _ in tabs])}"
            f"  </widget>"
        )

    def xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<display version="2.0.0">\n'
            "  <name>AWA zone1-&gt;zone2 twin</name>\n"
            f"  <width>{self.width}</width>\n  <height>{self.height}</height>\n"
            + "\n".join(self._widgets)
            + "\n</display>\n"
        )


# The y of the "zone1 | zone2" label, above the elements' rotated name labels
_BOUNDARY_Y = 30

# Font sizes of the section headers and the text under them
_HEADER_SIZE = 18.0
_TEXT_SIZE = 14.0

# The least space between two drawn elements, and between their left edges
# so their rotated name labels (16 px wide) don't collide
_ELEMENT_GAP_PX = 2
_LABEL_PITCH_PX = 16

# Visual-only width of a drawn cavity, solenoid or screen (a field map's
# real length includes its fringe field, and overlaps its neighbors') -- a
# schematic approximation, not a metrology-grade drawing. Every other
# group is drawn at its element's length.
_DISPLAY_WIDTH_M = {"cavity": 0.3, "solenoid": 0.15, "screen": 0.05}


def _zone1_layout():
    """`[(s_center, width_m, group, name), ...]` for `ZONE1_BEAMLINE`,
    plus zone1's own nominal length (its own
    `SETTINGS.tracking_range_m[1]`, not guessed from the last element's
    position -- a Volume's elements aren't necessarily stored in z
    order). Each element is drawn at its *absolute* z position, its
    Volume placement (`get_offsets()`, in mm).
    """
    layout = []
    for i in range(ZONE1_BEAMLINE.size()):
        e = ZONE1_BEAMLINE[i]
        group = _CLASS_GROUP.get(type(e).__name__)
        if group is None:
            continue
        z = float(e.get_offsets().get_origin().flatten()[2]) / 1000.0
        layout.append((z, _width(group, e), group, e.get_name()))
    return layout, float(ZONE1_MODULE.SETTINGS.tracking_range_m[1])


def _zone2_layout(start_s):
    """`[(s_center, width_m, group, name), ...]` for `ZONE2_BEAMLINE`,
    offset by `start_s` (zone1's own end) to make one continuous axis for
    the whole twin. A Lattice is sequential, so each element starts where
    the previous one ended, and is drawn at its center (a screen, of no
    length, as a marker at its position). A drift is skipped (never drawn).
    """
    layout = []
    end = start_s
    for i in range(ZONE2_BEAMLINE.size()):
        e = ZONE2_BEAMLINE[i]
        group = _CLASS_GROUP.get(type(e).__name__)
        if group is not None:
            w = _width(group, e)
            layout.append((end + w / 2.0, w, group, e.get_name()))
        end += e.get_length()
    return layout, end


def _draw_element(d, x, w, group, name):
    if group == "screen":
        # selects the screen's tab in the ScreenTabs navtabs strip below
        d.select_button(
            f"btn_{name}",
            x,
            AXIS_Y - 30,
            w,
            60,
            COLORS.screen,
            name,
            f"{name}: show its image below",
            pv=_SELECTED_SCREEN_PV,
        )
        d.label(f"lbl_{name}", name, x - 6, AXIS_Y - 95, 16, 56, rotated=True)
        return
    macros = _element_macros(group, name)
    tooltip = f"{name}: " + ", ".join(
        f"{m}={v}" for m, v in macros.items() if m != "NAME"
    )
    d.select_button(f"btn_{name}", x, AXIS_Y - 35, w, 70, COLORS[group], name, tooltip)
    d.label(f"lbl_{name}", name, x - 6, AXIS_Y - 100, 16, 56, rotated=True)


def build() -> str:
    """The display's XML."""
    zone1_elements, zone1_length = _zone1_layout()
    zone2_elements, total_length = _zone2_layout(zone1_length)
    px_per_m = WIDTH_PX / total_length

    d = _Display(LATTICE_X0 + WIDTH_PX + 40, AXIS_Y + 200)

    d.label(
        "Title",
        "AWA zone1 -> zone2 twin (click an element to edit it)",
        X0,
        10,
        700,
        28,
        bold=True,
        size=16.0,
    )
    d.label("StatusLabel", "Twin:", LATTICE_X0 + WIDTH_PX - 220, 12, 50, bold=True)
    d.led("StatusLed", "STATUS", LATTICE_X0 + WIDTH_PX - 170, 12)
    d.value("StatusText", "STATUS", LATTICE_X0 + WIDTH_PX - 140, 10, 130)

    # beam axis, from the first element (zone1 starts before s = 0, and the
    # elements are pushed apart) to the end
    placed = _placed(zone1_elements + zone2_elements, px_per_m)
    axis_x0 = min(LATTICE_X0, *(x for x, _, _, _ in placed))
    axis_x1 = max(LATTICE_X0 + WIDTH_PX, *(x + w for x, w, _, _ in placed))
    d.rect("axis", axis_x0, AXIS_Y - 1, axis_x1 - axis_x0, 2, (0, 0, 0))

    for x, w, group, name in placed:
        _draw_element(d, x, w, group, name)

    # zone1/zone2 boundary marker
    boundary_x = LATTICE_X0 + int(zone1_length * px_per_m)
    d.rect(
        "boundary",
        boundary_x,
        _BOUNDARY_Y + 28,
        1,
        AXIS_Y + 60 - _BOUNDARY_Y - 28,
        (120, 120, 120),
    )
    d.label(
        "BoundaryLabel",
        "zone1 | zone2",
        boundary_x - 80,
        _BOUNDARY_Y,
        160,
        26,
        bold=True,
        size=16.0,
        centered=True,
    )

    # legend
    ly = AXIS_Y + 80
    for i, (group, text) in enumerate(
        [
            ("cavity", "RF cavity"),
            ("solenoid", "solenoid"),
            ("corrector", "corrector"),
            ("quadrupole", "quadrupole"),
            ("screen", "screen"),
        ]
    ):
        lx = X0 + i * 230
        d.rect(f"legend_{group}", lx, ly, 20, 20, COLORS[group])
        d.label(f"legend_text_{group}", text, lx + 28, ly - 2, 195, 24, size=14)

    # per-zone diagnostics (readbacks only -- not editable, so no button/popup)
    y = ly + 36
    rows = 0
    for prefix, label_text, x, diagnostics in (
        ("zone1_", "Zone 1 exit", X0, actions.energy_diagnostic_variables),
        (
            "zone2_",
            "Zone 2 exit",
            X0 + WIDTH_PX // 2,
            actions.beam_diagnostic_variables,
        ),
    ):
        d.label(
            f"{prefix}DiagHeader",
            label_text,
            x,
            y,
            300,
            28,
            bold=True,
            size=_HEADER_SIZE,
        )
        for i, v in enumerate(diagnostics(prefix).values()):
            rows = max(rows, i + 1)
            d.label(
                f"{prefix}DiagL{i}",
                f"{v.label} [{v.unit}]",
                x,
                y + 34 + i * 26,
                200,
                24,
                size=_TEXT_SIZE,
            )
            d.value(
                f"{prefix}DiagV{i}",
                v.name,
                x + 205,
                y + 34 + i * 26,
                120,
                24,
                size=_TEXT_SIZE,
            )
    y += 34 + rows * 26 + 14

    # ScreenTabs: a column of tab buttons (navtabs' own VERTICAL default)
    # selects which screen's image is embedded here, replacing the
    # embedded display in place rather than opening a new window.
    d.label(
        "ScreenTabsHeader",
        "Screen images",
        X0,
        y,
        300,
        28,
        bold=True,
        size=_HEADER_SIZE,
    )
    y += 34
    screens = screen_names(ZONE1_BEAMLINE) + screen_names(ZONE2_BEAMLINE)
    tabs = [
        (
            name,
            "screen_view.bob",
            {
                "CAM": name,
                "XMIN": f"{-SCREEN_FOV_MM[name] / 2:.3f}",
                "XMAX": f"{SCREEN_FOV_MM[name] / 2:.3f}",
                "YMIN": f"{-SCREEN_FOV_MM[name] / 2:.3f}",
                "YMAX": f"{SCREEN_FOV_MM[name] / 2:.3f}",
            },
        )
        for name in screens
    ]
    d.navtabs(
        "ScreenTabs",
        tabs,
        X0,
        y,
        _SCREEN_TAB_WIDTH + _SCREEN_VIEW_WIDTH,
        _SCREEN_VIEW_HEIGHT,
        tab_width=_SCREEN_TAB_WIDTH,
    )

    # The element editor, beside the screen images: every drawn element's
    # editor sits at this one spot, and only the selected one is visible.
    ex = X0 + _SCREEN_TAB_WIDTH + _SCREEN_VIEW_WIDTH + 30
    d.label(
        "EditorHeader",
        "Element editor",
        ex,
        y - 34,
        300,
        28,
        bold=True,
        size=_HEADER_SIZE,
    )
    d.placeholder(
        "EditorHint", "Click an element above to edit it.", ex, y, _EDITOR_WIDTH
    )
    for _, _, group, name in zone1_elements + zone2_elements:
        if group == "screen":
            continue
        _, file, fields = _EDITORS[group]
        d.embedded(
            f"editor_{name}",
            ex,
            y,
            _EDITOR_WIDTH,
            _editor_height(fields),
            file,
            _element_macros(group, name),
            name,
        )

    d.height = y + _SCREEN_VIEW_HEIGHT + 20
    return d.xml()


def _write_editor_templates(out_dir: Path) -> None:
    """The four reusable, generic per-device-kind editor `.bob` files that
    `build()` embeds, plus the screen viewer. Each editor is parameterized
    by macros carrying the PV to control -- not a device name formatted
    into a fixed pattern -- so the same file serves every device of that
    kind.
    """
    templates = {
        file: _editor(group, title, fields)
        for group, (title, file, fields) in _EDITORS.items()
    }
    templates["screen_view.bob"] = _screen_view()
    for filename, xml in templates.items():
        (out_dir / filename).write_text(xml)


def _editor_height(fields) -> int:
    """The height of `_editor(group, title, fields)`."""
    return 50 + 66 * len(fields) + 10


def _editor(group: str, title: str, fields) -> str:
    """A small editor `.bob` with, for each `(label, macro, parameter, format)`
    in `fields`, a label, a text entry (it writes the PV once, on Enter,
    so one edit starts one simulation), its unit, and a readback, all on
    `pva://$(P=AWA:SIM:)$(<macro>)`. `format` is `"decimal"` or
    `"exponential"`, and each unit is that of `group`'s RF-Track class.
    `$(NAME)` (set by the embedding widget) is the
    element's name and appears in the title.
    """
    format_index = {"decimal": 1, "exponential": 2}
    y = 50
    widgets = []
    for label, macro, parameter, fmt in fields:
        pv = f"pva://$(P={DEFAULT_PREFIX})$({macro})"
        widgets.append(
            f"""  <widget type="label" version="2.0.0">
    <name>Label_{macro}</name>
    <text>{escape(label)}</text>
    <x>20</x><y>{y}</y><width>130</width><height>25</height>
  </widget>
  <widget type="textentry" version="3.0.0">
    <name>Entry_{macro}</name>
    <pv_name>{pv}</pv_name>
    <x>155</x><y>{y}</y><width>120</width><height>25</height>
    <format>{format_index[fmt]}</format>
    <precision>4</precision>
    <precision_from_pv>false</precision_from_pv>
    <show_units>false</show_units>
  </widget>
  <widget type="label" version="2.0.0">
    <name>Unit_{macro}</name>
    <text>{escape(_unit(group, parameter))}</text>
    <x>280</x><y>{y}</y><width>70</width><height>25</height>
  </widget>"""
        )
        y += 30
        widgets.append(
            f"""  <widget type="label" version="2.0.0">
    <name>RBLabel_{macro}</name>
    <text>readback:</text>
    <x>20</x><y>{y}</y><width>130</width><height>20</height>
  </widget>
  <widget type="textupdate" version="2.0.0">
    <name>RB_{macro}</name>
    <pv_name>{pv}</pv_name>
    <x>155</x><y>{y}</y><width>120</width><height>20</height>
    <format>{format_index[fmt]}</format>
    <precision>4</precision>
    <precision_from_pv>false</precision_from_pv>
    <show_units>false</show_units>
  </widget>"""
        )
        y += 36
    body = "\n".join(widgets)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<display version="2.0.0">
  <name>{escape(title)} $(NAME)</name>
  <width>{_EDITOR_WIDTH}</width>
  <height>{y + 10}</height>
  <widget type="label" version="2.0.0">
    <name>Title</name>
    <text>{escape(title)}: $(NAME)</text>
    <x>20</x><y>10</y><width>330</width><height>28</height>
    <font><font family="Liberation Sans" style="BOLD" size="16.0"/></font>
  </widget>
{body}
</display>
"""


def _screen_view() -> str:
    """A viewer `.bob` with one `image` widget on
    `pva://$(P=AWA:SIM:)$(CAM)_image`, `$(XMIN)/$(XMAX)/$(YMIN)/$(YMAX)`
    [mm] axis ranges, with `P` as a macro (default `AWA:SIM:`).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<display version="2.0.0">
  <name>$(CAM)</name>
  <width>{_SCREEN_VIEW_WIDTH}</width>
  <height>{_SCREEN_VIEW_HEIGHT}</height>
  <widget type="label" version="2.0.0">
    <name>Title</name>
    <text>$(CAM)</text>
    <x>10</x><y>5</y><width>400</width><height>24</height>
    <font><font family="Liberation Sans" style="BOLD" size="14.0"/></font>
  </widget>
  <widget type="image" version="2.0.0">
    <name>Image</name>
    <pv_name>pva://$(P={DEFAULT_PREFIX})$(CAM)_image</pv_name>
    <x>10</x><y>35</y><width>{_SCREEN_VIEW_WIDTH - 20}</width><height>{_SCREEN_VIEW_HEIGHT - 60}</height>
    <autoscale>true</autoscale>
    <x_axis>
      <visible>true</visible>
      <minimum>$(XMIN)</minimum>
      <maximum>$(XMAX)</maximum>
      <title>x [mm]</title>
    </x_axis>
    <y_axis>
      <visible>true</visible>
      <minimum>$(YMIN)</minimum>
      <maximum>$(YMAX)</maximum>
      <title>y [mm]</title>
    </y_axis>
  </widget>
</display>
"""


def _unit(group, parameter):
    """The unit of `group`'s variable for `parameter` (a vector
    attribute's component, e.g. `strength_x`, has the attribute's), or
    "" for a dimensionless one (e.g. `scaling_factor`). The twin's own
    `current` is in A."""
    if parameter == "current":
        return "A"
    u = lume_rftrack.units.unit(
        getattr(rft, _GROUP_CLASS[group]),
        parameter[:-2] if parameter[-2:] in ("_x", "_y") else parameter,
    )
    return "" if u in (None, "1") else u


def _placed(elements, px_per_m):
    """`[(x, width_px, group, name), ...]` for the layout `elements`, each
    at its true position unless that would overlap the element before it
    (or crowd its rotated name label): it is then pushed right, so the
    first of a group of overlapping elements (e.g. the gun and its
    solenoids) is shown in full and the next follows it. Only the
    overlapping ones move; every element clear of the one before keeps its
    true position, so the distances to the following elements stay right.
    """
    res = []
    prev_left = prev_right = None
    for _, (s_center, width_m, group, name) in sorted(
        enumerate(elements), key=lambda e: (e[1][0] - e[1][1] / 2.0, e[0])
    ):
        w = max(int(width_m * px_per_m), 10)
        x = LATTICE_X0 + int(s_center * px_per_m) - w // 2
        if prev_right is not None:
            x = max(x, prev_right + _ELEMENT_GAP_PX, prev_left + _LABEL_PITCH_PX)
        res.append((x, w, group, name))
        prev_left, prev_right = x, x + w
    return res


def _width(group, element):
    """The drawn width [m] of `element`, in `group`"""
    return _DISPLAY_WIDTH_M.get(group) or element.get_length()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0]) if argv else Path(__file__).parent
    out = out_dir / "zone_twin.bob"
    out.write_text(build())
    _write_editor_templates(out_dir)
    print(f"wrote {out} and its 5 editor/viewer templates in {out_dir}")


if __name__ == "__main__":
    main()
