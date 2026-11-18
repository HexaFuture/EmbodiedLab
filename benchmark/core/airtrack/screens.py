"""The two photogate screens (USD authoring and runtime). Each lists the passes of the latest run through its gate.

A screen faces +Y (towards the robot and the fixed cameras). Under the title GATE 1 / GATE 2 it shows ROWS rows, one
per pass in order: the pass index (2 digits), the direction (+ or -) and the pass time in milliseconds with 0.01 ms
resolution (5 + 2 seven-segment digits and a decimal point). With more passes than rows the screen scrolls to the
latest ones (the index says which). Times of 100 s or more show dashes. Segments are lit by visibility only.
"""
from __future__ import annotations

ROWS = 6
DIGIT_H = 0.018
ROW_PITCH = 0.026
HOUSING = (0.25, 0.025, 0.215)
INDEX_X = 0.092            # display-local x of the index field centre (+x is the viewer's left)
SIGN_X = 0.062
TIME_X = -0.028
TIME_DIGITS = 7            # 5 integer + 2 decimal
DOT_GAP = 0.006


def _row_z(k):
    return 0.052 - k * ROW_PITCH


def build_screen(stage, path, centre, title):
    """Author one gate screen centred at `centre` (parent coordinates), face +Y."""
    from pxr import Gf, UsdGeom
    from pendulum_asset import _FONT, _box, _digits, _material, _text
    for char, bits in (('F', (31, 16, 16, 30, 16, 16, 16)), ('H', (17, 17, 17, 31, 17, 17, 17))):
        _FONT.setdefault(char, bits)
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    mats = {'housing': _material(stage, path + '/Materials/Housing', (.006, .012, .02)),
            'face': _material(stage, path + '/Materials/Face', (.002, .004, .003)),
            'ink': _material(stage, path + '/Materials/Ink', (.93, .95, .97)),
            'lit': _material(stage, path + '/Materials/Lit', (.22, 1.0, .58), emission=1.0)}
    _box(stage, path + '/Housing', (0, 0, 0), HOUSING, mats['housing'])
    face_y = HOUSING[1] / 2 + .0006
    _box(stage, path + '/Face', (0, face_y - .0003, -.012), (HOUSING[0] - .014, .0004, HOUSING[2] - .048), mats['face'])
    _text(stage, path + '/Title', title, (0, face_y + .001, .088), .013, mats['ink'])
    _text(stage, path + '/Header', 'NO  DIR      T MS', (0, face_y + .001, .071), .0065, mats['ink'])
    _text(stage, path + '/Spec', 'BEAMS 20 MM APART', (0, face_y + .001, -.096), .0060, mats['ink'])
    y = face_y + .0012
    sp = DIGIT_H * .68
    for k in range(ROWS):
        z = _row_z(k)
        row = f'{path}/Row{k}'
        UsdGeom.Xform.Define(stage, row)
        _digits(stage, row + '/Index', (INDEX_X, y, z), 2, DIGIT_H, mats['lit'])
        _box(stage, row + '/SignH', (SIGN_X, y, z), (DIGIT_H * .45, .0005, DIGIT_H * .1), mats['lit'])
        _box(stage, row + '/SignV', (SIGN_X, y, z), (DIGIT_H * .1, .0005, DIGIT_H * .45), mats['lit'])
        _digits(stage, row + '/Time', (TIME_X, y, z), TIME_DIGITS, DIGIT_H, mats['lit'], separator_after=5,
                separator_gap=DOT_GAP)
        dot_x = TIME_X + (TIME_DIGITS - 1) * sp / 2 - 4.5 * sp
        _box(stage, row + '/Dot', (dot_x, y, z - DIGIT_H / 2 + .0012), (.0024, .0005, .0024), mats['lit'])
    return path


def row_text(entry):
    """(index text, sign or None, time text) of one screen row; entry (index, direction, dt_ms) or None (blank)."""
    if entry is None:
        return '  ', None, ' ' * TIME_DIGITS
    index, direction, dt_ms = entry
    hundredths = int(round(dt_ms * 100))
    time = f'{hundredths:7d}' if hundredths < 10 ** TIME_DIGITS else '-' * TIME_DIGITS
    # keep a leading zero before the decimal point (0.xx ms never occurs in practice, but stay legible)
    time = time[:4] + time[4].replace(' ', '0') + time[5:].replace(' ', '0')
    return f'{index % 100:2d}', direction, time


class Screens:
    """Runtime: shows {gate: [(index, direction, dt_ms), ...]} on both screens, touching only rows that changed."""

    def __init__(self, stage, paths):
        self.stage, self.paths = stage, dict(paths)
        self.shown = {}

    def _visible(self, prim_path, on):
        from pxr import UsdGeom
        UsdGeom.Imageable(self.stage.GetPrimAtPath(prim_path)).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)

    def update(self, content):
        from pendulum_asset import _set_digits
        for gate, path in self.paths.items():
            entries = list(content.get(gate, []))[-ROWS:]
            entries += [None] * (ROWS - len(entries))
            for k, entry in enumerate(entries):
                text = row_text(entry)
                if self.shown.get((gate, k)) == text:
                    continue
                self.shown[(gate, k)] = text
                row = f'{path}/Row{k}'
                _set_digits(self.stage, row + '/Index', text[0])
                _set_digits(self.stage, row + '/Time', text[2])
                self._visible(row + '/SignH', text[1] is not None)
                self._visible(row + '/SignV', text[1] == '+')
                self._visible(row + '/Dot', entry is not None)
