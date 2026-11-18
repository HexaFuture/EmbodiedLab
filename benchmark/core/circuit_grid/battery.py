"""battery_emf on the circuit board: a cell fixed on the board (its + and - terminals strapped to two neighbouring
pads), load resistor blocks 20/10/5/3/2 ohm (+-2 %), shunt blocks 0.1/0.5/1.0 ohm (+-0.1 %), a switch, six wires,
the two voltmeters, the ammeter, the logger channels and the logger unit. No DC supply and no black box.
The cell is an EMF E in series with its internal resistance r (hidden), ideal otherwise (no drift, no heating).
"""
from __future__ import annotations

from . import layout as L
from .variants import Profile, build_scene, register

LOADS = {'L20': 20.0, 'L10': 10.0, 'L5': 5.0, 'L3': 3.0, 'L2': 2.0}
SHUNTS = {'S01': 0.1, 'S05': 0.5, 'S10': 1.0}
LOAD_PRINT = {'L20': '20', 'L10': '10', 'L5': '5', 'L3': '3', 'L2': '2'}
SHUNT_PRINT = {'S01': '0.1', 'S05': '0.5', 'S10': '1.0'}
SHUNT_COLOUR = (0.78, 0.45, 0.22)            # copper handle
CELL_BOX = L.SUPPLY_BOX                      # the cell holder stands where the resistor_blackbox supply stands
CELL_COLOUR = (0.12, 0.13, 0.15)
CELL_BODY = (0.10, 0.36, 0.62)
CELL_NOMINAL_V = 1.5


@register
class BatteryProfile(Profile):
    recipe = 'circuit_grid_battery'
    blocks = {**{k: L.BLOCKS[k] for k in ('V1', 'V2', 'A', 'CH1', 'CH2', 'SW')},
              **{k: ('resistor', LOAD_PRINT[k], k, 0.040) for k in LOADS},
              **{k: ('shunt', SHUNT_PRINT[k], k, 0.040) for k in SHUNTS},
              **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 7)}}
    tray_layout = (('V1', 'V2', 'A', 'CH1', 'CH2'),
                   ('L20', 'L10', 'L5', 'L3', 'L2'),
                   ('S01', 'S05', 'S10', 'SW', 'W1'),
                   ('W2', 'W3', 'W4', 'W5', 'W6'))
    supply = False
    dut = 'cell'
    terminals = {'+': L.SUPPLY_PLUS, '-': L.SUPPLY_MINUS}
    ground = L.SUPPLY_MINUS
    colours = {'shunt': SHUNT_COLOUR}

    def configuration(self, instance):
        from instance_episodes import appearance_seed
        h = instance['hidden']
        tag = {'S01': '01', 'S05': '05', 'S10': '10'}
        ohm = {k: v * float(h[f'load_factor_{int(v)}']) for k, v in LOADS.items()}
        ohm.update({k: v * float(h[f'shunt_factor_{tag[k]}']) for k, v in SHUNTS.items()})
        return {'recipe': self.recipe, 'emf_v': float(h['E_V']), 'r_ohm': float(h['r_ohm']), 'block_ohm': ohm,
                'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return [('cell', 'cell', L.node_name(L.SUPPLY_PLUS), L.node_name(L.SUPPLY_MINUS),
                 (float(config['emf_v']), float(config['r_ohm'])))]

    def device(self, config):
        return {'load_nominal_ohm': dict(LOADS), 'shunt_nominal_ohm': dict(SHUNTS), 'cell_nominal_v': CELL_NOMINAL_V}

    def build(self, stage):
        return build_scene(stage, self, build_cell)

    def public_apparatus(self):
        looks = {'voltmeter': L.BLOCK_LOOKS['voltmeter'],
                 'ammeter': L.BLOCK_LOOKS['ammeter'],
                 'channel': 'red band at the + end of the base', 'switch': 'blue, with a rocker printed I and O on top',
                 'wire': 'grey, no resistance', 'resistor': 'tan handle printed with its value in ohm',
                 'shunt': 'copper handle printed with its value in ohm'}
        handles = {'CH1': 'yellow handle', 'CH2': 'cyan handle'}
        return {
            'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on '
                              'its edges',
                      'connection': 'everything with a pin or a terminal on one pad is connected there; pads are '
                                    'otherwise insulated from each other'},
            'fixed_terminals': {'cell_plus': 'red strap from the cell + terminal to a board pad',
                                'cell_minus': 'black strap from the cell - terminal to a board pad'},
            'cell': {'what': 'cell holder printed CELL with a blue cell printed 1.5 V; + and - marks at its straps'},
            'blocks': {b: {'kind': k, 'printed': p or 'none', 'looks': handles.get(b, looks[k]), 'tray_seat': b,
                           'polarised': k in L.POLARISED} for b, (k, p, _, _) in self.blocks.items()},
            'tray': {'what': 'green-grey tray with one labelled well per block',
                     'seats': {b: {'printed': p or 'W', 'takes': c} for b, (_, p, c, _) in self.blocks.items()}},
            'logger': {'what': 'dark blue two-channel logger unit printed LOGGER',
                       'keys': [L.LOGGER_KEY_PRINT.get(k, k) for row in L.LOGGER_KEYS for k in row if k is not None],
                       'key_travel_m': L.TRAVEL_M, 'key_acts_at_m': L.SWITCH_DOWN_M,
                       'display': ['CH1 live (V)', 'SET (s)', 'RATE (Hz)', 'NO', 'LEFT', 'REC lamp', 'ERR lamp']},
        }


def build_cell(stage, material, key_material):
    """The cell holder (static housing) with a blue cell lying on it, and its terminal straps to the board."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    from .display import Frame, INK, WHITE, text
    A._housing(stage, A.ROOT + '/Cell/Housing', CELL_BOX, CELL_COLOUR, material)
    (x0, x1), (y0, y1), h = CELL_BOX
    top = L.TABLE_Z + h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    body = UsdGeom.Cylinder.Define(stage, A.ROOT + '/Cell/Body')
    body.CreateRadiusAttr(0.017)
    body.CreateHeightAttr(0.090)
    body.CreateAxisAttr('Y')
    body.CreateExtentAttr([Gf.Vec3f(-0.017, -0.045, -0.017), Gf.Vec3f(0.017, 0.045, 0.017)])
    UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(cx + 0.012, cy, top + 0.017))
    body.CreateDisplayColorAttr([Gf.Vec3f(*CELL_BODY)])
    cap = UsdGeom.Cylinder.Define(stage, A.ROOT + '/Cell/PlusCap')
    cap.CreateRadiusAttr(0.006)
    cap.CreateHeightAttr(0.006)
    cap.CreateAxisAttr('Y')
    cap.CreateExtentAttr([Gf.Vec3f(-0.006, -0.003, -0.006), Gf.Vec3f(0.006, 0.003, 0.006)])
    UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(cx + 0.012, y1 - 0.002, top + 0.017))
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.75, 0.78)])
    frame = lambda x, y: Frame((x, y), A.READ, A.UP, top)       # noqa: E731
    text(stage, A.ROOT + '/Cell/Title', 'CELL', frame(x0 + 0.012, cy), 0, 0, 0.009, WHITE, align='centre')
    text(stage, A.ROOT + '/Cell/Nominal', '1.5 V', Frame((cx + 0.012, cy), A.READ, A.UP, top + 0.0345), 0, 0, 0.009,
         WHITE, align='centre')
    for node, colour, mark in ((L.SUPPLY_PLUS, A.RED, '+'), (L.SUPPLY_MINUS, (0.03, 0.03, 0.03), '-')):
        nx, ny = L.node_xy(node)
        name = 'Plus' if mark == '+' else 'Minus'
        A._strap(stage, A.ROOT + f'/Cell/Strap{name}', x1, nx - L.PAD_HALF + 0.004, ny, colour)
        text(stage, A.ROOT + f'/Cell/Mark{name}', mark, frame(x1 - 0.008, ny), 0, 0, 0.009,
             A.RED if mark == '+' else WHITE, align='centre')
    del INK
    return None
