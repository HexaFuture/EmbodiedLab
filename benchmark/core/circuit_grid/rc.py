"""rc_capacitance on the circuit board: the DC supply, a sealed box with one capacitor between its terminals A and B
(C not used), series resistor blocks 1k/10k/100k (+-0.1 %), the changeover block CO (a three-node block: its lever
connects the common C to T1 or T2, break before make), ten wires, the two voltmeters, the ammeter, the logger channels
and the logger unit. The capacitor is ideal (no leakage) and starts discharged; it keeps its charge while nothing
connects A and B.
"""
from __future__ import annotations

from . import layout as L
from .variants import Profile, build_scene, register

RS = {'R1k': 1000.0, 'R10k': 10000.0, 'R100k': 100000.0}
RS_TAG = {'R1k': '1k', 'R10k': '10k', 'R100k': '100k'}
SUPPLY_V_NOMINAL, SUPPLY_LIMIT_A = 5.0, 0.10
CO_SEAT = ((L.TRAY_ROW_X[3], (L.TRAY_COL_Y[3] + L.TRAY_COL_Y[4]) / 2), L.TRAY_YAW_DEG)


@register
class RcProfile(Profile):
    recipe = 'circuit_grid_rc'
    blocks = {**{k: L.BLOCKS[k] for k in ('V1', 'V2', 'A', 'CH1', 'CH2', 'R1k', 'R10k', 'R100k')},
              **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 11)}}
    three = {'CO': ('changeover', 'C/O', 'changeover', 0.080)}
    tray_layout = (('V1', 'V2', 'A', 'CH1', 'CH2'),
                   ('R1k', 'R10k', 'R100k', 'W1', 'W2'),
                   ('W3', 'W4', 'W5', 'W6', 'W7'),
                   ('W8', 'W9', 'W10', None, None))          # CO lies in the long well over the last two positions
    supply = True
    dut = 'box'
    terminals = {'A': L.BOX_TERMINALS['A'], 'B': L.BOX_TERMINALS['B']}
    box_terminals = ('A', 'B')

    def configuration(self, instance):
        from instance_episodes import appearance_seed
        h = instance['hidden']
        return {'recipe': self.recipe, 'c_f': float(h['C_F']),
                'block_ohm': {k: v * float(h[f'rs_factor_{RS_TAG[k]}']) for k, v in RS.items()},
                'supply_v': SUPPLY_V_NOMINAL * float(h.get('supply_factor', 1.0)), 'supply_limit_a': SUPPLY_LIMIT_A,
                'cap_v0': 0.0, 'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return [('C1', 'C', L.node_name(L.BOX_TERMINALS['A']), L.node_name(L.BOX_TERMINALS['B']), float(config['c_f']))]

    def device(self, config):
        return {'rs_nominal_ohm': {k: v for k, v in RS.items()}, 'supply_v_nominal': SUPPLY_V_NOMINAL,
                'supply_limit_a': SUPPLY_LIMIT_A, 'box_elements': ['C1']}

    def three_seats(self):
        return {'CO': CO_SEAT}

    def build(self, stage):
        return build_scene(stage, self, build_supply_and_box)

    def public_apparatus(self):
        looks = {'voltmeter': L.BLOCK_LOOKS['voltmeter'],
                 'ammeter': L.BLOCK_LOOKS['ammeter'],
                 'channel': 'red band at the + end of the base', 'wire': 'grey, no resistance',
                 'switch': 'blue, with a rocker printed I and O on top',
                 'resistor': 'tan handle printed with its value', 'changeover': 'grey three-pin block with a yellow '
                 'lever on top; its pins are marked T1, C and T2'}
        handles = {'CH1': 'yellow handle', 'CH2': 'cyan handle'}
        blocks = {b: {'kind': k, 'printed': p or 'none', 'looks': handles.get(b, looks[k]), 'tray_seat': b,
                      'polarised': k in L.POLARISED} for b, (k, p, _, _) in self.blocks.items()}
        blocks['CO'] = {'kind': 'changeover', 'printed': 'T1, C, T2 at its pins', 'looks': looks['changeover'],
                        'tray_seat': 'CO', 'polarised': False}
        seats = {b: {'printed': p or 'W', 'takes': c} for b, (_, p, c, _) in self.blocks.items()}
        seats['CO'] = {'printed': 'C/O', 'takes': 'changeover'}
        return {
            'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on '
                              'its edges',
                      'connection': 'everything with a pin or a terminal on one pad is connected there; pads are '
                                    'otherwise insulated from each other'},
            'fixed_terminals': {'supply_plus': 'red strap from the supply + terminal to a board pad',
                                'supply_minus': 'black strap from the supply - terminal to a board pad',
                                'box_A': 'red strap from box terminal A to a board pad',
                                'box_B': 'black strap from box terminal B to a board pad'},
            'supply': {'what': 'grey unit printed DC SUPPLY with an ON/OFF push button and ON and CC lamps'},
            'box': {'what': 'sealed near-black box printed BOX with terminals A and B'},
            'blocks': blocks,
            'tray': {'what': 'green-grey tray with one labelled well per block (a long well for CO)', 'seats': seats},
            'logger': {'what': 'dark blue two-channel logger unit printed LOGGER',
                       'keys': [L.LOGGER_KEY_PRINT.get(k, k) for row in L.LOGGER_KEYS for k in row if k is not None],
                       'display': ['CH1 live (V)', 'SET (s)', 'RATE (Hz)', 'NO', 'LEFT', 'REC lamp', 'ERR lamp']},
        }


def build_supply_and_box(stage, material, key_material):
    """The platform supply (returns its button) and the box with terminals A and B (layout.BOX_TERMINALS is A, B while
    this profile is applied)."""
    from . import asset as A
    A.build_supply(stage, material, key_material)
    A.build_box(stage, material)
    return A.SUPPLY_BUTTON
