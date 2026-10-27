"""circuit_blackbox3 on the circuit board: the DC supply, a sealed box with terminals A, B, C holding one of the
public candidate networks (T1: R1 A-C, R2 C-B; T2: R1 A-C, C1 C-B; T4: R1 and C1 in parallel A-C, R2 C-B; values
hidden), series resistor blocks 1k/10k/100k (+-0.1 %), the switch, the changeover CO, nine wires, the two
voltmeters, the ammeter, the logger channels and the logger unit. A box capacitor is ideal (no leakage) and starts
discharged.
"""
from __future__ import annotations

from . import layout as L
from .rc import CO_SEAT, RS, RS_TAG, SUPPLY_LIMIT_A, SUPPLY_V_NOMINAL, build_supply_and_box
from .variants import BOX_TERMINALS, Profile, build_scene, register

NETWORKS = {'T1': (('R1', 'R', 'A', 'C'), ('R2', 'R', 'C', 'B')),
            'T2': (('R1', 'R', 'A', 'C'), ('C1', 'C', 'C', 'B')),
            'T4': (('R1', 'R', 'A', 'C'), ('C1', 'C', 'A', 'C'), ('R2', 'R', 'C', 'B'))}
HIDDEN = {'R1': 'R1_ohm', 'R2': 'R2_ohm', 'C1': 'C1_F'}


@register
class NetworkProfile(Profile):
    recipe = 'circuit_grid_network'
    blocks = {**{k: L.BLOCKS[k] for k in ('V1', 'V2', 'A', 'CH1', 'CH2', 'R1k', 'R10k', 'R100k', 'SW')},
              **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 10)}}
    three = {'CO': ('changeover', 'C/O', 'changeover', 0.080)}
    tray_layout = (('V1', 'V2', 'A', 'CH1', 'CH2'),
                   ('R1k', 'R10k', 'R100k', 'SW', 'W1'),
                   ('W2', 'W3', 'W4', 'W5', 'W6'),
                   ('W7', 'W8', 'W9', None, None))           # CO lies in the long well over the last two positions
    supply = True
    dut = 'box'
    terminals = dict(BOX_TERMINALS)
    box_terminals = ('A', 'B', 'C')

    def configuration(self, instance):
        from instance_episodes import appearance_seed
        h = instance['hidden']
        topology = h['topology']
        return {'recipe': self.recipe, 'topology': topology,
                'elements': {name: float(h[HIDDEN[name]]) for name, *_ in NETWORKS[topology]},
                'block_ohm': {k: v * float(h[f'rs_factor_{RS_TAG[k]}']) for k, v in RS.items()},
                'supply_v': SUPPLY_V_NOMINAL * float(h.get('supply_factor', 1.0)), 'supply_limit_a': SUPPLY_LIMIT_A,
                'cap_v0': 0.0, 'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return [(name, kind, L.node_name(BOX_TERMINALS[a]), L.node_name(BOX_TERMINALS[b]),
                 config['elements'][name]) for name, kind, a, b in NETWORKS[config['topology']]]

    def device(self, config):
        return {'rs_nominal_ohm': dict(RS), 'supply_v_nominal': SUPPLY_V_NOMINAL, 'supply_limit_a': SUPPLY_LIMIT_A,
                'box_elements': [name for name, *_ in NETWORKS[config['topology']]]}

    def three_seats(self):
        return {'CO': CO_SEAT}

    def build(self, stage):
        return build_scene(stage, self, build_supply_and_box)

    def public_apparatus(self):
        from .rc import RcProfile
        out = RcProfile.public_apparatus(self)
        out['fixed_terminals']['box_C'] = 'green strap from box terminal C to a board pad'
        out['box'] = {'what': 'sealed near-black box printed BOX with terminals A, B and C'}
        return out
