"""Chemistry side of the reaction bench (reaction_yield), independent of Isaac: the kinetics bench's station,
spectrophotometer and cuvettes, plus a dry block heater with four cuvette wells.

S + R -> P, P + R -> Q and S + R -> D run in every cuvette compartment at that cuvette's temperature (reaction_model);
compartments exchange as on the kinetics bench (contact model) until MIX. A cuvette in a block well relaxes to the
block's true temperature, anywhere else to the room; liquid
from a bottle arrives at room temperature and mixes in by volume. The block keys (+1, -1, +10, -10 C) change the set point on their
press edge; the block heats towards it and drifts down by itself (no cooling). The LOG key switches the block's data
logger on and off (its LOG mark lit on the screen while it records).

The world dict of ChemBench.step also names, under 'block', the cuvette in each block well ({well: id | None}).
Records: the public block log, what a dry block's data logger keeps (one row a second while LOG is on: t, SET, BLOCK
display; not which wells hold a cuvette, nor rows from before the logger was switched on) and, private,
each cuvette's temperature history, the wells' occupants and the block's true temperature (the evaluator's heat log,
one row a second from the start).
"""
from __future__ import annotations

from copy import deepcopy

import reaction_model as RM
from chem_bench import ChemBench

BLOCK_KEYS = {'block:up1': 1, 'block:down1': -1, 'block:up10': 10, 'block:down10': -10}
LOG_KEY = 'block:log'
WELLS = (1, 2, 3, 4)
CHEM_DT_S = 0.1                 # the chemistry and heat advance every 0.1 s of simulated time
LOG_EVERY_S = 1.0


class ReactionBench(ChemBench):
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        RM.register_spectra()
        super().__init__(episode_id, config, physics_hz=physics_hz, noise_seed=noise_seed)
        self.hidden = dict(config['kinetics'])                 # reaction_model.HIDDEN values of this instance
        for container in self.containers.values():             # jet, layer exchange and beam band as on the kinetics bench
            container.contact = True
        self.block = RM.Block(self.hidden['block_offset_C'])
        self.temps = {cid: RM.ROOM_C for cid in self.containers}
        self.wells = {w: None for w in WELLS}
        self.chem_ticks = max(1, round(CHEM_DT_S * physics_hz))
        self.log_ticks = round(LOG_EVERY_S * physics_hz)
        self.logging = False        # the block's data logger (LOG key)
        self.block_log = []         # public: the logger's rows
        self.heat_log = []          # private

    def _press(self, button, seats):
        if button in BLOCK_KEYS:
            self.actions.append(dict(tick=self.tick, button=button))
            self.block.nudge(BLOCK_KEYS[button])
        elif button == LOG_KEY:
            self.actions.append(dict(tick=self.tick, button=button))
            self.logging = not self.logging
        else:
            super()._press(button, seats)

    def step(self, tick, world):
        before = {cid: c.volume_mL for cid, c in self.containers.items()}
        super().step(tick, world)
        for cid, c in self.containers.items():                 # liquid from a bottle arrives at room temperature
            added = c.volume_mL - before[cid]
            if added > 1e-9 and c.volume_mL > 0:
                self.temps[cid] = RM.mix_temperature([(before[cid], self.temps[cid]), (added, RM.ROOM_C)])
        self.wells = {w: (world.get('block') or {}).get(w) for w in WELLS}
        if tick % self.chem_ticks == 0:
            self._advance(self.chem_ticks / self.physics_hz)
        if tick % self.log_ticks == 0:
            if self.logging:
                self.block_log.append(dict(t_s=round(self.t_s, 3), set_C=self.block.set_c,
                                           block_C=round(self.block.display_c, 1)))
            self.heat_log.append(dict(t_s=round(self.t_s, 3), set_C=self.block.set_c, block_true_C=round(self.block.true_c, 3),
                                      wells=dict(self.wells), cuvettes={cid: round(t, 3) for cid, t in self.temps.items()}))

    def _advance(self, dt):
        self.block.update(dt)
        in_block = {cid for cid in self.wells.values() if cid is not None}
        for cid, container in self.containers.items():
            env = self.block.true_c if cid in in_block else RM.ROOM_C
            self.temps[cid] = RM.cuvette_temperature(self.temps[cid], env, cid in in_block, dt)
            if not container.layers:
                continue
            container._exchange(dt)
            container.layers = [(name, v, RM.react(comp, self.temps[cid], self.hidden, dt))
                                for name, v, comp in container.layers]

    def block_display(self):
        """What the block's screen shows: SET, BLOCK (0.1 C) and the LOG mark."""
        return {'set_C': self.block.set_c, 'block_C': round(self.block.display_c, 1), 'log': self.logging}

    def export_block(self):
        return deepcopy(self.block_log)

    def private_record(self):
        record = super().private_record()
        record['heat_log'] = deepcopy(self.heat_log)
        record['compositions'] = {cid: [(name, v, dict(comp)) for name, v, comp in c.layers]
                                  for cid, c in self.containers.items()}
        record['block_set_C'] = self.block.set_c
        record['block_logging'] = self.logging
        return record
