"""Electrical side of the circuit bench, independent of Isaac.

Three slider selectors (PAIR, RS, MODE) are read from their measured carriage
positions every physics tick. A carriage within the detent tolerance selects
that detent; between detents the selector is in transition and electrically
open (break before make). Moving across an intermediate detent therefore
connects it briefly, and that is logged like any other change. A detent change
takes effect at the tick boundary: the capacitor evolves over a tick with the
switch state read at the previous tick.

Records: actions (every selector change: tick, selector, old, new; public) and
the private state log (capacitor voltage at every switch change) for the
evaluator. Neither the solver nor the state log is visible to the Agent.
"""
import math

from circuit_logger import CircuitLogger
from circuit_solver import MODES, PAIRS, RS_NOMINAL_OHM

SELECTORS = ('PAIR', 'RS', 'MODE')
DETENT_LABELS = {'PAIR': PAIRS, 'RS': RS_NOMINAL_OHM, 'MODE': MODES}
DETENT_SPACING_M = 0.030
DETENT_TOLERANCE_M = 0.004


class Selector:
    """Detent positions 0, spacing, 2*spacing ... along the slider; labels in order."""

    def __init__(self, labels, spacing_m=DETENT_SPACING_M, tolerance_m=DETENT_TOLERANCE_M):
        if not 0 < tolerance_m < spacing_m / 2:
            raise ValueError('detent tolerance must be positive and below half the spacing')
        self.labels = tuple(labels)
        self.spacing_m = spacing_m
        self.tolerance_m = tolerance_m

    def position_of(self, label):
        return self.labels.index(label) * self.spacing_m

    def at(self, position_m):
        if not math.isfinite(position_m):
            raise ValueError('non-finite selector position')
        index = round(position_m / self.spacing_m)
        if 0 <= index < len(self.labels) and abs(position_m - index * self.spacing_m) <= self.tolerance_m:
            return self.labels[index]
        return None


class CircuitBench:
    def __init__(self, episode_id, model, *, initial, physics_hz=240, max_arms=20, noise_seed=0,
                 initial_vc_v=0.0, noise_v=None, detent_labels=None):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.model = model
        self.physics_hz = physics_hz
        # A bench may relabel the three selectors (the LED board: LED, R_s, supply mode); the default is the panel's.
        labels = {name: tuple(detent_labels[name]) for name in SELECTORS} if detent_labels else DETENT_LABELS
        self.selectors = {name: Selector(labels[name]) for name in SELECTORS}
        self.switch = tuple(initial[name] for name in SELECTORS)
        for name, value in zip(SELECTORS, self.switch):
            if value not in labels[name]:
                raise ValueError(f'initial {name} must be a detent')
        self.vc_v = float(initial_vc_v)
        kwargs = {} if noise_v is None else dict(noise_v=noise_v)
        self.logger = CircuitLogger(episode_id, max_arms=max_arms, noise_seed=noise_seed,
                                    physics_hz=physics_hz, **kwargs)
        self.tick = 0
        self.actions = []
        self.state_log = [dict(tick=0, switch=list(self.switch), vc_V=self.vc_v)]

    def initial_positions_m(self):
        return {name: self.selectors[name].position_of(value) for name, value in zip(SELECTORS, self.switch)}

    def step(self, tick, positions_m):
        """One physics tick; positions_m: measured carriage offsets from each slider's first detent."""
        if tick != self.tick + 1:
            raise ValueError('circuit bench requires every consecutive physics tick')
        self.vc_v = self.model.advance(self.switch, self.vc_v, 1.0 / self.physics_hz)
        self.tick = tick
        switch = tuple(self.selectors[name].at(positions_m[name]) for name in SELECTORS)
        if switch != self.switch:
            for name, old, new in zip(SELECTORS, self.switch, switch):
                if old != new:
                    self.actions.append(dict(tick=tick, t_s=round(tick / self.physics_hz, 6), selector=name,
                                             old=_label(old), new=_label(new)))
            self.switch = switch
            self.state_log.append(dict(tick=tick, switch=list(switch), vc_V=self.vc_v))
        ch1, ch2 = self.model.channels(self.switch, self.vc_v)
        self.logger.sample(tick, ch1, ch2, self.switch)
        return ch1, ch2

    def labels(self):
        """Selector positions now, by selector name ('transition' between detents)."""
        return {name: _label(value) for name, value in zip(SELECTORS, self.switch)}

    def process_extra(self):
        """Records for the offline process score beyond actions and state_log: the logger acquisition still
        recording at stop (None when idle). Never shown to the Agent."""
        return {'logger_open': self.logger.open_record()}


def _label(value):
    return 'transition' if value is None else value
