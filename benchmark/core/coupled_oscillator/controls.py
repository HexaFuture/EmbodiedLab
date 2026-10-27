"""Coupled bench controls without Isaac: the recorder front panel (keypad + display) and the catch hook rule.

The recorder is operated only through its keypad: digits enter the record length in seconds, CLR empties the entry,
REC takes the entry (within `record_range`, default 40 to 180 s; anything else shows ERR and does nothing). By default (a bench with a release, as
the coupled bench) a valid REC sets the recorder READY and the bench's release starts the record; with
`start_on_rec=True` (a bench without a release) REC's effect is 'start' and the owner starts the record at once.
READY cannot be undone; a record runs its full length. While READY or recording every key has no effect (the press
is logged). The display shows one of the states IDLE, READY, REC, DONE or ERR, the SET field (the entry being typed,
or the keyed record length), the T field (seconds recorded so far during REC, the length of the last record at DONE)
and the NO field (the number of the record being made or last finished). A key press after DONE or ERR returns the
panel to IDLE before it acts.

Hook rule: a slider and its catch hook passively when they meet: their offsets from zero within HOOK_DISTANCE_M of
each other and their relative speed at most HOOK_SPEED_M_S, outside a recording (the release holds the pawls open
from the release until the record ends). A hooked pair moves together until the release post unhooks both.
"""
from __future__ import annotations

RECORD_S_RANGE = (40, 180)
KEYS = tuple(str(d) for d in range(10)) + ('clear', 'rec')
HOOK_DISTANCE_M = 0.001
HOOK_SPEED_M_S = 0.02
ENTRY_DIGITS = 3


class RecorderPanel:
    def __init__(self, start_on_rec=False, record_range=RECORD_S_RANGE):
        self.start_on_rec = start_on_rec
        self.record_range = tuple(record_range)       # (shortest, longest) record length REC accepts, seconds
        self.entry = ''
        self.error = False
        self.ready_s = None          # record length keyed with REC, until the record starts
        self.recording_s = None      # record length of the record running
        self.state = 'IDLE'
        self.number = 0              # record being made or last finished (0: none yet)
        self.elapsed_s = 0.0
        self.last_length_s = None

    def press(self, key):
        """One key press (the closing edge of its switch). Returns its effect, for the log."""
        if key not in KEYS:
            raise ValueError(f'unknown recorder key {key!r}')
        if self.ready_s is not None or self.recording_s is not None:
            return 'ignored_ready' if self.ready_s is not None else 'ignored_recording'
        if self.state in ('DONE', 'ERR'):
            self.state = 'IDLE'
        self.error = False
        if key == 'clear':
            self.entry = ''
            return 'cleared'
        if key == 'rec':
            value = int(self.entry) if self.entry else None
            self.entry = ''
            if value is None or not self.record_range[0] <= value <= self.record_range[1]:
                self.error = True
                self.state = 'ERR'
                return 'error'
            self.ready_s = value
            self.state = 'READY'
            return 'start' if self.start_on_rec else 'ready'
        if len(self.entry) >= ENTRY_DIGITS:
            return 'ignored_full'
        self.entry += key
        return 'digit'

    def start(self, number):
        """The record keyed with REC starts (fired by the bench's release, or at once with start_on_rec)."""
        if self.ready_s is None:
            raise ValueError('recorder not ready')
        self.recording_s, self.ready_s = self.ready_s, None
        self.number, self.elapsed_s, self.state = number, 0.0, 'REC'

    def tick(self, elapsed_s):
        if self.recording_s is not None:
            self.elapsed_s = elapsed_s

    def finish(self):
        self.last_length_s, self.recording_s = self.recording_s, None
        self.state = 'DONE'

    @property
    def ready(self):
        return self.ready_s is not None

    @property
    def recording(self):
        return self.recording_s is not None

    def display(self):
        """What the panel shows: state word and the three numeric fields (None = blank)."""
        if self.state in ('READY', 'REC'):
            setting = self.ready_s if self.ready_s is not None else self.recording_s
        else:
            setting = int(self.entry) if self.entry else None
        if self.state == 'REC':
            seconds = int(self.elapsed_s)
        elif self.state == 'DONE':
            seconds = self.last_length_s
        else:
            seconds = None
        return {'state': self.state, 'set_s': setting, 'entry': self.entry, 't_s': seconds, 'number': self.number}

    def record(self):
        return {'state': self.state, 'entry': self.entry, 'ready_s': self.ready_s, 'recording_s': self.recording_s,
                'number': self.number, 'elapsed_s': round(self.elapsed_s, 3)}


def hooks(offset_slider_m, offset_catch_m, v_slider, v_catch):
    """Whether a free slider and its catch meet closely and slowly enough to hook."""
    return abs(offset_slider_m - offset_catch_m) <= HOOK_DISTANCE_M and abs(v_slider - v_catch) <= HOOK_SPEED_M_S
