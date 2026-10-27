"""Air-track photogates and the offline reduction, independent of Isaac.

Two double-beam photogates (gate 1: beams at track X 0.22 and 0.24 m; gate 2: 0.66 and 0.68 m; beam 1 is the beam at
lower X). Each glider carries a 30 mm opaque flag centred on it, longer than the 20 mm beam spacing. A gate timer
knows only its two beams: a pass starts when one beam is blocked while the other is clear and is complete when the
other beam is blocked; its time is the interval between the two blocking edges, its direction + when beam 1 blocked
first (motion towards +X) and - otherwise. A flag that withdraws before reaching the second beam makes no pass. After a
pass both beams must clear before the next one can start. Edge times are interpolated between physics ticks (positions
are linear in time within a tick) and counted by a 10 us clock from the start of the run.

The instrument record (public) never says which glider made a pass. The bench keeps that privately with each pass, and
the evaluator reduces a run with it (reduce_record).
"""
from __future__ import annotations

import math

GATES = {1: (0.22, 0.24), 2: (0.66, 0.68)}
BEAM_SPACING_M = 0.020
FLAG_LENGTH_M = 0.030
GLIDER_LENGTH_M = 0.100                # housing length along X: centres this far apart when the gliders touch
TIME_QUANTUM_S = 1e-5                  # 10 us timer clock: 0.01 ms resolution of the pass time
ROW_TOLERANCE_M_S = 0.0005             # a submitted velocity must agree with the reduction this closely
B_REST_M_S = 0.001                     # B counts as at rest before the collision at or below this speed
CONTACT_GAP_M = 0.001                  # gliders within this of touching count as in contact (private witness)


def _boundary(x0, x1, plane):
    """Fraction of the tick at which a flag centred at x0 -> x1 reaches the block/clear boundary of `plane`."""
    for b in (plane - FLAG_LENGTH_M / 2, plane + FLAG_LENGTH_M / 2):
        if min(x0, x1) <= b <= max(x0, x1) and x1 != x0:
            return (b - x0) / (x1 - x0)
    return 1.0


def blocks(x, plane):
    return abs(x - plane) < FLAG_LENGTH_M / 2


class GateTimer:
    """Both gate timers over one run. positions: {glider: track X of its centre} (the glider label stays private)."""

    def __init__(self, hz, start_tick, positions):
        self.hz, self.start_tick, self.tick = hz, start_tick, start_tick
        self.prev = dict(positions)
        self.beam = {(g, b): any(blocks(x, p) for x in positions.values())
                     for g, planes in GATES.items() for b, p in enumerate(planes, 1)}
        # A gate whose beam is blocked at the start waits for both beams to clear (no pass from a flag already in it).
        self.state = {g: ('wait_clear' if self.beam[(g, 1)] or self.beam[(g, 2)] else 'idle') for g in GATES}
        self.pending = {g: None for g in GATES}
        self.count = {g: 0 for g in GATES}
        self.passes = []                # complete passes, in completion order (with private glider and ticks)

    def step(self, tick, positions):
        """Advance one physics tick: positions at `tick` (consecutive ticks only). Returns the passes completed."""
        if tick != self.tick + 1:
            raise ValueError('gate timers need consecutive physics ticks')
        events = []
        for glider, x1 in positions.items():
            x0 = self.prev[glider]
            for g, planes in GATES.items():
                for b, plane in enumerate(planes, 1):
                    before, after = blocks(x0, plane), blocks(x1, plane)
                    if before == after:
                        continue
                    t = (self.tick + _boundary(x0, x1, plane) - self.start_tick) / self.hz
                    count = int(round(t / TIME_QUANTUM_S))
                    events.append((count, 0 if after else 1, g, b, after, glider, tick))
        done = []
        for count, _, g, b, blocked, glider, at in sorted(events):
            self.beam[(g, b)] = blocked
            other = self.beam[(g, 3 - b)]
            state = self.state[g]
            if state == 'idle' and blocked and not other:
                self.pending[g] = (b, count, glider, at)
                self.state[g] = 'pending'
            elif state == 'pending':
                b0, c0, glider0, at0 = self.pending[g]
                if blocked and b != b0:
                    self.count[g] += 1
                    p = {'gate': g, 'index': self.count[g], 'direction': '+' if b0 == 1 else '-',
                         'first_count': c0, 'second_count': count, 'glider': glider0,
                         'tick_first': at0, 'tick_second': at}
                    self.passes.append(p)
                    done.append(p)
                    self.state[g], self.pending[g] = 'wait_clear', None
                elif not blocked and b == b0 and not other:
                    self.state[g], self.pending[g] = 'idle', None      # the flag withdrew: no pass
            elif state == 'wait_clear' and not self.beam[(g, 1)] and not self.beam[(g, 2)]:
                self.state[g] = 'idle'
        self.prev = dict(positions)
        self.tick = tick
        return done


def public_pass(p):
    """The instrument's view of a pass: gate, index on that gate's screen, direction, beam times and pass time."""
    return {'gate': p['gate'], 'index': p['index'], 'direction': p['direction'],
            't_first_beam_s': round(p['first_count'] * TIME_QUANTUM_S, 5),
            't_second_beam_s': round(p['second_count'] * TIME_QUANTUM_S, 5),
            'dt_ms': round((p['second_count'] - p['first_count']) * TIME_QUANTUM_S * 1000, 2)}


def pass_velocity(public):
    """Signed glider speed (m/s) from a public pass: beam spacing over the pass time, signed by the direction."""
    dt = float(public['dt_ms']) / 1000
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('nonpositive pass time')
    return (1 if public['direction'] == '+' else -1) * BEAM_SPACING_M / dt


def _overlaps(intervals, lo, hi):
    return any(a <= hi and b >= lo for a, b in intervals if lo <= hi)


def reduce_record(record, mass_A_kg=0.2):
    """The evaluator's reduction of one sealed run (public passes + private witnesses).

    A before: A's pass into the collision (its only pass before the gliders first touch; gate 1, +).
    B after: B's first pass, after the collision (gate 2, +).
    A after: A's first complete pass through either gate after the gliders separate (signed).
    Valid only when B was at rest from the release until the collision, the gliders touched only once until both
    after-passes were complete, and no external force (robot, launcher, bumpers or a catch) acted on A from the start
    of its incoming pass to the collision and from the separation to the end of its after-pass, nor on B from the
    release to the collision and from the separation to the end of its pass. Raises ValueError(reason) otherwise."""
    public = record['public']
    by_key = {(p['gate'], p['index']): p for p in public['passes']}
    labels = sorted(record['pass_gliders'], key=lambda p: (p['tick_first'], p['gate']))
    for p in labels:
        if (p['gate'], p['index']) not in by_key:
            raise ValueError('private pass witness without a public pass')
    contacts = record['contact_ticks']
    if not contacts:
        raise ValueError('the gliders never collided during the run')
    c_start, c_end = contacts[0]
    a = [p for p in labels if p['glider'] == 'A']
    b = [p for p in labels if p['glider'] == 'B']
    a_before = [p for p in a if p['tick_second'] < c_start]
    if len(a_before) != 1:
        raise ValueError('no single incoming pass of A before the collision')
    a_in = a_before[0]
    if a_in['gate'] != 1 or by_key[(1, a_in['index'])]['direction'] != '+':
        raise ValueError('A did not enter the collision through gate 1 in the + direction')
    if any(p['tick_first'] <= c_end for p in b):
        raise ValueError('B passed a gate before the collision')
    if not b:
        raise ValueError('no pass of B after the collision')
    b_out = b[0]
    if b_out['gate'] != 2 or by_key[(2, b_out['index'])]['direction'] != '+':
        raise ValueError('B did not leave through gate 2 in the + direction')
    a_after = [p for p in a if p['tick_first'] > c_end]
    if not a_after:
        raise ValueError('no complete pass of A after the collision')
    a_out = a_after[0]
    end = max(b_out['tick_second'], a_out['tick_second'])
    if any(s <= end for s, _ in contacts[1:]):
        raise ValueError('the gliders touched again before both after-passes were complete')
    if record['B_speed_before_contact_m_s'] > B_REST_M_S:
        raise ValueError('B was not at rest before the collision')
    windows = {'A': [(a_in['tick_first'], c_start - 1), (c_end + 1, a_out['tick_second'])],
               'B': [(record['start_tick'], c_start - 1), (c_end + 1, b_out['tick_second'])]}
    for glider, spans in windows.items():
        for kind in ('external_ticks', 'catch_ticks'):
            if any(_overlaps(record[kind][glider], lo, hi) for lo, hi in spans):
                what = 'external contact' if kind == 'external_ticks' else 'a catch'
                raise ValueError(f'{what} on glider {glider} while it was measured')
    return {'v_A_before_m_s': pass_velocity(by_key[(1, a_in['index'])]),
            'v_A_after_m_s': pass_velocity(by_key[(a_out['gate'], a_out['index'])]),
            'v_B_after_m_s': pass_velocity(by_key[(2, b_out['index'])]),
            'notch': record.get('notch'), 'complete_tick': end,
            'passes': {'A_before': [1, a_in['index']], 'A_after': [a_out['gate'], a_out['index']],
                       'B_after': [2, b_out['index']]}}


def joint_mass(measurements, mass_A_kg=0.2, min_rows=3, min_speed_gap=0.025):
    """Least squares over momentum changes, retaining the signs of A's speed. min_rows / min_speed_gap (None: no
    check): the evaluator's gate and refit use 1 / None since the six levels."""
    if len(measurements) < max(1, min_rows):
        raise ValueError("three distinct valid collisions required" if min_rows == 3
                         else f"at least {max(1, min_rows)} valid collision(s) required")
    velocities = [m["v_A_before_m_s"] for m in measurements]
    if min_speed_gap is not None and len(velocities) > 1 and min(
            abs(a - b) for i, a in enumerate(velocities) for b in velocities[i + 1:]) < min_speed_gap:
        raise ValueError("incoming speeds are insufficiently distinct")
    ys = [mass_A_kg * (m["v_A_before_m_s"] - m["v_A_after_m_s"]) for m in measurements]
    xs = [m["v_B_after_m_s"] for m in measurements]
    if not all(math.isfinite(v) for v in xs + ys) or any(v <= 0 for v in xs):
        raise ValueError("invalid momentum observations")
    estimate = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    return {"mass_B_kg": estimate, "individual_masses_kg": [y / x for x, y in zip(xs, ys)],
            "momentum_rms_kg_m_s": math.sqrt(sum((y - estimate * x) ** 2 for x, y in zip(xs, ys)) / len(xs))}
