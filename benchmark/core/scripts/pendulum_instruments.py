"""Simulation-time optical gate counter, independent of Isaac Sim.

The caller must raycast the *real beam* every physics tick and pass only the
binary occlusion and the simulation timestamp.  This module never reads a
pendulum angle, mass, length, or gravity.  One interrupted beam pulse is a
half-cycle; every second leading edge is one full cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

# The bench's gate timers are fixed instruments: one warm-up cycle, then this many timed complete cycles.
TIMED_CYCLES = 10


@dataclass(frozen=True)
class GateConfig:
    target_cycles: int = 10
    warmup_cycles: int = 1
    min_half_period_s: float = 0.30
    max_half_period_s: float = 0.90
    half_period_tolerance: float = 0.25
    max_sample_gap_s: float = 0.020
    max_blocked_s: float = 0.25
    first_crossing_timeout_s: float = 6.0

    def __post_init__(self) -> None:
        for name in ("target_cycles", "warmup_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.target_cycles < 1 or self.warmup_cycles < 1:
            raise ValueError("At least one measurement and one warmup cycle are required")
        for name in (
            "min_half_period_s", "max_half_period_s", "max_sample_gap_s",
            "max_blocked_s", "first_crossing_timeout_s", "half_period_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_half_period_s >= self.max_half_period_s:
            raise ValueError("The half-period interval is empty")
        if self.max_sample_gap_s >= self.min_half_period_s:
            raise ValueError("Sample gap must be shorter than a half-period")
        if self.max_blocked_s >= self.min_half_period_s:
            raise ValueError("The beam must clear between crossings")
        if self.half_period_tolerance >= 0.5:
            raise ValueError("Half-period tolerance must be below 0.5 to reject skipped pulses")


@dataclass(frozen=True)
class GateReading:
    status: str
    cycles: int
    target_cycles: int
    elapsed_s: float
    elapsed_text: str
    cycles_text: str
    valid: bool
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class OpticalGateTimer:
    """Count full cycles using alternate rising edges of a binary beam.

    Call ``reset(sim_time_s)`` at physical release, then ``update`` on *every*
    physics tick, even while an Agent waits or the renderer is idle.  A panel
    redraw may lag; it must display this counter rather than retime events.

    Transition times are the midpoint of the two bracketing ticks.  This is
    not extra sensor precision: each timestamp has an uncertainty of half a
    tick.  ``audit`` reports that resolution and the accepted event times.

    ``waiting -> settling -> timing -> complete`` is normal.  ``invalid``
    latches until reset, retaining the reason.  The maximum/minimum half-period
    limits are public instrument limits, not a hidden gravity reference.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()
        self.reset(0.0)

    def reset(self, sim_time_s: float) -> GateReading:
        if not math.isfinite(sim_time_s):
            raise ValueError("Reset time must be finite")
        self._armed_at = float(sim_time_s)
        self._last_time: Optional[float] = None
        self._last_blocked: Optional[bool] = None
        self._blocked_since: Optional[float] = None
        self._has_seen_clear = False
        self._status = "waiting"
        self._error: Optional[str] = None
        self._edges: list[float] = []
        self._intervals: list[float] = []
        self._start: Optional[float] = None
        self._finish: Optional[float] = None
        self._cycles = 0
        self._max_dt = 0.0
        return self.reading()

    def abort(self, reason: str) -> GateReading:
        """End a running acquisition as invalid (e.g. its RESET button was pressed); a finished one is unchanged."""
        if self._status in ("complete", "invalid"):
            return self.reading()
        return self._invalidate(reason)

    def _invalidate(self, reason: str) -> GateReading:
        self._status = "invalid"
        self._error = reason
        return self.reading()

    def update(self, sim_time_s: float, blocked: bool) -> GateReading:
        if self._status in ("complete", "invalid"):
            return self.reading()
        if not math.isfinite(sim_time_s):
            return self._invalidate("non_finite_timestamp")
        if not isinstance(blocked, bool):
            return self._invalidate("occlusion_must_be_boolean")
        t = float(sim_time_s)
        if t < self._armed_at or (self._last_time is not None and t <= self._last_time):
            return self._invalidate("stale_timestamp")
        previous_t = self._last_time
        dt = t - (previous_t if previous_t is not None else self._armed_at)
        if dt > self.config.max_sample_gap_s + 1e-9:
            return self._invalidate("missing_physics_samples")
        self._max_dt = max(self._max_dt, dt)
        edge_t = t if previous_t is None else (previous_t + t) / 2
        rising = blocked and self._last_blocked is False and self._has_seen_clear
        if blocked:
            if self._blocked_since is None:
                self._blocked_since = edge_t
        else:
            self._has_seen_clear = True
            self._blocked_since = None
        self._last_time = t
        self._last_blocked = blocked

        if self._blocked_since is not None and t - self._blocked_since > self.config.max_blocked_s:
            return self._invalidate("beam_stuck_blocked")
        if rising:
            self._accept_edge(edge_t)
        if self._status in ("invalid", "complete"):
            return self.reading()
        if not self._edges and t - self._armed_at > self.config.first_crossing_timeout_s:
            return self._invalidate("no_crossing_after_release")
        if self._edges and t - self._edges[-1] > self.config.max_half_period_s + self._max_dt:
            return self._invalidate("crossing_timeout")
        return self.reading()

    def _accept_edge(self, t: float) -> None:
        if self._edges:
            interval = t - self._edges[-1]
            if not self.config.min_half_period_s <= interval <= self.config.max_half_period_s:
                self._invalidate("half_period_outside_instrument_range")
                return
            if self._intervals:
                reference = sum(self._intervals) / len(self._intervals)
                # Include finite sampling uncertainty; never infer a missed edge.
                tolerance = reference * self.config.half_period_tolerance + self._max_dt
                if abs(interval - reference) > tolerance:
                    self._invalidate("inconsistent_crossing_interval")
                    return
            self._intervals.append(interval)
        self._edges.append(t)
        index = len(self._edges) - 1
        start_index = 2 * self.config.warmup_cycles
        self._status = "settling"
        if index < start_index:
            return
        if self._start is None:
            self._start = t
        self._status = "timing"
        # Each leading edge is a half cycle, independent of panel redraw rate.
        self._cycles = (index - start_index) // 2
        if self._cycles >= self.config.target_cycles:
            self._finish = t
            self._status = "complete"

    def reading(self) -> GateReading:
        stop = self._finish if self._finish is not None else self._last_time
        elapsed = 0.0 if self._start is None or stop is None else max(0.0, stop - self._start)
        return GateReading(
            status=self._status,
            cycles=self._cycles,
            target_cycles=self.config.target_cycles,
            elapsed_s=elapsed,
            elapsed_text=f"{elapsed:06.3f}",
            cycles_text=f"{self._cycles:02d}",
            valid=self._status == "complete",
            error=self._error,
        )

    def audit(self) -> dict:
        """Event provenance for measurement logs; contains no physics answers."""
        return {
            "instrument": "binary_optical_gate",
            "clock": "simulation_time",
            "edge_time_method": "midpoint_of_bracketing_physics_ticks",
            "full_cycle_method": "every_second_rising_edge",
            "timestamp_uncertainty_s": self._max_dt / 2,
            "elapsed_uncertainty_bound_s": self._max_dt,
            "config": asdict(self.config),
            "armed_at_s": self._armed_at,
            "start_s": self._start,
            "finish_s": self._finish,
            "rising_edge_times_s": list(self._edges),
            "reading": self.reading().to_dict(),
        }
