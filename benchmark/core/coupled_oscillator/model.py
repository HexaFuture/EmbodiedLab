"""Two-degree-of-freedom damped oscillator model of the coupled-oscillator bench.

The numerical integrator is intentionally independent of Isaac/PhysX.  It is
used for parameter sampling and analytic sanity checks;
the eventual Isaac asset must be validated against these trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class CoupledParameters:
    omega1: float
    omega2: float
    coupling_c: float
    damping_gamma: float
    identical: bool
    mass_kg: float = 0.20

    def __post_init__(self):
        for name in ("omega1", "omega2", "coupling_c", "damping_gamma", "mass_kg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.coupling_c < 0 or self.damping_gamma < 0:
            raise ValueError("coupling and damping must be non-negative")
        if self.identical and abs(self.omega1 - self.omega2) > 1e-10:
            raise ValueError("identical parameters require omega1 == omega2")

    @property
    def stiffness_matrix(self) -> np.ndarray:
        return np.array([[self.omega1**2 + self.coupling_c, -self.coupling_c],
                         [-self.coupling_c, self.omega2**2 + self.coupling_c]], dtype=float)

    def to_dict(self):
        return asdict(self)


def sample_parameters(seed: int) -> CoupledParameters:
    """Deterministically sample a visible-equivalent episode configuration."""
    rng = np.random.default_rng(seed)
    identical = bool(rng.integers(0, 2))
    omega1 = float(rng.uniform(3.0, 4.8))
    if identical:
        omega2 = omega1
    else:
        delta = float(rng.uniform(0.35, 0.9))
        omega2 = omega1 + (delta if bool(rng.integers(0, 2)) else -delta)
        omega2 = max(2.5, min(5.3, omega2))
    return CoupledParameters(omega1, omega2, float(rng.uniform(.25, 1.15)),
                             float(rng.uniform(.035, .095)), identical)


def normal_mode_frequencies(params: CoupledParameters) -> np.ndarray:
    """Return undamped normal angular frequencies in ascending order."""
    return np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(params.stiffness_matrix)))


def simulate(params: CoupledParameters, initial: tuple[float, float, float, float],
             duration_s: float = 24.0, dt: float = 1 / 240) -> np.ndarray:
    """Integrate [x1, v1, x2, v2] with velocity Verlet/RK-style substeps."""
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    n = int(round(duration_s / dt)) + 1
    out = np.empty((n, 5), dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    if state.shape != (4,):
        raise ValueError("initial must be (x1, v1, x2, v2)")
    k = params.stiffness_matrix

    def accel(s):
        x = np.array([s[0], s[2]])
        v = np.array([s[1], s[3]])
        a = -k @ x - 2 * params.damping_gamma * v
        return np.array([a[0], a[1]], dtype=float)

    t = 0.0
    for i in range(n):
        out[i] = (t, state[0], state[1], state[2], state[3])
        a0 = accel(state)
        # RK4 is used here for smooth reference traces; PhysX validation is
        # separate and must compare its own integration against this model.
        def f(s):
            a = accel(s)
            return np.array([s[1], a[0], s[3], a[1]])
        k1 = f(state)
        k2 = f(state + .5 * dt * k1)
        k3 = f(state + .5 * dt * k2)
        k4 = f(state + dt * k3)
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
    return out


def release_trace(params: CoupledParameters, kind: str, amplitude_m: float = .03,
                  duration_s: float = 24.0, dt: float = 1 / 240) -> np.ndarray:
    if kind == "in_phase":
        initial = (amplitude_m, 0.0, amplitude_m, 0.0)
    elif kind == "anti_phase":
        initial = (amplitude_m, 0.0, -amplitude_m, 0.0)
    else:
        raise ValueError("kind must be in_phase or anti_phase")
    return simulate(params, initial, duration_s, dt)


def estimate_dominant_frequencies(trace: np.ndarray, sample_hz: float = 240.0):
    """Return rough spectral peaks for diagnostics, not the benchmark solver."""
    result = []
    for col in (1, 3, 1):
        y = trace[:, col] - np.mean(trace[:, col])
        window = np.hanning(len(y))
        spectrum = np.abs(np.fft.rfft(y * window))
        frequencies = np.fft.rfftfreq(len(y), 1 / sample_hz)
        spectrum[0] = 0
        indices = np.argsort(spectrum)[-4:][::-1]
        result.append([float(2 * math.pi * frequencies[i]) for i in indices])
    return result


def save_trace(path: str | Path, trace: np.ndarray, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trace, delimiter=",", header="t_s,x1_m,v1_m_s,x2_m,v2_m_s", comments="")
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
