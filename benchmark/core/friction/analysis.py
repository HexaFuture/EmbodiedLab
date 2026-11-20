"""Optional calculations from public incline instrument records.

These functions are analysis helpers, not Agent answer gates. They accept no
material coefficients, simulator state, or private episode metadata.
"""
import math

import numpy as np


def static_interval(last_rest_angle_deg, first_slip_angle_deg):
    """Convert an observed critical-angle bracket into a coefficient bracket."""
    lo, hi = float(last_rest_angle_deg), float(first_slip_angle_deg)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi < 80):
        raise ValueError("need an ordered finite angle bracket in [0, 80)")
    a, b = math.tan(math.radians(lo)), math.tan(math.radians(hi))
    return {"mu_s": (a + b) / 2, "mu_s_interval": [a, b],
            "critical_angle_interval_deg": [lo, hi]}


def kinetic_fit(samples, gravity_m_s2=9.81, max_angle_span_deg=0.05):
    """Fit s(t) over a caller-selected free sliding segment at a fixed angle.

    Samples contain time_s, position_m, angle_deg. The caller must select the
    interval after release and before catching/side contact. A free intercept
    and initial velocity absorb timing origin and small release velocity.
    """
    if len(samples) < 8:
        raise ValueError("need at least eight position samples")
    g = float(gravity_m_s2)
    if not math.isfinite(g) or g <= 0:
        raise ValueError("gravity must be finite and positive")
    values = np.asarray([[r["time_s"], r["position_m"], r["angle_deg"]]
                         for r in samples], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("instrument readings must be finite")
    ts, positions, angles = values.T
    if np.any(np.diff(ts) <= 0) or np.ptp(ts) < 0.02:
        raise ValueError("time must increase over a measurable interval")
    if np.any(angles < 0) or np.any(angles >= 80):
        raise ValueError("incline angle outside supported range")
    if np.ptp(angles) > max_angle_span_deg:
        raise ValueError("fixed-angle model requires a stationary incline")
    # Center times to avoid ill-conditioning for long-running instrument clocks.
    origin = float(np.mean(ts))
    dt = ts - origin
    design = np.column_stack([np.ones(len(ts)), dt, 0.5 * dt ** 2])
    coeff, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    if rank != 3:
        raise ValueError("trajectory does not identify acceleration")
    position, velocity, acceleration = map(float, coeff)
    if np.min(velocity + acceleration * dt) <= 0:
        raise ValueError("select a segment sliding in the downhill direction")
    theta = math.radians(float(np.mean(angles)))
    mu = (g * math.sin(theta) - acceleration) / (g * math.cos(theta))
    residual = positions - design @ coeff
    return {"mu_k": mu, "acceleration_m_s2": acceleration,
            "angle_deg": math.degrees(theta), "time_origin_s": origin,
            "position_at_origin_m": position, "velocity_at_origin_m_s": velocity,
            "samples": len(ts), "fit_rms_m": float(np.sqrt(np.mean(residual ** 2)))}


def static_onset(samples, speed_m_s=0.05, window_s=0.1, sustain_s=0.1):
    """Locate sustained downhill motion from quantized public instruments.

    This operational onset uses a fixed 0.1 s mean speed threshold, followed by
    0.1 s persistence. It suppresses sub-mm creep and isolated impulses. The
    reported bracket is detection resolution, not a bound on systematic error.
    Defaults are fixed before the new robot batch, not fitted to hidden truth.
    """
    rows=[r for r in samples if r['position_m'] is not None]
    if len(rows)<8:raise ValueError('insufficient supported position samples')
    t=np.asarray([r['time_s'] for r in rows],float)
    s=np.asarray([r['position_m'] for r in rows],float)
    angles=np.asarray([r['angle_deg'] for r in rows],float)
    if not np.isfinite(np.r_[t,s,angles]).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite ordered instruments required')
    if speed_m_s<=0 or window_s<=0 or sustain_s<=0:raise ValueError('positive detection settings required')
    velocities=np.full(len(rows),np.nan)
    for i in range(len(rows)):
        j=int(np.searchsorted(t,t[i]-window_s+1e-9,side='right'))-1
        if j>=0 and t[i]-t[j]>=window_s-1e-8 and np.max(np.diff(t[j:i+1]))<=.02:
            velocities[i]=(s[i]-s[j])/(t[i]-t[j])
    for i in range(1,len(rows)):
        stop=int(np.searchsorted(t,t[i]+sustain_s,side='left'))
        if stop>=len(rows):break
        if (np.isfinite(velocities[i-1]) and velocities[i-1]<=speed_m_s
            and np.all(velocities[i:stop+1]>speed_m_s)
            and angles[i]>1 and angles[i]>=angles[i-1]
            and np.max(np.diff(t[i-1:stop+1]))<=.02):
            result=static_interval(angles[i-1],angles[i])
            result.update(onset_tick=rows[i]['physics_tick'],onset_time_s=float(t[i]),
                mean_speed_m_s=float(velocities[i]),speed_threshold_m_s=speed_m_s,
                velocity_window_s=window_s,persistence_s=sustain_s,
                interpretation='operational sustained-slip onset; bracket excludes systematic bias')
            return result
    raise ValueError('no sustained downhill onset detected')
