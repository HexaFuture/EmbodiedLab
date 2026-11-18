"""Reference prediction from public trial shots only: interpolate the squared range between the bracketing notches.

For a spring launcher the exit kinetic energy grows close to the square of the compression, so range squared is
nearly linear in compression squared between neighbouring notches; the measured landings fix the local slope. No
hidden coefficient or held-out measurement can enter.
"""
import math


def predict_landing(calibration, target_notch_m):
    if not isinstance(target_notch_m, (int, float)) or not math.isfinite(target_notch_m):
        raise ValueError('target notch must be finite')
    grouped = {}
    for row in calibration:
        d = float(row['notch_m']); x = float(row['landing_distance_m'])
        if not math.isfinite(d) or not math.isfinite(x) or d <= 0 or x <= 0:
            raise ValueError('invalid trial measurement')
        if abs(d - target_notch_m) < .0005:
            raise ValueError('target-notch shots are not calibration')
        grouped.setdefault(round(d, 4), []).append(x * x)
    notches = sorted(grouped)
    lower = [d for d in notches if d < target_notch_m]
    upper = [d for d in notches if d > target_notch_m]
    if not lower or not upper:
        raise ValueError('target must be bracketed; extrapolation is not supported')
    lo, hi = lower[-1], upper[0]
    a = sum(grouped[lo]) / len(grouped[lo]); b = sum(grouped[hi]) / len(grouped[hi])
    if b <= a:
        raise ValueError('range does not increase across the bracket; repeat the shots')
    fraction = (target_notch_m ** 2 - lo ** 2) / (hi ** 2 - lo ** 2)
    return {'predicted_distance_m': math.sqrt(a + fraction * (b - a)),
            'model': 'range squared linear in compression squared between bracketing notches',
            'bracket_notches_m': [lo, hi], 'n_trials': len(calibration)}
