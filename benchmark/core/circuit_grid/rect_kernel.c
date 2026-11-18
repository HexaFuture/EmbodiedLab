/* Fixed-clock integration of the circuit board's AC source + bridge module feeding a linear block network with up
 * to three capacitor states (circuit_grid/rectifier.py).
 *
 * The network is reduced (by the caller, at every change of the circuit) to the bridge port and the capacitor
 * states: over one step of length dt the capacitor voltages move as v' = Pm v + p i + q i', where i and i' are the
 * bridge currents at the start and the end of the step (trapezoidal rule; backward Euler: p = 0), and the port
 * voltage at the end of the step is u' = aPm . v + ap i + c1 i'. The bridge: an AC source amp sin(2 pi f t) with
 * series resistance rs and two equal Shockley diodes in series (I_s, a = 2 n k T / e) conducting from the source's
 * rectified voltage into the port: rs i + a ln(1 + i / I_s) = |source| - u, i >= 0; the source open when disabled.
 * No reverse leakage or recovery. No waveform lookup or steady-state shortcut.
 */
#include <math.h>
#include <stddef.h>

/* i >= 0 with r i + a log1p(i / is) = drive (safeguarded Newton, bracketed). */
double bridge_current(double drive, double r, double a, double is) {
    if (drive <= 0) return 0;
    double lo = 0, hi = drive / r;
    double x = fmax(1e-12, (drive - 1.1) / r);
    if (x >= hi) x = hi / 2;
    for (int j = 0; j < 60; j++) {
        double f = r * x + a * log1p(x / is) - drive;
        if (fabs(f) < 1e-12) return x;
        if (f > 0) hi = x; else lo = x;
        double y = x - f / (r + a / (x + is));
        x = (y > lo && y < hi) ? y : (lo + hi) / 2;
    }
    return x;
}

/* Advance `steps` steps from step index `start` (t = (start + k + 1) / hz at the end of step k). v (n) and *i are
 * updated in place. out, when not NULL, gets per step [v'_0 .. v'_{n-1}, i'] (n + 1 doubles). Returns 0. */
int rect_advance(int n, double *v, double *i_io, const double *Pm, const double *p, const double *q,
                 const double *aPm, double ap, double c1, double amp, double freq, double rs, double a, double is,
                 long long start, int steps, double hz, int enabled, double *out) {
    const double omega = 2 * 3.14159265358979323846 * freq;
    double w[3];
    double i = *i_io;
    for (int k = 0; k < steps; k++) {
        double t1 = (double)(start + k + 1) / hz;
        double src = amp * sin(omega * t1);
        double c0 = ap * i;
        for (int r = 0; r < n; r++) c0 += aPm[r] * v[r];
        double i1 = enabled ? bridge_current(fabs(src) - c0, rs + c1, a, is) : 0.0;
        for (int r = 0; r < n; r++) {
            double s = p[r] * i + q[r] * i1;
            for (int c = 0; c < n; c++) s += Pm[r * n + c] * v[c];
            w[r] = s;
        }
        for (int r = 0; r < n; r++) v[r] = w[r];
        i = i1;
        if (out) {
            double *row = out + (size_t)k * (n + 1);
            for (int r = 0; r < n; r++) row[r] = v[r];
            row[n] = i;
        }
    }
    *i_io = i;
    return 0;
}
