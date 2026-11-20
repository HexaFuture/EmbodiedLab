"""torsion_centroid specimen mass model (trusted construction only, no simulator dependencies).

Closed 160 x 60 x 20 mm shell plus two internal 20 x 30 x 12 mm ballast
blocks at x = +/-25 to +/-65 mm. The shell's effective inertia includes its lightweight
handle. All constituent centroids lie at mid-height. Both ballast blocks have
positive mass; the physical inertia tensor obeys the triangle inequalities.
"""
MASS_KG = 0.35
LENGTH_M, WIDTH_M, HEIGHT_M = .160, .060, .020
BALLAST_X_M = .050


def mass_properties(ballast_fraction, imbalance, ballast_halfspan_m=.050):
    q, b, a = float(ballast_fraction), float(imbalance), float(ballast_halfspan_m)
    if not (.2 <= q <= .8 and -.6 <= b <= .6 and .025 <= a <= .065):
        raise ValueError('invalid specimen construction')
    parts = [(MASS_KG * (1-q), 0., (LENGTH_M, WIDTH_M, HEIGHT_M)),
             (MASS_KG*q*(1-b)/2, -a, (.020, .030, .012)),
             (MASS_KG*q*(1+b)/2, a, (.020, .030, .012))]
    xc = sum(m*x for m, x, _ in parts)/MASS_KG
    ix = sum(m*(w*w+h*h)/12 for m, _, (_, w, h) in parts)
    iy = sum(m*((l*l+h*h)/12+(x-xc)**2) for m, x, (l, _, h) in parts)
    iz = sum(m*((l*l+w*w)/12+(x-xc)**2) for m, x, (l, w, _) in parts)
    return {'mass_kg': MASS_KG, 'com_m': [xc, 0., HEIGHT_M/2], 'diagonal_inertia_kgm2': [ix, iy, iz]}
