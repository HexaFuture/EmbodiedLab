"""Optics kit imaging model (thin lenses, paraxial ray transfer), headless.

Along the rail (x, metres, increasing away from the light box): the object (an illuminated "F" on the light box
at x = 0), thin lenses at x_i with focal lengths f_i and a clear aperture radius a_i, the screen at x_s. A ray
(height y, angle t) transfers by translation T(d) = [[1, d], [0, 1]] and lens L(f) = [[1, 0], [-1/f, 1]]; the system
matrix M = [[A, B], [C, D]] from the object plane to the screen gives, for an object point at height y and a ray
leaving it with angle t, the screen height A y + B t. Rays that pass the first lens at height h leave the object
at t = (h - y)/u1 (u1 = x_1): the chief ray (h = 0) lands at (A - B/u1) y, the rays across the first aperture
spread over a disc of radius |B| a_1 / u1 (the blur circle; zero where the screen is conjugate to the object).
Later apertures are assumed not to vignette. The screen image is the object pattern scaled by m = A - B/u1
(negative: inverted), convolved with that disc, at irradiance proportional to 1/m^2, on a fixed ambient level.
"""
from __future__ import annotations

import math

import numpy as np

# The object: a 5 x 7 cell "F" on a 20 mm x 28 mm window (cells 4 mm), bright on dark.
F_CELLS = ("11111", "10000", "10000", "11110", "10000", "10000", "10000")
F_CELL_M = 0.004
SCREEN_M = 0.120                 # square screen side
SCREEN_PX = 240                  # 0.5 mm per pixel
AMBIENT = 0.04
SUBSAMPLE = 3                    # samples per pixel side (pixel-area integration)


def system_matrix(lenses, x_screen):
    """lenses: [(x, f)] sorted by x, all between the object (x = 0) and the screen."""
    m = np.eye(2)
    x = 0.0
    for xl, f in lenses:
        m = np.array([[1.0, xl - x], [0.0, 1.0]]) @ m
        m = np.array([[1.0, 0.0], [-1.0 / f, 1.0]]) @ m
        x = xl
    return np.array([[1.0, x_screen - x], [0.0, 1.0]]) @ m


def magnification_and_blur(lenses, x_screen, aperture_m):
    """(m, blur radius on the screen in metres) for lenses [(x, f)] and the first lens's aperture radius."""
    if not lenses:
        return None, None
    lenses = sorted(lenses)
    (a, b), _ = system_matrix(lenses, x_screen)
    u1 = lenses[0][0]
    return a - b / u1, abs(b) * aperture_m / u1


def image_distance(u, f):
    """Thin-lens image distance for an object distance u (both positive for a real object and converging lens)."""
    return 1.0 / (1.0 / f - 1.0 / u)


def object_mask(px_m):
    """Object pattern sampled at px_m per pixel, centred: (array, half-size in metres)."""
    rows, cols = len(F_CELLS), len(F_CELLS[0])
    n_y, n_x = round(rows * F_CELL_M / px_m), round(cols * F_CELL_M / px_m)
    img = np.zeros((n_y, n_x))
    for r, line in enumerate(F_CELLS):
        for c, ch in enumerate(line):
            if ch == "1":
                img[round(r * F_CELL_M / px_m):round((r + 1) * F_CELL_M / px_m),
                    round(c * F_CELL_M / px_m):round((c + 1) * F_CELL_M / px_m)] = 1.0
    return img


def screen_image(lenses, x_screen, aperture_m, *, px=SCREEN_PX, side_m=SCREEN_M):
    """Grey image (0..1) of the screen, row 0 at the top (+y up), column 0 at the left as seen facing the screen
    from the light box side."""
    import cv2
    px_m = side_m / px
    out = np.full((px, px), AMBIENT)
    m, blur = magnification_and_blur(lenses, x_screen, aperture_m)
    if m is None or abs(m) < 1e-3:
        return out
    radius_px = blur / px_m
    if radius_px > 16 and px % 4 == 0 and px > SCREEN_PX:
        # a wide blur: the light is smooth at a quarter of the resolution
        small = screen_image(lenses, x_screen, aperture_m, px=px // 4, side_m=side_m)
        return cv2.resize(small.astype(np.float32), (px, px), interpolation=cv2.INTER_LINEAR).astype(float)
    # The sharp image: each lit cell of the F maps to an axis-aligned rectangle on the screen (scaled by m; m < 0
    # inverts both axes), and each screen pixel takes the fraction of its area that rectangle covers, as a camera
    # pixel does (exact: row cover x column cover); edges carry sub-pixel position.
    rows, cols = len(F_CELLS), len(F_CELLS[0])
    ys = (np.arange(px)[::-1] - (px - 1) / 2) * px_m                  # pixel centre heights, top row = +
    xs = (np.arange(px) - (px - 1) / 2) * px_m

    def cover(centres, a, b):
        lo, hi = min(a, b), max(a, b)
        return np.clip(np.minimum(hi, centres + px_m / 2) - np.maximum(lo, centres - px_m / 2), 0.0, None) / px_m

    row_cover, col_cover = [], []
    for r, line in enumerate(F_CELLS):
        for c, ch in enumerate(line):
            if ch == "1":
                row_cover.append(cover(ys, m * (rows * F_CELL_M / 2 - (r + 1) * F_CELL_M),
                                       m * (rows * F_CELL_M / 2 - r * F_CELL_M)))
                col_cover.append(cover(xs, m * (c * F_CELL_M - cols * F_CELL_M / 2),
                                       m * ((c + 1) * F_CELL_M - cols * F_CELL_M / 2)))
    sharp = np.asarray(row_cover).T @ np.asarray(col_cover)
    level = min(1.0, 0.9 / max(m * m, 1e-6))                   # irradiance ~ 1/m^2, clipped at full scale
    if radius_px > 0.05:
        # Antialiased disc (coverage of each kernel pixel, 4 x 4 subsamples), so blur grows continuously from 0.
        k = int(math.ceil(radius_px)) + 1
        sub = (np.arange(4) + 0.5) / 4 - 0.5
        gy, gx = np.mgrid[-k:k + 1, -k:k + 1]
        sy, sx = np.meshgrid(sub, sub, indexing="ij")
        disc = ((gx[..., None, None] + sx) ** 2 + (gy[..., None, None] + sy) ** 2 <= radius_px * radius_px).mean(axis=(2, 3))
        if disc.sum() <= 0:                    # smaller than the subsample grid: no visible blur
            return np.clip(AMBIENT + level * sharp, 0.0, 1.0)
        disc /= disc.sum()
        sharp = cv2.filter2D(sharp.astype(np.float32), -1, disc.astype(np.float32),
                             borderType=cv2.BORDER_CONSTANT).astype(float)
    return np.clip(AMBIENT + level * sharp, 0.0, 1.0)

def sharpness(image):
    """Focus measure used in tests and diagnostics: mean squared gradient over the image variance (independent of
    the image's brightness, which changes with magnification)."""
    gy, gx = np.gradient(image)
    var = float(np.var(image))
    return float(np.mean(gx * gx + gy * gy)) / var if var > 0 else 0.0


# --- grating: laser spots on the screen ---------------------------------------------------------------------
SPOT_SIGMA_M = 0.0006
ORDER_LEVELS = {0: 1.0, 1: 0.55, 2: 0.25, 3: 0.12}


def grating_spots(wavelength_m, lines_per_mm, distance_m, max_order=3):
    """Horizontal screen positions (m, centred on the axis) of the diffraction orders that exist and the order
    numbers: x_m = L tan(theta_m), sin theta_m = m lambda / d, the lines vertical so the orders spread sideways."""
    d = 1e-3 / lines_per_mm
    out = []
    for m in range(-max_order, max_order + 1):
        s = m * wavelength_m / d
        if abs(s) < 1:
            out.append((m, distance_m * s / math.sqrt(1 - s * s)))
    return out


def grating_image(wavelength_m, lines_per_mm, distance_m, *, px=SCREEN_PX, side_m=SCREEN_M, max_order=3):
    """Grey screen image of the spots (Gaussian, sigma SPOT_SIGMA_M) on the ambient level; orders beyond the screen
    edge fall outside the image."""
    px_m = side_m / px
    out = np.full((px, px), AMBIENT)
    if distance_m <= 0:
        return out
    xs = (np.arange(px) - (px - 1) / 2) * px_m
    ys = (np.arange(px)[::-1] - (px - 1) / 2) * px_m
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    for m, x in grating_spots(wavelength_m, lines_per_mm, distance_m, max_order):
        out += 0.9 * ORDER_LEVELS.get(abs(m), 0.05) * np.exp(-((xx - x) ** 2 + yy ** 2) / (2 * SPOT_SIGMA_M ** 2))
    return np.clip(out, 0.0, 1.0)


# --- single slit: Fraunhofer pattern on the screen ----------------------------------------------------------
SLIT_PEAK = 8.0            # central maximum at 1 m (screen light, 1 = fully lit): the centre over-exposes the card at the
SLIT_BEAM_SIGMA_M = 0.0010  # bench's distances, so the first three or four side lobes show (a bright laser). Vertical beam
                            # profile (the slit is vertical, the pattern sideways).


def slit_dark_fringes(wavelength_m, width_m, distance_m):
    """Horizontal screen positions (m, centred on the axis) of the dark fringes and their orders k:
    sin theta_k = k lambda / a, x_k = L tan theta_k."""
    out, k = [], 1
    while k * wavelength_m / width_m < 1:
        s = k * wavelength_m / width_m
        x = distance_m * s / math.sqrt(1 - s * s)
        out += [(-k, -x), (k, x)]
        k += 1
    return sorted(out, key=lambda e: e[1])


def slit_profile(wavelength_m, width_m, distance_m, x_m):
    """Relative irradiance sinc^2(pi a sin(theta) / lambda) at screen positions x_m, scaled 1/L (the beam spreads
    in one direction only)."""
    sin_t = x_m / np.hypot(x_m, distance_m)
    return SLIT_PEAK / distance_m * np.sinc(width_m * sin_t / wavelength_m) ** 2


def slit_image(wavelength_m, width_m, distance_m, *, px=SCREEN_PX, side_m=SCREEN_M):
    """Grey screen image of a vertical slit's diffraction pattern on the ambient level (pixel-area integration across
    the fringes, then the camera's saturation)."""
    px_m = side_m / px
    out = np.full((px, px), AMBIENT)
    if distance_m <= 0:
        return out
    sub = 5
    xs = (np.arange(px * sub) - (px * sub - 1) / 2) * px_m / sub
    row = slit_profile(wavelength_m, width_m, distance_m, xs).reshape(px, sub).mean(axis=1)
    ys = (np.arange(px)[::-1] - (px - 1) / 2) * px_m
    vertical = np.exp(-ys ** 2 / (2 * SLIT_BEAM_SIGMA_M ** 2))
    return np.clip(out + vertical[:, None] * row[None, :], 0.0, 1.0)


# --- glass plates: a plate of thickness t and index n shortens the optical path after it by t (1 - 1/n) ----------
def plate_shift_m(thickness_m, n):
    return thickness_m * (1.0 - 1.0 / n)


def fold_plates(elements, screen_m):
    """elements: [(x_m, spec)] sorted along the rail, spec {'kind': 'lens', 'f_m'} or {'kind': 'plate', 'thickness_m',
    'n'}. Returns the lenses [(x_m, f_m)] and the screen position (m) of the equivalent system without plates (paraxial;
    a plate lies wholly between its neighbours, as the carriages' spacing ensures)."""
    lenses, shift = [], 0.0
    for x, spec in elements:
        if spec['kind'] == 'plate':
            shift += plate_shift_m(spec['thickness_m'], spec['n'])
        elif spec['kind'] == 'lens':
            lenses.append((x - shift, spec['f_m']))
    return lenses, screen_m - shift


def focus_position_m(elements):
    """Screen position (m) of the sharp image of the object (x = 0) through one lens and any plates."""
    lenses, _ = fold_plates(elements, 1e9)
    (x, f), = lenses
    total = sum(plate_shift_m(s['thickness_m'], s['n']) for _, s in elements if s['kind'] == 'plate')
    return x + image_distance(x, f) + total
