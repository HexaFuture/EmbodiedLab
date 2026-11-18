"""Graduated screen of the hand-slid optics bench (optics kit), headless: what is printed on the screen and what the
light from the source draws on it.

The screen is a 120 mm white card. Printed on it (dark ink):
- 'grid' (lens_focal): a faint millimetre grid, darker every 5 mm and every 10 mm;
- 'scale' (grating_wavelength): a horizontal millimetre scale with 0 at the centre, its baseline SCALE_BELOW_MM
  below the centre line, ticks hanging down from it (1 mm, longer every 5 mm and 10 mm) and the numbers every 10 mm
  below them (-50 ... 50, negative on the viewer's left as seen from the source).
The light image (optics_model, 0..1 at the texture resolution for the F, or computed here for laser spots) falls on the card:
the renderer gets one texture of TEXTURE_PX pixels, the card as lit (lit_card: the printed reflectance times the room
light plus the projected light; the ink absorbs both). Image column 0 is the viewer's left as seen from the source, row 0 the top, as in
optics_model. Screen coordinate x (mm) runs along the image rows from the centre, positive to the viewer's right.
"""
from __future__ import annotations

import math

import numpy as np

import optics_model as M

SCREEN_MM = M.SCREEN_M * 1000.0
TEXTURE_PX = 960                    # 8 px per mm
PAPER = 0.86                        # card reflectance
INK = 0.10
ROOM = 0.5                          # the card's brightness under the room light alone, as a fraction of fully lit
GRID_DARKNESS = {1: 0.22, 5: 0.40, 10: 0.62}       # fraction of the light the grid lines absorb (faint)
GRID_WIDTH_MM = {1: 0.12, 5: 0.18, 10: 0.25}
SCALE_BELOW_MM = 3.0                # scale baseline below the centre line (the spots run along the centre line)
SCALE_TICK_MM = {1: 2.0, 5: 3.5, 10: 5.0}
SCALE_TICK_WIDTH_MM = 0.20
SCALE_NUMBER_MM = 3.2               # digit height
SCALE_NUMBER_GAP_MM = 1.5           # below the longest ticks
SCALE_RANGE_MM = 55                 # ticks from -55 to 55 mm, numbers every 10 mm to +-50


def px_mm(px=TEXTURE_PX):
    return SCREEN_MM / px


def coordinate_mm(px=TEXTURE_PX):
    """Screen coordinate (mm, positive to the viewer's right) of each image column (and, downward, row)."""
    return (np.arange(px) - (px - 1) / 2) * px_mm(px)


def _lines(coord, spacing, width):
    """Fraction (0..1) of each pixel that a line of `width` mm at every multiple of `spacing` mm covers."""
    step = px_mm(len(coord))
    d = np.abs((coord + spacing / 2) % spacing - spacing / 2)          # distance to the nearest line centre
    return np.clip((width / 2 + step / 2 - d) / step, 0.0, 1.0)


def grid_ink(px=TEXTURE_PX):
    """Absorbed fraction (0..1) of the printed millimetre grid."""
    c = coordinate_mm(px)
    out = np.zeros((px, px))
    for spacing in (1, 5, 10):
        line = _lines(c, spacing, GRID_WIDTH_MM[spacing]) * GRID_DARKNESS[spacing]
        out = np.maximum(out, np.maximum(line[None, :], line[:, None]))
    return out


def scale_ink(px=TEXTURE_PX):
    """Absorbed fraction (0..1) of the printed horizontal scale: baseline, ticks and numbers."""
    import cv2
    step = px_mm(px)
    c = coordinate_mm(px)
    rows = -c                                     # row coordinate: + up (row 0 is the top)
    out = np.zeros((px, px))
    base = -SCALE_BELOW_MM
    # baseline (0.2 mm) from -SCALE_RANGE to +SCALE_RANGE
    span = np.abs(c) <= SCALE_RANGE_MM + 0.1
    line = np.clip((0.1 + step / 2 - np.abs(rows - base)) / step, 0, 1)
    out = np.maximum(out, line[:, None] * span[None, :])
    for mm in range(-SCALE_RANGE_MM, SCALE_RANGE_MM + 1):
        kind = 10 if mm % 10 == 0 else 5 if mm % 5 == 0 else 1
        length = SCALE_TICK_MM[kind]
        width = SCALE_TICK_WIDTH_MM * (1.4 if kind == 10 else 1.0)
        col = np.clip((width / 2 + step / 2 - np.abs(c - mm)) / step, 0, 1)
        row = ((rows <= base + 1e-9) & (rows >= base - length)).astype(float)
        out = np.maximum(out, row[:, None] * col[None, :])
    # numbers every 10 mm (cv2 Hershey simplex at the digit height), centred under their ticks
    canvas = np.zeros((px, px), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for mm in range(-50, 51, 10):
        text = str(mm)
        (w, h), _ = cv2.getTextSize(text, font, 1.0, 2)
        scale = SCALE_NUMBER_MM / step / h
        thickness = max(1, round(0.35 / step))
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = int(round((mm + SCREEN_MM / 2) / step - w / 2))
        if text.startswith('-'):           # centre the digits, the sign hangs to the left
            (wd, _), _ = cv2.getTextSize(text[1:], font, scale, thickness)
            x = int(round((mm + SCREEN_MM / 2) / step - wd / 2 - (w - wd)))
        y = int(round((SCREEN_MM / 2 + SCALE_BELOW_MM + SCALE_TICK_MM[10] + SCALE_NUMBER_GAP_MM) / step + h))
        cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    return np.maximum(out, canvas / 255.0)


def ink(kind, px=TEXTURE_PX):
    if kind == 'grid':
        return grid_ink(px)
    if kind == 'scale':
        return scale_ink(px)
    return np.zeros((px, px))


def card(ink_fraction):
    """The printed card as an 8-bit diffuse texture."""
    return np.clip(np.round(255 * (PAPER - (PAPER - INK) * ink_fraction)), 0, 255).astype(np.uint8)


def lit_card(light, ink_fraction):
    """The card as the cameras see it: reflectance x (room light + projected light (0..1, any resolution)), 8-bit."""
    import cv2
    px = ink_fraction.shape[0]
    light = np.asarray(light, np.float32)
    if light.shape[0] != px:
        light = cv2.resize(light, (px, px), interpolation=cv2.INTER_LINEAR)
    reflectance = PAPER - (PAPER - INK) * ink_fraction
    return np.clip(np.round(255 * reflectance * (ROOM + (1.0 - ROOM) * np.clip(light, 0.0, 1.0))), 0, 255).astype(np.uint8)


def emission(light, ink_fraction):
    """The light on the card (0..1, any resolution; resampled to the texture) as an 8-bit emission texture."""
    import cv2
    px = ink_fraction.shape[0]
    if light.shape[0] != px:
        light = cv2.resize(light.astype(np.float32), (px, px), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.round(255 * light * (1.0 - ink_fraction)), 0, 255).astype(np.uint8)


# --- laser spots at the texture resolution -------------------------------------------------------------------
def spots_image(wavelength_m, lines_per_mm, distance_m, px=TEXTURE_PX, max_order=3):
    """As optics_model.grating_image (Gaussian spots of sigma SPOT_SIGMA_M on the ambient level, orders up to 3 that
    fall on the screen), at px pixels, computed separably."""
    out = np.full((px, px), M.AMBIENT)
    if distance_m <= 0:
        return out
    xs = coordinate_mm(px) / 1000.0
    ys = xs[::-1]
    vertical = np.exp(-ys ** 2 / (2 * M.SPOT_SIGMA_M ** 2))
    for m, x in M.grating_spots(wavelength_m, lines_per_mm, distance_m, max_order):
        if abs(x) > M.SCREEN_M / 2 + 5 * M.SPOT_SIGMA_M:
            continue
        level = 0.9 * M.ORDER_LEVELS.get(abs(m), 0.05)
        out += level * np.outer(vertical, np.exp(-(xs - x) ** 2 / (2 * M.SPOT_SIGMA_M ** 2)))
    return np.clip(out, 0.0, 1.0)


def dark(px=TEXTURE_PX):
    return np.zeros((px, px))


def spot_positions_mm(wavelength_m, lines_per_mm, distance_m, max_order=1):
    """{order: screen coordinate mm} of the orders up to max_order (the +1 order at positive x)."""
    return {m: 1000.0 * x for m, x in M.grating_spots(wavelength_m, lines_per_mm, distance_m, max_order)}


def on_screen(x_mm, margin_mm=3 * M.SPOT_SIGMA_M * 1000):
    return x_mm is not None and abs(x_mm) <= SCREEN_MM / 2 - margin_mm and math.isfinite(x_mm)
