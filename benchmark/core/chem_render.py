"""Liquid colour of the chemistry kit, independent of Isaac.

The transmitted spectrum through a viewing path L_view (1 cm for a cuvette) is T(lambda) = 10^(-A_1cm(lambda) L_view).
Under a D65 illuminant, integrated with the CIE 1931 2-degree colour matching functions (the multi-lobe
Gaussian fit of Wyman, Sloan and Shirley 2013), normalised so that water (T = 1) is white, the tristimulus
values give linear sRGB (D65, IEC 61966-2-1 matrix). Linear values go to the renderer; sRGB-encoded values are
what a display shows. Absorption outside the dye table's 380-720 nm grid is taken as zero.
"""
from __future__ import annotations

import numpy as np

import chem_solver as S

# CIE D65 relative spectral power, 380-720 nm in 10 nm steps.
D65_NM = np.arange(380.0, 721.0, 10.0)
D65 = np.array([49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35,
                107.80, 104.79, 107.69, 104.41, 104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29,
                83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35, 61.60])
XYZ_TO_LINEAR_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(nm):
    """CIE 1931 x-bar, y-bar, z-bar (Wyman-Sloan-Shirley multi-lobe fit)."""
    x = 1.056 * _g(nm, 599.8, 37.9, 31.0) + 0.362 * _g(nm, 442.0, 16.0, 26.7) - 0.065 * _g(nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(nm, 568.8, 46.9, 40.5) + 0.286 * _g(nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(nm, 437.0, 11.8, 36.0) + 0.681 * _g(nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


_NM = S.GRID_NM[::10]                                   # 1 nm steps
_WEIGHTS = cmf(_NM) * np.interp(_NM, D65_NM, D65)       # 3 x N
_WHITE = _WEIGHTS.sum(axis=1)


def absorbance_spectrum(composition_M, dyes):
    """A_1cm on the 1 nm grid of a solution (mol/L by dye name; other species do not absorb)."""
    total = np.zeros(len(_NM))
    for name, c in composition_M.items():
        if name in dyes and c > 0:
            total += dyes[name].eps[::10] * c
    return total


def linear_rgb(composition_M, dyes, path_cm=1.0):
    """Linear sRGB (0..1) of white D65 light through path_cm of the solution; water gives (1, 1, 1)."""
    transmitted = 10.0 ** (-absorbance_spectrum(composition_M, dyes) * path_cm)
    xyz = (_WEIGHTS * transmitted).sum(axis=1) / _WHITE[1]
    white = XYZ_TO_LINEAR_SRGB @ (_WHITE / _WHITE[1])
    rgb = (XYZ_TO_LINEAR_SRGB @ xyz) / white        # white balance: water renders neutral
    return np.clip(rgb, 0.0, 1.0)


def encode_srgb(linear):
    linear = np.clip(np.asarray(linear, float), 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
