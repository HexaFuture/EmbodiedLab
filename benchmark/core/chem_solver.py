"""Chemistry kit solver, independent of Isaac.

Dyes: molar absorptivity spectra eps(lambda) (L mol^-1 cm^-1) as sums of Gaussian bands, scaled so that
the effective spectrum (convolved with the 2 nm triangular slit) peaks at the literature eps(lambda_max).
The band table is a first approximation of the published aqueous spectra (refit to digitised
literature spectra, residual <= 2 % of the peak, lambda_max within 1 nm).

Photometry of a single-beam instrument: the detector signal through a cuvette is
    S = I0(lambda) * (10^-(A_content + b(lambda)) + s)
with stray light s and an instrument/cuvette baseline b(lambda) = b0 + b1 (lambda - 550)/150. A reading is
    A = log10(S_ref / S)
where S_ref is the signal stored by the last BLANK at that wavelength, or, without a valid BLANK, the factory
reference I0 (1 + s). So a water blank removes b, a dyed blank lowers every later reading, and a cuvette whose
liquid does not reach the beam reads like an empty (water-free) path. Noise sigma_A = 0.001 + 0.002 A is added
to A and the display rounds to 3 decimals. All path lengths are 1.00 cm.
"""
from __future__ import annotations

import math

import numpy as np

STRAY_LIGHT_MAX = 0.001        # the instrument spec (cards); each instance draws its own s <= this
SLIT_NM = 2.0
PATH_CM = 1.0
GRID_NM = np.round(np.arange(380.0, 720.0 + 1e-9, 0.1), 1)
SCAN_NM = (400, 700)

# name: (lambda_max literature nm, eps(lambda_max) literature, [(centre nm, sigma nm, relative height), ...])
# Values: literature order of magnitude; band shapes approximate the aqueous spectra.
DYES = {
    'tartrazine': (427.0, 2.7e4, [(427.0, 38.0, 1.0), (360.0, 25.0, 0.25)]),
    'allura_red': (504.0, 2.6e4, [(506.0, 36.0, 1.0), (465.0, 28.0, 0.30)]),
    'crystal_violet': (590.0, 8.7e4, [(592.0, 22.0, 1.0), (545.0, 25.0, 0.45)]),
    'brilliant_blue_fcf': (630.0, 1.3e5, [(630.0, 20.0, 1.0), (585.0, 25.0, 0.25), (410.0, 25.0, 0.05)]),
    'methylene_blue': (664.0, 7.4e4, [(665.0, 17.0, 1.0), (612.0, 24.0, 0.40)]),   # ideal monomer model
}


def register_dye(name, lambda_max_nm, eps_max, bands):
    """Add a dye of given band parameters (dye_absorptivity's unknown dye) under `name` for this process."""
    DYES[name] = (float(lambda_max_nm), float(eps_max), [tuple(float(v) for v in band) for band in bands])


def _raw(dye, grid=GRID_NM, shift_nm=0.0):
    _, _, bands = DYES[dye]
    grid = np.asarray(grid, float)
    return sum(h * np.exp(-0.5 * ((grid - mu - shift_nm) / sigma) ** 2) for mu, sigma, h in bands)


def _slit(values, step_nm=0.1, fwhm_nm=SLIT_NM):
    """Convolution with a triangular slit function of the given FWHM (base 2 x FWHM)."""
    half = int(round(fwhm_nm / step_nm))
    kernel = 1.0 - np.abs(np.arange(-half, half + 1)) / half
    kernel = kernel[kernel > 0]
    kernel /= kernel.sum()
    padded = np.pad(values, len(kernel) // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


class Dye:
    """Effective (slit-convolved) molar absorptivity of one dye batch."""

    def __init__(self, name, eps_factor=1.0):
        if name not in DYES:
            raise ValueError(f'unknown dye {name!r}')
        if not 0.5 < eps_factor < 1.5:
            raise ValueError('eps_factor out of range')
        self.name = name
        lit_peak, lit_eps, _ = DYES[name]
        window = (GRID_NM >= lit_peak - 30) & (GRID_NM <= lit_peak + 30)
        # Shoulder bands pull the effective maximum off the main band centre: shift all bands together so the
        # effective spectrum peaks at the literature lambda_max (to the 0.1 nm grid).
        shift = 0.0
        for _ in range(5):
            effective = _slit(_raw(name, shift_nm=shift))
            peak_nm = float(GRID_NM[np.flatnonzero(window)[np.argmax(effective[window])]])
            if abs(peak_nm - lit_peak) < 0.05:
                break
            shift += lit_peak - peak_nm
        scale = lit_eps / effective[window].max()
        self.eps = effective * scale * eps_factor          # on GRID_NM
        peak = int(np.flatnonzero(window)[np.argmax(effective[window])])
        self.lambda_max_nm = float(GRID_NM[peak])

    def eps_at(self, wavelength_nm):
        return float(np.interp(wavelength_nm, GRID_NM, self.eps))


def absorbance(composition_M, dyes, wavelength_nm, path_cm=PATH_CM):
    """True absorbance of a solution: sum eps_i(lambda) c_i l (composition in mol/L by dye name)."""
    return sum(dyes[name].eps_at(wavelength_nm) * c * path_cm for name, c in composition_M.items())


def baseline(wavelength_nm, b0, b1):
    return b0 + b1 * (wavelength_nm - 550.0) / 150.0


def signal(a_content, wavelength_nm, b0, b1, stray):
    """Detector signal relative to the lamp intensity I0 for a path of absorbance a_content (None: no liquid
    in the beam, i.e. only the cuvette/instrument baseline)."""
    a = baseline(wavelength_nm, b0, b1) + (a_content or 0.0)
    return 10.0 ** (-a) + stray


def factory_reference(stray):
    return 1.0 + stray


def reading(s_ref, s_sample):
    return math.log10(s_ref / s_sample)


def noise_sigma(a_meas):
    return 0.001 + 0.002 * abs(a_meas)


def measured_ideal(a_true, stray):
    """The public stray-light formula for a water-blanked reading: A_meas = -log10((10^-A + s)/(1 + s))."""
    return -math.log10((10.0 ** (-a_true) + stray) / (1.0 + stray))
