"""
Map ferric iron-bearing AMD precipitates at the Iron King Mine
Superfund site (Dewey-Humboldt, AZ) from laboratory spectra and airborne/orbital imaging
spectroscopy.

Inputs (paths are relative to the current working directory; run from the
repository root):

- ``data/asd/*.txt``             ASD spectra, one file per sample
- ``data/emit_ortho_crop.nc``    EMIT L2A reflectance, orthorectified and cropped
- ``data/aviris_rfl.hdr``        AVIRIS reflectance, ENVI format (+ binary)
- ``data/aviris5_rfl.nc``        AVIRIS-5 reflectance, NetCDF with a
                                 ``reflectance`` group
- ``data/IronKing.kml``          Area-of-interest (AOI) polygon(s)
- ``ref_spectra/{crowley,usgs_minerals,relab}``  laboratory reference spectra

Mineral classes (``MINERALS``): 1 coquimbite, 2 rhomboclase, 3 ferricopiapite,
4 hematite, 5 jarosite, 6 goethite, 7 schwertmannite, 8 ferrihydrite;
0 means unclassified.
"""
import os
import re
import glob
import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
# Registers the ``.rio`` accessor used in main(). It would otherwise only be
# registered as a side effect of xr.open_dataset loading rioxarray's backend.
import rioxarray  # noqa: F401
import geopandas as gpd
import spectral
from spectral import BandResampler
from spectral.algorithms.algorithms import spectral_angles
from pysptools.spectro import SpectrumConvexHullQuotient
from pysptools.abundance_maps.amaps import NNLS
from shapely.geometry import mapping, box
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol
from rasterio.features import geometry_mask
from pyproj import Transformer

# Silences all library warnings (e.g. all-NaN slices in nanmean, CRS notices).
warnings.simplefilter('ignore')

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input / output locations
# ---------------------------------------------------------------------------
DATA = 'data'
ASD_DIR = f'{DATA}/asd'
EMIT_PATH = f'{DATA}/emit_ortho_crop.nc'
AVIRIS_HDR = f'{DATA}/aviris_rfl.hdr'
AVIRIS5_PATH = f'{DATA}/aviris5_rfl.nc'
AOI_PATH = f'{DATA}/IronKing.kml'
REF_DIR = 'ref_spectra'
USGS_WAVELENGTH_FILE = (
    f'{REF_DIR}/usgs_minerals/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')
OUT_DIR = 'outputs'

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------
# Atmospheric water-vapor windows (nm) set to NaN in every spectrum.
BAD_WAVELENGTH_WINDOWS = [(1275, 1475), (1750, 1975)]
# SAM pixels whose best spectral angle exceeds this (radians) are unclassified.
SAM_MAX_ANGLE_RAD = 0.2
# WGS 84 / UTM zone 12N; pit coordinates and AVIRIS-5 grid are in this CRS.
UTM = 'EPSG:32612'

MINERALS = [
    (1, 'Coquimbite'),
    (2, 'Rhomboclase'),
    (3, 'Ferricopiapite'),
    (4, 'Hematite'),
    (5, 'Jarosite'),
    (6, 'Goethite'),
    (7, 'Schwertmannite'),
    (8, 'Ferrihydrite'),
]
MINERAL_NAMES = [name for _, name in MINERALS]
MINERAL_KEYS = [name.lower() for name in MINERAL_NAMES]
# Reserved class id; no classifier in this script currently emits it.
UNCONFIRMED_ID = 9

# Field sampling pits: name -> (easting, northing) in UTM 12N meters.
# Each name must match an ASD file stem in ASD_DIR (e.g. IK24-PIT-001.txt).
PITS = {
    'IK24-PIT-001': (384838, 3818515),
    'IK24-PIT-002': (384968, 3818473),
    'IK24-PIT-003': (384765, 3818461),
    'IK24-PIT-004': (385065, 3818422),
    'IK24-PIT-005': (384925, 3818328),
    'IK24-PIT-006': (385095, 3818306),
    'IK24-PIT-007': (385018, 3818178),
    'IK24-PIT-008': (385256, 3818223),
    'IK24-PIT-009': (385428, 3818247),
    'IK24-PIT-010': (385323, 3818149),
}

# Decision-tree positions written to band_params.csv. A center is NaN when
# that band is absent or shallower than 0.01. Depths, areas, and WAC are
# not calculated. See compute_band_params.
PARAM_KEYS = [
    'b900_center', 'b900_start', 'b2265_center',
    'cr_500_600', 'cr_450_550', 'cr_600_750',
]


# ---------------------------------------------------------------------------
# Generic spectrum utilities
# ---------------------------------------------------------------------------
def bad_wavelength_mask(wavelengths: npt.ArrayLike) -> np.ndarray:
    """
    Flag bands that fall inside the atmospheric water-vapor windows.

    Parameters
    ----------
    wavelengths : array_like
        Band centers in nm, shape (n_bands,).

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n_bands,); True where the band lies inside any
        interval of ``BAD_WAVELENGTH_WINDOWS`` (inclusive).
    """
    wvl = np.asarray(wavelengths, dtype=float)
    bad = np.zeros(len(wvl), dtype=bool)
    for lo, hi in BAD_WAVELENGTH_WINDOWS:
        bad |= (wvl >= lo) & (wvl <= hi)
    return bad


def apply_bad_wavelength_mask(wavelengths: npt.ArrayLike, reflectance: npt.ArrayLike) -> np.ndarray:
    """
    Return a flattened copy of a spectrum with water-vapor bands set to NaN.

    Parameters
    ----------
    wavelengths : array_like
        Band centers in nm, shape (n_bands,).
    reflectance : array_like
        Reflectance values; flattened to shape (n_bands,).

    Returns
    -------
    np.ndarray
        Float copy of ``reflectance`` with bad bands replaced by NaN.
    """
    out = np.asarray(reflectance, dtype=float).copy().reshape(-1)
    out[bad_wavelength_mask(wavelengths)] = np.nan
    return out


def max_normalize_spectrum(rfl: npt.ArrayLike) -> np.ndarray:
    """
    Scale a spectrum so its largest finite, positive value equals 1.

    Parameters
    ----------
    rfl : array_like
        Reflectance spectrum (any shape).

    Returns
    -------
    np.ndarray
        Float copy divided by its finite positive maximum. Returned unscaled if
        no finite positive values exist.
    """
    out = np.asarray(rfl, dtype=float).copy()
    finite = np.isfinite(out) & (out > 0)
    if not finite.any():
        return out
    peak = float(np.max(out[finite]))
    if peak > 0:
        out = out / peak
    return out


def envi_georef_from_hdr(hdr_path: str) -> tuple[CRS, Affine]:
    """
    Build a CRS and affine transform from an ENVI header's ``map info`` field.

    Parameters
    ----------
    hdr_path : str
        Path to the ENVI ``.hdr`` file.

    Returns
    -------
    crs : rasterio.crs.CRS
        UTM CRS from the zone/hemisphere fields, or the CRS parsed from
        ``coordinate system string`` for non-UTM projections.
    transform : rasterio.transform.Affine
        North-up pixel-to-map transform (rotation is not supported). The
        ``map info`` tie point is the upper-left corner of the 1-based
        reference pixel, matching GDAL's ENVI driver.

    Raises
    ------
    ValueError
        If the header has no ``map info``, or uses a non-UTM projection
        without a ``coordinate system string``.
    """
    text = open(hdr_path).read()
    m = re.search(r'map info\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)
    if not m:
        raise ValueError(f'No map info in ENVI header: {hdr_path}')
    # map info = {proj, ref_px, ref_py, map_x, map_y, size_x, size_y, zone, hemisphere, ...}
    parts = [p.strip() for p in m.group(1).split(',')]
    proj_name = parts[0]
    ref_px, ref_py = float(parts[1]), float(parts[2])
    map_x, map_y = float(parts[3]), float(parts[4])
    ps_x, ps_y = float(parts[5]), abs(float(parts[6]))
    zone = int(float(parts[7]))
    hemisphere = parts[8].strip().lower()
    if proj_name.upper() == 'UTM':
        epsg = (32600 if hemisphere.startswith('north') else 32700) + zone
        crs = CRS.from_epsg(epsg)
    else:
        m_wkt = re.search(r'coordinate system string\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)
        if not m_wkt:
            raise ValueError(f'Unsupported ENVI projection: {proj_name}')
        crs = CRS.from_wkt(m_wkt.group(1))
    # Tie point is the upper-left corner of the 1-based reference pixel.
    west = map_x - ps_x * (ref_px - 1.0)
    north = map_y + ps_y * (ref_py - 1.0)
    return crs, Affine(ps_x, 0.0, west, 0.0, -ps_y, north)


def scale_aviris_reflectance(pix: npt.ArrayLike) -> np.ndarray:
    """
    Convert an AVIRIS pixel spectrum to unit reflectance.

    Parameters
    ----------
    pix : array_like
        Raw pixel spectrum, shape (n_bands,).

    Returns
    -------
    np.ndarray
        Flattened float copy, divided by 10000 if it appears to be stored as
        scaled integers (max > 2), with values below -0.1 set to NaN.
    """
    pix = np.asarray(pix, dtype=float).copy().ravel()
    if pix.size and np.nanmax(pix) > 2:
        pix /= 10000.0
    pix[pix < -0.1] = np.nan
    return pix


def nearest_emit_valid_mask(
    src_wvl: npt.ArrayLike, emit_wvl: npt.ArrayLike, emit_rfl: npt.ArrayLike
) -> np.ndarray:
    """
    Transfer EMIT's valid-band mask onto another sensor's band grid.

    Each source band inherits the validity (finite and > 0) of the nearest
    EMIT band, so spectra from different sensors share the same gaps.

    Parameters
    ----------
    src_wvl : array_like
        Band centers of the target sensor in nm, shape (n_src,).
    emit_wvl : array_like
        EMIT band centers in nm, shape (n_emit,).
    emit_rfl : array_like
        EMIT reflectance spectrum, shape (n_emit,).

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n_src,).
    """
    emit_wvl = np.asarray(emit_wvl, dtype=float)
    src_wvl = np.asarray(src_wvl, dtype=float)
    emit_ok = np.isfinite(emit_rfl) & (np.asarray(emit_rfl, dtype=float) > 0)
    idx = np.abs(emit_wvl[:, None] - src_wvl[None, :]).argmin(axis=0)
    return emit_ok[idx]


def gaussian_resample(
    src_wvl: npt.ArrayLike, src_rfl: npt.ArrayLike, dst_wvl: npt.ArrayLike, dst_fwhm: npt.ArrayLike
) -> np.ndarray:
    """
    Convolve a spectrum to a target band set with Gaussian spectral responses.

    Each output band is the Gaussian-weighted mean of finite source samples
    within +/- 3 sigma of the band center (sigma = FWHM / 2.3548).

    Parameters
    ----------
    src_wvl : array_like
        Source wavelengths in nm, shape (n_src,).
    src_rfl : array_like
        Source reflectance, shape (n_src,).
    dst_wvl : array_like
        Target band centers in nm, shape (n_dst,).
    dst_fwhm : array_like
        Target band FWHM in nm, shape (n_dst,).

    Returns
    -------
    np.ndarray
        Resampled reflectance, shape (n_dst,). NaN for target bands outside the
        source range, with invalid FWHM, or with no source samples in range.
    """
    src_wvl = np.asarray(src_wvl, dtype=float)
    src_rfl = np.asarray(src_rfl, dtype=float)
    dst_wvl = np.asarray(dst_wvl, dtype=float)
    dst_fwhm = np.asarray(dst_fwhm, dtype=float)
    out = np.full(len(dst_wvl), np.nan)
    good = np.isfinite(src_wvl) & np.isfinite(src_rfl)
    sw, sr = src_wvl[good], src_rfl[good]
    if sw.size < 2:
        return out
    order = np.argsort(sw)
    sw, sr = sw[order], sr[order]
    sig = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM-to-sigma factor
    for i, (c, f) in enumerate(zip(dst_wvl, dst_fwhm)):
        if not (np.isfinite(c) and np.isfinite(f) and f > 0):
            continue
        if c < sw[0] or c > sw[-1]:
            continue
        sigma = f / sig
        lo, hi = c - 3.0 * sigma, c + 3.0 * sigma
        m = (sw >= lo) & (sw <= hi)
        if m.sum() < 1:
            continue
        w = np.exp(-0.5 * ((sw[m] - c) / sigma) ** 2)
        wsum = w.sum()
        if wsum > 0:
            out[i] = float(np.sum(w * sr[m]) / wsum)
    return out


# ---------------------------------------------------------------------------
# Continuum removal and band parameters
# ---------------------------------------------------------------------------
def continuum_removal(refl_values: npt.ArrayLike, wavelengths: npt.ArrayLike) -> np.ndarray:
    """
    Divide a spectrum by its upper convex hull (continuum removal).

    Water-vapor bands and non-positive values are excluded before the hull is
    fitted.

    Parameters
    ----------
    refl_values : array_like
        Reflectance spectrum, shape (n_bands,).
    wavelengths : array_like
        Band centers in nm, shape (n_bands,).

    Returns
    -------
    np.ndarray
        Continuum-removed spectrum, shape (n_bands,), with values in (0, 1]
        on valid bands and NaN elsewhere. All NaN if fewer than 11 valid
        bands remain or the hull fit fails.
    """
    refl = apply_bad_wavelength_mask(wavelengths, refl_values)
    mask = np.isfinite(refl) & (refl > 0)
    cr = np.full(len(wavelengths), np.nan)
    if mask.sum() < 11:
        return cr
    try:
        chq = SpectrumConvexHullQuotient(refl[mask].tolist(), np.asarray(wavelengths, dtype=float)[mask].tolist())
        cr[mask] = chq.get_continuum_removed_spectrum()
    except (ValueError, TypeError):
        return cr
    return cr


def vis_cr_band_center(
    wvl: np.ndarray, cr: np.ndarray, wmin: float, wmax: float, min_depth: float = 0.01
) -> float:
    """
    Wavelength of the continuum-removed minimum inside a window.

    Parameters
    ----------
    wvl : np.ndarray
        Band centers in nm.
    cr : np.ndarray
        Continuum-removed spectrum on ``wvl``.
    wmin, wmax : float
        Inclusive search window in nm.
    min_depth : float, optional
        Minimum band depth (1 - CR) required to report a center.

    Returns
    -------
    float
        Band-center wavelength in nm, or NaN if fewer than 3 valid bands are
        in the window or the band is not deeper than ``min_depth``.
    """
    m = (wvl >= wmin) & (wvl <= wmax) & np.isfinite(wvl) & np.isfinite(cr)
    if m.sum() < 3:
        return np.nan
    w_seg, cr_seg = wvl[m], cr[m]
    i = int(np.argmin(cr_seg))
    if (1.0 - float(cr_seg[i])) <= min_depth:
        return np.nan
    return float(w_seg[i])


def first_cr_unity(
    wavelengths: npt.ArrayLike, cr: npt.ArrayLike, wmin: float, wmax: float, descending: bool = False
) -> float:
    """
    First wavelength in a window where the continuum-removed spectrum reaches 1.

    A CR value of 1 marks a convex-hull tie point, i.e. a band shoulder.

    Parameters
    ----------
    wavelengths : array_like
        Band centers in nm.
    cr : array_like
        Continuum-removed spectrum on ``wavelengths``.
    wmin, wmax : float
        Inclusive search window in nm.
    descending : bool, optional
        If True, scan from ``wmax`` down to ``wmin`` (finds the short-wavelength
        shoulder nearest a band); otherwise scan upward.

    Returns
    -------
    float
        Shoulder wavelength in nm, or NaN if none is found.
    """
    wvl = np.asarray(wavelengths, dtype=float)
    cr = np.asarray(cr, dtype=float)
    sel = (wvl >= wmin) & (wvl <= wmax)
    w, c = wvl[sel], cr[sel]
    if len(w) < 2:
        return np.nan
    order = np.argsort(-w if descending else w)
    w, c = w[order], c[order]
    good = np.isfinite(c)
    w, c = w[good], c[good]
    hit = np.where(c >= 1.0)[0]
    return float(w[hit[0]]) if len(hit) else np.nan


def compute_band_params(
    cr: npt.ArrayLike,
    wavelengths: npt.ArrayLike,
    b900_max: float = 1275,
) -> dict[str, float] | None:
    """
    Derive the band positions used by the decision tree.

    A center is recorded only when the continuum-removed band is deeper than
    0.01. Numeric depths, band areas, and the area-bisecting wavelength are
    not computed.

    Parameters
    ----------
    cr : array_like
        Continuum-removed spectrum, shape (n_bands,).
    wavelengths : array_like
        Band centers in nm, shape (n_bands,).
    b900_max : float, optional
        Upper limit (nm) of the ~900 nm Fe3+ band search window. The default,
        1275 nm, is the short-wavelength edge of the first water-vapor window.
        The lower limit is 720 nm.

    Returns
    -------
    dict or None
        Keys (NaN when a feature is absent or shallower than 0.01):

        - ``b900_center``: CR minimum in 720-``b900_max`` nm.
        - ``b900_start``: short-wavelength shoulder of that band (first CR = 1
          scanning down from 800 to 625 nm).
        - ``b2265_center``: jarosite feature in 2230-2290 nm.
        - ``cr_500_600``, ``cr_450_550``, ``cr_600_750``: CR minimum in each
          window, used by the coquimbite, rhomboclase, and goethite tests.

        None if fewer than 11 valid CR values or fewer than 3 in the b900
        window.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    cr = np.asarray(cr, dtype=float)
    valid = ~np.isnan(cr)
    wvl_v, cr_v = wavelengths[valid], cr[valid]
    if len(cr_v) < 11:
        return None

    def store_center(
        result: dict[str, float], prefix: str, wvl_seg: np.ndarray, cr_seg: np.ndarray,
    ) -> bool:
        """
        Write ``{prefix}_center`` when the band is deeper than 0.01.

        Returns True if a center was stored.
        """
        result[f'{prefix}_center'] = np.nan
        if len(wvl_seg) < 3:
            return False
        min_idx = int(np.argmin(cr_seg))
        if (1.0 - float(cr_seg[min_idx])) <= 0.01:
            return False
        result[f'{prefix}_center'] = float(wvl_seg[min_idx])
        return True

    # ~900 nm Fe3+ crystal-field band
    b900_min = 720
    bm = (wvl_v >= b900_min) & (wvl_v <= b900_max)
    wav_seg, cr_seg = wvl_v[bm], cr_v[bm]
    if len(wav_seg) < 3:
        return None
    result: dict[str, float] = {}
    if store_center(result, 'b900', wav_seg, cr_seg):
        result['b900_start'] = first_cr_unity(wvl_v, cr_v, 625, 800, descending=True)
    else:
        result['b900_start'] = np.nan
    # ~2265 nm jarosite (Fe-OH) band
    store_center(result, 'b2265', wvl_v[(wvl_v >= 2230) & (wvl_v <= 2290)],
                 cr_v[(wvl_v >= 2230) & (wvl_v <= 2290)])
    result['cr_500_600'] = vis_cr_band_center(wvl_v, cr_v, 500, 600)
    result['cr_450_550'] = vis_cr_band_center(wvl_v, cr_v, 450, 550)
    result['cr_600_750'] = vis_cr_band_center(wvl_v, cr_v, 600, 750)
    return result


def band_params_from_spectrum(
    wvl: npt.ArrayLike, rfl: npt.ArrayLike, b900_max: float = 1275
) -> dict[str, Any] | None:
    """
    Continuum-remove a reflectance spectrum and compute its band parameters.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm, shape (n_bands,).
    rfl : array_like
        Reflectance spectrum, shape (n_bands,).
    b900_max : float, optional
        Upper limit of the ~900 nm band window, passed to
        ``compute_band_params``.

    Returns
    -------
    dict or None
        Output of ``compute_band_params`` plus ``cr`` (continuum-removed
        spectrum) and ``wavelengths``; None if there are fewer than 11 finite
        reflectance values or no parameters could be computed.
    """
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    if np.isfinite(rfl).sum() < 11:
        return None
    cr = continuum_removal(rfl, wvl)
    params = compute_band_params(cr, wvl, b900_max=b900_max)
    if not params:
        return None
    params['cr'] = cr
    params['wavelengths'] = wvl
    return params


def cr_absorption_depth(
    cr: npt.ArrayLike, wvl: npt.ArrayLike, wmin: float, wmax: float, min_points: int = 2
) -> tuple[float, float]:
    """
    Depth and position of the continuum-removed minimum in a window.

    Parameters
    ----------
    cr : array_like
        Continuum-removed spectrum.
    wvl : array_like
        Band centers in nm.
    wmin, wmax : float
        Inclusive window in nm.
    min_points : int, optional
        Minimum number of finite CR values required in the window.

    Returns
    -------
    depth : float
        1 - min(CR), or 0.0 if too few points.
    center : float
        Wavelength of the minimum in nm, or NaN if too few points.
    """
    wvl = np.asarray(wvl, dtype=float)
    cr = np.asarray(cr, dtype=float)
    m = (wvl >= wmin) & (wvl <= wmax) & np.isfinite(cr)
    if m.sum() < min_points:
        return 0.0, np.nan
    i = int(np.argmin(cr[m]))
    return float(1.0 - cr[m][i]), float(wvl[m][i])


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def refined_mineral_class(params: dict[str, Any]) -> int:
    """
    Rule-based decision tree that assigns a mineral class from band parameters.

    The first split is the ~900 nm band center (``b900_center``); each branch
    then checks a second diagnostic feature. The feature/mineral pairs are the
    ones summarized by ``Reference_Spectra_Analysis.py``; the numeric windows
    are hard-coded here.

    ==============  ===============  ==========================================
    b900 center     Class            Second test
    ==============  ===============  ==========================================
    780-800 nm      1 Coquimbite     CR minimum 545-590 nm (500-600 window),
                                     and no rhomboclase band
                    2 Rhomboclase    CR minimum 500-550 nm (450-550 window),
                                     and no coquimbite band
    865-895 nm      3 Ferricopiapite band shoulder (b900_start) 660-695 nm
                    4 Hematite       band shoulder 725-755 nm
    910-940 nm      5 Jarosite       2230-2290 nm band centered 2255-2275 nm
    940-980 nm      6 Goethite       CR minimum 625-715 nm (600-750 window)
                    7 Schwertmannite otherwise, band shoulder 710-750 nm
    980-1010 nm     8 Ferrihydrite   band shoulder 755-780 nm
    ==============  ===============  ==========================================

    A center is stored only when its band is deeper than 0.01, so a missing
    center fails that test. Visible-band checks apply the same depth cutoff.

    Parameters
    ----------
    params : dict
        Output of ``band_params_from_spectrum`` (must include ``cr`` and
        ``wavelengths``).

    Returns
    -------
    int
        Class id 1-8, or 0 if no rule matches.
    """
    b900 = params.get('b900_center', np.nan)
    if not np.isfinite(b900):
        return 0
    cr, wvl = params['cr'], params['wavelengths']
    b900_start = params.get('b900_start', np.nan)
    if 780 <= b900 <= 800:
        d_coq, c_coq = cr_absorption_depth(cr, wvl, 500, 600)
        d_rho, c_rho = cr_absorption_depth(cr, wvl, 450, 550)
        has_coq = np.isfinite(c_coq) and d_coq > 0.01 and 545 <= c_coq <= 590
        has_rho = np.isfinite(c_rho) and d_rho > 0.01 and 500 <= c_rho <= 550
        if has_coq and not has_rho:
            return 1
        if has_rho and not has_coq:
            return 2
        return 0
    if 865 <= b900 <= 895:
        if np.isfinite(b900_start) and 660 <= b900_start <= 695:
            return 3
        if np.isfinite(b900_start) and 725 <= b900_start <= 755:
            return 4
        return 0
    if 910 <= b900 <= 940:
        c2265 = params.get('b2265_center', np.nan)
        if np.isfinite(c2265) and 2255 <= c2265 <= 2275:
            return 5
        return 0
    if 940 < b900 < 980:
        d_goe, c_goe = cr_absorption_depth(cr, wvl, 600, 750)
        if np.isfinite(c_goe) and d_goe > 0.01 and 625 <= c_goe <= 715:
            return 6
        if np.isfinite(b900_start) and 710 <= b900_start <= 750:
            return 7
        return 0
    if 980 <= b900 <= 1010 and np.isfinite(b900_start) and 755 <= b900_start <= 780:
        return 8
    return 0


def refined_class_from_spectrum(wvl: npt.ArrayLike, rfl: npt.ArrayLike) -> int:
    """
    Classify one reflectance spectrum with the decision tree.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm.
    rfl : array_like
        Reflectance spectrum.

    Returns
    -------
    int
        Class id 0-8 (see ``refined_mineral_class``).
    """
    params = band_params_from_spectrum(wvl, rfl)
    if not params:
        return 0
    return refined_mineral_class(params)


def prep_spectrum_for_sam(wvl: npt.ArrayLike, rfl: npt.ArrayLike) -> np.ndarray:
    """
    Mask water-vapor and non-positive bands, then max-normalize, for SAM.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm.
    rfl : array_like
        Reflectance spectrum.

    Returns
    -------
    np.ndarray
        Max-normalized spectrum with invalid bands as NaN.
    """
    rfl = apply_bad_wavelength_mask(wvl, rfl)
    rfl = np.asarray(rfl, dtype=float)
    rfl[~(np.isfinite(rfl) & (rfl > 0))] = np.nan
    return max_normalize_spectrum(rfl)


def sam_class_from_spectrum(wvl: npt.ArrayLike, rfl: npt.ArrayLike, members: npt.ArrayLike) -> int:
    """
    Spectral Angle Mapper classification of one spectrum.

    Only bands that are valid in the pixel and in every endmember are used;
    both are re-normalized on that common band set.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm, shape (n_bands,).
    rfl : array_like
        Reflectance spectrum, shape (n_bands,).
    members : array_like
        Endmember spectra, shape (n_minerals, n_bands), ordered as
        ``MINERALS``.

    Returns
    -------
    int
        1-based index of the endmember with the smallest angle, or 0 if fewer
        than 10 common bands, any angle is non-finite, or the best angle
        exceeds ``SAM_MAX_ANGLE_RAD``.
    """
    pix = prep_spectrum_for_sam(wvl, rfl)
    members = np.asarray(members, dtype=float).copy()
    members[:, bad_wavelength_mask(wvl)] = np.nan
    valid = np.isfinite(pix) & np.all(np.isfinite(members), axis=0)
    if valid.sum() < 10:
        return 0
    pix_n = max_normalize_spectrum(pix[valid])
    mem = np.vstack([max_normalize_spectrum(row) for row in members[:, valid]])
    # spectral_angles expects an image cube; wrap the pixel as 1x1.
    ang = np.asarray(spectral_angles(pix_n.reshape(1, 1, -1), mem), dtype=float).ravel()
    if not np.all(np.isfinite(ang)):
        return 0
    k = int(np.argmin(ang))
    if SAM_MAX_ANGLE_RAD is not None and ang[k] > SAM_MAX_ANGLE_RAD:
        return 0
    return k + 1


def prep_spectrum_for_unmix(wvl: npt.ArrayLike, rfl: npt.ArrayLike) -> np.ndarray:
    """
    Mask water-vapor bands, zero negative values, then max-normalize, for NNLS.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm.
    rfl : array_like
        Reflectance spectrum.

    Returns
    -------
    np.ndarray
        Max-normalized spectrum; NaN on masked or non-finite bands.
    """
    rfl = apply_bad_wavelength_mask(wvl, rfl)
    rfl = np.asarray(rfl, dtype=float)
    rfl[np.isfinite(rfl) & (rfl < 0)] = 0.0
    rfl[~np.isfinite(rfl)] = np.nan
    return max_normalize_spectrum(rfl)


def nnls_abundances(
    wvl: npt.ArrayLike, specs: npt.ArrayLike, members: npt.ArrayLike, min_bands: int = 10
) -> np.ndarray:
    """
    Non-negative least-squares unmixing of spectra against mineral endmembers.

    Spectra and endmembers are max-normalized on the bands where every
    endmember is finite. Abundances are constrained to be >= 0 but are not
    forced to sum to one, so they are relative weights rather than areal
    fractions.

    Parameters
    ----------
    wvl : array_like
        Band centers in nm, shape (n_bands,).
    specs : array_like
        One spectrum (n_bands,) or many (n_spectra, n_bands).
    members : array_like
        Endmembers, shape (n_minerals, n_bands).
    min_bands : int, optional
        Minimum number of common valid bands; otherwise all-NaN output.

    Returns
    -------
    np.ndarray
        Abundances, shape (n_minerals,) for one input spectrum or
        (n_spectra, n_minerals). Rows are NaN for spectra with any invalid
        band in the common band set.
    """
    specs = np.asarray(specs, dtype=float)
    one = specs.ndim == 1
    if one:
        specs = specs.reshape(1, -1)
    wvl = np.asarray(wvl, dtype=float)
    U = np.asarray(members, dtype=float).copy()
    U[:, bad_wavelength_mask(wvl)] = np.nan
    M = np.asarray([prep_spectrum_for_unmix(wvl, row) for row in specs], dtype=float)
    band_ok = np.all(np.isfinite(U), axis=0)
    if int(band_ok.sum()) < min_bands:
        abund = np.full((specs.shape[0], U.shape[0]), np.nan)
        return abund[0] if one else abund
    U_use = np.vstack([max_normalize_spectrum(row) for row in U[:, band_ok]])
    M_use = M[:, band_ok]
    pix_ok = np.all(np.isfinite(M_use), axis=1)
    abund = np.full((specs.shape[0], U.shape[0]), np.nan, dtype=float)
    if pix_ok.any():
        M_fit = np.vstack([max_normalize_spectrum(row) for row in M_use[pix_ok]])
        abund[pix_ok] = NNLS(M_fit, U_use)
    return abund[0] if one else abund


def nnls_class_from_abundance(abund: npt.ArrayLike) -> int:
    """
    Class id of the dominant (largest positive) NNLS abundance.

    Parameters
    ----------
    abund : array_like
        Abundance vector, shape (n_minerals,).

    Returns
    -------
    int
        1-based mineral id, or 0 if the vector is all NaN, has the wrong
        length, or its maximum is <= 0.
    """
    abund = np.asarray(abund, dtype=float).ravel()
    if abund.size != len(MINERAL_KEYS) or not np.any(np.isfinite(abund)):
        return 0
    a = np.where(np.isfinite(abund), abund, -np.inf)
    k = int(np.argmax(a))
    if not np.isfinite(abund[k]) or abund[k] <= 0:
        return 0
    return k + 1


def class_from_abund_cube(abund: np.ndarray) -> np.ndarray:
    """
    Per-pixel dominant-abundance class map.

    Parameters
    ----------
    abund : np.ndarray
        Abundance cube, shape (n_rows, n_cols, n_minerals).

    Returns
    -------
    np.ndarray
        uint8 class map, shape (n_rows, n_cols); 0 where all abundances are
        NaN or the maximum is <= 0.
    """
    cls = np.zeros(abund.shape[:2], dtype=np.uint8)
    finite = np.any(np.isfinite(abund), axis=2)
    if finite.any():
        a = np.where(np.isfinite(abund), abund, -np.inf)
        k = np.argmax(a, axis=2)
        good = finite & (np.max(a, axis=2) > 0)
        cls[good] = (k[good] + 1).astype(np.uint8)
    return cls


# ---------------------------------------------------------------------------
# Reference spectral library
# ---------------------------------------------------------------------------
def has_vnir_coverage(
    wvl: npt.ArrayLike, rfl: npt.ArrayLike, lo: float = 750.0, hi: float = 1000.0, min_points: int = 10
) -> bool:
    """
    True if a spectrum has at least ``min_points`` finite samples in [lo, hi] nm.

    Parameters
    ----------
    wvl : array_like
        Wavelengths in nm.
    rfl : array_like
        Reflectance.
    lo, hi : float, optional
        Window in nm (default covers the ~900 nm Fe3+ band).
    min_points : int, optional
        Required number of finite samples.

    Returns
    -------
    bool
    """
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    return int(((wvl >= lo) & (wvl <= hi) & np.isfinite(rfl)).sum()) >= min_points


def classify_minerals(text: str) -> set[str]:
    """
    Identify target minerals named in a reference-spectrum label.

    Matching is case-insensitive substring search, so e.g. "natrojarosite"
    counts as jarosite. Paracoquimbite is not counted as coquimbite;
    ``load_reference_spectra`` labels Crowley ``paracoquimbite.txt`` as
    coquimbite separately.

    Parameters
    ----------
    text : str
        File name, header line or metadata text.

    Returns
    -------
    set of str
        Lower-case mineral keys found; a single-element set marks a pure
        reference.
    """
    t = text.lower()
    found = set()
    if 'ferricopiapite' in t:
        found.add('ferricopiapite')
    if 'schwertmannite' in t:
        found.add('schwertmannite')
    if 'ferrihydrite' in t:
        found.add('ferrihydrite')
    if 'rhomboclase' in t:
        found.add('rhomboclase')
    if 'coquimbite' in t and 'paracoquimbite' not in t:
        found.add('coquimbite')
    if 'jarosite' in t:
        found.add('jarosite')
    if 'goethite' in t:
        found.add('goethite')
    if 'hematite' in t:
        found.add('hematite')
    return found


def read_crowley(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a Crowley reference spectrum (two whitespace columns, wavelength in µm).

    Parameters
    ----------
    path : str
        Text file path.

    Returns
    -------
    wvl : np.ndarray
        Wavelengths in nm.
    rfl : np.ndarray
        Reflectance.
    """
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['w', 'r'])
    return df['w'].values * 1000.0, df['r'].values.astype(float)


def read_usgs(path: str, s07_wvl: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read a USGS splib07 ASD-resampled spectrum (one header line, one value per line).

    Parameters
    ----------
    path : str
        Text file path.
    s07_wvl : np.ndarray
        Shared 2151-channel ASD wavelength grid in nm.

    Returns
    -------
    wvl, rfl : np.ndarray or None
        Wavelengths (copy of ``s07_wvl``) and reflectance with USGS "deleted"
        values (< -1) set to NaN; ``(None, None)`` if the channel count does
        not match ``s07_wvl``.
    """
    rfl = pd.read_csv(path, skiprows=1, header=None)[0].values.astype(float)
    rfl[rfl < -1] = np.nan
    if len(rfl) != len(s07_wvl):
        return None, None
    return s07_wvl.copy(), rfl


def read_relab(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a RELAB ``.tab`` spectrum; non-numeric lines are skipped.

    Parameters
    ----------
    path : str
        Text file path; first two whitespace columns are wavelength (nm) and
        reflectance.

    Returns
    -------
    wvl, rfl : np.ndarray
        Rows with 0 < wavelength < 1e5.
    """
    w, r = [], []
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 2:
                continue
            try:
                wi, ri = float(p[0]), float(p[1])
            except ValueError:
                continue
            if 0 < wi < 1e5:
                w.append(wi)
                r.append(ri)
    return np.asarray(w), np.asarray(r)


def usgs_header_name(path: str) -> str:
    """
    Sample description from a USGS file's first line (text after the first ':').

    Parameters
    ----------
    path : str
        USGS spectrum file.

    Returns
    -------
    str
        Header description, or the file stem if the header has no ':'.
    """
    with open(path) as fh:
        header = fh.readline().strip()
    if ':' in header:
        return header.split(':', 1)[-1].strip()
    return os.path.splitext(os.path.basename(path))[0]


def relab_xml_name(tab_path: str) -> tuple[str, str]:
    """
    Mineral subtype and specimen name from the PDS4 label next to a RELAB file.

    Parameters
    ----------
    tab_path : str
        Path to the ``.tab`` file; the label is the same path with ``.xml``.

    Returns
    -------
    subtype, name : str
        Contents of ``speclib:mineral_subtype`` and ``speclib:specimen_name``
        (empty strings if absent).
    """
    xml = tab_path[:-4] + '.xml'
    sub = name = ''
    if os.path.exists(xml):
        txt = open(xml).read()
        m = re.search(r'<speclib:mineral_subtype>([^<]*)', txt)
        sub = m.group(1).strip() if m else ''
        m = re.search(r'<speclib:specimen_name>([^<]*)', txt)
        name = m.group(1).strip() if m else ''
    return sub, name


def load_reference_spectra() -> list[dict[str, Any]]:
    """
    Load the Crowley, USGS and RELAB libraries and keep pure target-mineral spectra.

    A spectrum is kept if its label names exactly one mineral in
    ``MINERAL_KEYS`` and it has VNIR coverage (``has_vnir_coverage``).
    Crowley ``paracoquimbite.txt`` is labeled coquimbite. USGS spectra with
    a noise index above 2e-4 were removed from the library by hand and are
    not filtered here.

    Returns
    -------
    list of dict
        Records with keys ``wvl`` (nm), ``rfl``, ``mineral``, ``is_pure`` and
        ``library``.
    """
    s07 = pd.read_csv(
        USGS_WAVELENGTH_FILE,
        skiprows=1, header=None)[0].values.astype(float) * 1000.0
    records = []
    for path in sorted(glob.glob(f'{REF_DIR}/crowley/*.txt')):
        raw = os.path.splitext(os.path.basename(path))[0]
        w, r = read_crowley(path)
        mins = classify_minerals(raw)
        # Crowley paracoquimbite.txt is a coquimbite reference.
        if raw.lower() == 'paracoquimbite':
            mins = {'coquimbite'}
        records.append(dict(wvl=w, rfl=r, mineral=(next(iter(mins)) if len(mins) == 1 else None),
                            is_pure=len(mins) == 1, library='crowley'))
    for path in sorted(glob.glob(f'{REF_DIR}/usgs_minerals/*.txt')):
        if 'Wavelength' in path:
            continue
        raw = usgs_header_name(path)
        w, r = read_usgs(path, s07)
        if w is None:
            continue
        mins = classify_minerals(raw + ' ' + os.path.basename(path))
        records.append(dict(wvl=w, rfl=r, mineral=(next(iter(mins)) if len(mins) == 1 else None),
                            is_pure=len(mins) == 1, library='usgs'))
    for path in sorted(glob.glob(f'{REF_DIR}/relab/*.tab')):
        sub, name = relab_xml_name(path)
        w, r = read_relab(path)
        mins = classify_minerals(f'{sub} {name}')
        records.append(dict(wvl=w, rfl=r, mineral=(next(iter(mins)) if len(mins) == 1 else None),
                            is_pure=len(mins) == 1, library='relab'))
    kept = [rec for rec in records if has_vnir_coverage(rec['wvl'], rec['rfl'])]
    return [r for r in kept if r['is_pure'] and r['mineral'] in MINERAL_KEYS]


def mean_endmember_on_grid(
    pure_refs: list[dict[str, Any]], dst_wvl: npt.ArrayLike, dst_fwhm: npt.ArrayLike | None = None
) -> np.ndarray:
    """
    Average all reference spectra of each mineral onto a sensor's band grid.

    Each reference is resampled (Gaussian convolution if ``dst_fwhm`` is
    given, else linear interpolation), water-vapor masked and
    max-normalized; the per-mineral mean is then re-normalized.

    Parameters
    ----------
    pure_refs : list of dict
        Output of ``load_reference_spectra``.
    dst_wvl : array_like
        Target band centers in nm, shape (n_bands,).
    dst_fwhm : array_like, optional
        Target band FWHM in nm, shape (n_bands,).

    Returns
    -------
    np.ndarray
        Endmembers, shape (n_minerals, n_bands), ordered as ``MINERALS``. A row
        is all NaN if no reference of that mineral has >= 10 valid bands.
    """
    dst_wvl = np.asarray(dst_wvl, dtype=float)
    members = np.full((len(MINERAL_KEYS), len(dst_wvl)), np.nan)
    for i, mineral in enumerate(MINERAL_KEYS):
        stack = []
        for rec in pure_refs:
            if rec['mineral'] != mineral:
                continue
            src_rfl = apply_bad_wavelength_mask(rec['wvl'], rec['rfl'])
            if dst_fwhm is not None:
                conv = gaussian_resample(rec['wvl'], src_rfl, dst_wvl, dst_fwhm)
            else:
                w = np.asarray(rec['wvl'], dtype=float)
                r = np.asarray(src_rfl, dtype=float)
                ok = np.isfinite(w) & np.isfinite(r)
                if ok.sum() < 2:
                    continue
                order = np.argsort(w[ok])
                conv = np.interp(dst_wvl, w[ok][order], r[ok][order], left=np.nan, right=np.nan)
            conv = apply_bad_wavelength_mask(dst_wvl, conv)
            if np.isfinite(conv).sum() >= 10:
                stack.append(max_normalize_spectrum(conv))
        if stack:
            members[i] = max_normalize_spectrum(
                apply_bad_wavelength_mask(dst_wvl, np.nanmean(np.vstack(stack), axis=0)))
    return members


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def window_for_bounds(
    transform: Affine, n_rows: int, n_cols: int, xmin: float, ymin: float, xmax: float, ymax: float
) -> tuple[int, int, int, int]:
    """
    Row/column slice bounds of a raster covering a map-coordinate box.

    Parameters
    ----------
    transform : rasterio.transform.Affine
        Raster pixel-to-map transform.
    n_rows, n_cols : int
        Raster size (used to clamp the window).
    xmin, ymin, xmax, ymax : float
        Box in the raster's CRS.

    Returns
    -------
    r0, r1, c0, c1 : int
        Half-open slice bounds ``[r0:r1, c0:c1]``.
    """
    rc = [rowcol(transform, x, y) for x, y in (
        (xmin, ymax), (xmax, ymax), (xmin, ymin), (xmax, ymin))]
    rows = [r for r, _ in rc]
    cols = [c for _, c in rc]
    r0, r1 = max(0, min(rows)), min(n_rows, max(rows) + 1)
    c0, c1 = max(0, min(cols)), min(n_cols, max(cols) + 1)
    return r0, r1, c0, c1


def inside_mask(geoms: list[dict[str, Any]], shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """
    Boolean raster that is True for pixels whose center falls inside ``geoms``.

    Parameters
    ----------
    geoms : list of dict
        GeoJSON-like geometries in the raster's CRS.
    shape : tuple of int
        Output (n_rows, n_cols).
    transform : rasterio.transform.Affine
        Raster pixel-to-map transform.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``shape``.
    """
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def map_classes(
    wvl: np.ndarray, cube: np.ndarray, mask: np.ndarray, classify: Callable[[np.ndarray], int]
) -> np.ndarray:
    """
    Apply a per-spectrum classifier to every masked pixel of a cube.

    Parameters
    ----------
    wvl : np.ndarray
        Band centers (unused; kept for a uniform call signature).
    cube : np.ndarray
        Reflectance cube, shape (n_rows, n_cols, n_bands).
    mask : np.ndarray
        Boolean AOI mask, shape (n_rows, n_cols).
    classify : callable
        Function mapping a spectrum (n_bands,) to a class id.

    Returns
    -------
    np.ndarray
        uint8 class map, shape (n_rows, n_cols); 0 outside the mask.
    """
    cls = np.zeros(mask.shape, dtype=np.uint8)
    for i, j in zip(*np.where(mask)):
        cls[i, j] = classify(cube[i, j])
    return cls


def nnls_cube(wvl: np.ndarray, cube: np.ndarray, mask: np.ndarray, members: np.ndarray) -> np.ndarray:
    """
    NNLS abundances for every masked pixel of a cube.

    Parameters
    ----------
    wvl : np.ndarray
        Band centers in nm, shape (n_bands,).
    cube : np.ndarray
        Reflectance cube, shape (n_rows, n_cols, n_bands).
    mask : np.ndarray
        Boolean AOI mask, shape (n_rows, n_cols).
    members : np.ndarray
        Endmembers, shape (n_minerals, n_bands).

    Returns
    -------
    np.ndarray
        Abundance cube, shape (n_rows, n_cols, n_minerals); NaN outside the
        mask.
    """
    abund = np.full(mask.shape + (len(MINERAL_KEYS),), np.nan, dtype=float)
    if mask.any():
        specs = cube[mask].astype(float)
        specs[np.isfinite(specs) & (specs < 0)] = 0.0
        abund[mask] = nnls_abundances(wvl, specs, members)
    return abund


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def share_rows(ids: npt.ArrayLike, method: str, source: str) -> list[dict[str, Any]]:
    """
    Per-class pixel counts and percentages for one method/source combination.

    Parameters
    ----------
    ids : array_like
        Class ids for every pixel (or pit) considered.
    method : str
        Classifier label, e.g. ``'SAM'``.
    source : str
        Data source label, e.g. ``'EMIT'``.

    Returns
    -------
    list of dict
        One row per class (Unclassified + each mineral) with ``count``,
        ``n_total``, ``n_classified``, ``pct_total`` and ``pct_classified``.
    """
    ids = np.asarray(ids, dtype=int).ravel()
    ids[ids == UNCONFIRMED_ID] = 0
    n = ids.size
    n_clf = int(np.isin(ids, list(range(1, 9))).sum())
    rows = []
    for cid, name in [(0, 'Unclassified')] + MINERALS:
        c = int((ids == cid).sum())
        rows.append(dict(
            method=method, source=source, mineral=name, class_id=cid, count=c,
            n_total=n, n_classified=n_clf,
            pct_total=100.0 * c / n if n else np.nan,
            pct_classified=(100.0 * c / n_clf if n_clf and cid else np.nan),
        ))
    return rows


def class_count_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Pivot ``share_rows`` output into one row per method/source with class counts.

    Parameters
    ----------
    rows : list of dict
        Concatenated ``share_rows`` output.

    Returns
    -------
    pd.DataFrame
        Columns ``method``, ``source``, ``n`` and one count column per class.
    """
    df = pd.DataFrame(rows)
    names = ['Unclassified'] + MINERAL_NAMES
    out = []
    for (method, source), sub in df.groupby(['method', 'source'], sort=False):
        rec = {'method': method, 'source': source, 'n': int(sub['n_total'].iloc[0])}
        for name in names:
            rec[name] = int(sub.loc[sub['mineral'] == name, 'count'].iloc[0])
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_asd() -> dict[str, pd.DataFrame]:
    """
    Load ASD field spectra from ``ASD_DIR``.

    Each ``*.txt`` file must be whitespace-delimited with a header row that
    includes ``Wavelength`` (nm) and ``Reflectance`` columns. Files with
    ``POW`` in the name are skipped.

    Returns
    -------
    dict of str to pd.DataFrame
        File stem -> table.
    """
    samples = {}
    for path in sorted(glob.glob(f'{ASD_DIR}/*.txt')):
        name = os.path.splitext(os.path.basename(path))[0]
        if 'POW' in name:
            continue
        samples[name] = pd.read_csv(path, sep=r'\s+', skipinitialspace=True)
    return samples


def load_emit() -> tuple[np.ndarray, np.ndarray, np.ndarray, xr.DataArray]:
    """
    Load the orthorectified EMIT reflectance subset.

    Expects variable ``reflectance`` with dimensions
    (``wavelengths``, ``latitude``, ``longitude``) in any order, plus
    ``wavelengths`` and ``fwhm`` variables; -9999 is treated as fill.

    Returns
    -------
    wvl : np.ndarray
        Band centers in nm, shape (n_bands,).
    fwhm : np.ndarray
        Band FWHM in nm, shape (n_bands,).
    data : np.ndarray
        Reflectance, shape (n_bands, n_lat, n_lon), fill set to NaN.
    refl : xr.DataArray
        The reflectance DataArray (fill not masked), kept for its coordinates.
    """
    ds = xr.open_dataset(EMIT_PATH)
    refl = ds['reflectance']
    if refl.dims != ('wavelengths', 'latitude', 'longitude'):
        refl = refl.transpose('wavelengths', 'latitude', 'longitude')
    data = refl.values.astype(float)
    data[data == -9999] = np.nan
    wvl = ds['wavelengths'].values.astype(float)
    fwhm = ds['fwhm'].values.astype(float)
    return wvl, fwhm, data, refl


def load_aviris() -> tuple[np.ndarray, np.ndarray, np.ndarray, CRS, Affine]:
    """
    Load the AVIRIS ENVI reflectance image, sorted by wavelength.

    Returns
    -------
    wvl : np.ndarray
        Band centers in nm, rounded to integers, ascending.
    fwhm : np.ndarray
        Band FWHM in nm from the header ``fwhm`` field, or 10 nm if absent.
    data : np.ndarray
        Raw values, shape (n_rows, n_cols, n_bands) (not yet scaled).
    crs : rasterio.crs.CRS
        Image CRS.
    transform : rasterio.transform.Affine
        Pixel-to-map transform.
    """
    img = spectral.open_image(AVIRIS_HDR)
    data = img.load()
    wvl = np.array(img.bands.centers, dtype=float)
    order = np.argsort(wvl)
    wvl = np.round(wvl[order])
    data = np.asarray(data)[:, :, order]
    crs, transform = envi_georef_from_hdr(AVIRIS_HDR)
    bw = getattr(img.bands, 'bandwidths', None)
    if bw is not None and len(bw) == len(order):
        fwhm = np.asarray(bw, dtype=float)[order]
    else:
        fwhm = np.full(len(wvl), 10.0)
    return wvl, fwhm, data, crs, transform


def load_aviris5() -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """
    Load the AVIRIS-5 reflectance NetCDF.

    Expects root variables ``easting`` and ``northing`` (UTM 12N meters) and a
    ``reflectance`` group holding ``reflectance`` with dimensions
    (band, ``northing``, ``easting``), ``wavelength`` and ``fwhm``.

    Returns
    -------
    wvl : np.ndarray
        Band centers in nm, rounded to integers.
    fwhm : np.ndarray
        Band FWHM in nm.
    refl : xr.DataArray
        Reflectance with easting/northing coordinates attached.
    """
    root = xr.open_dataset(AVIRIS5_PATH)
    refl_ds = xr.open_dataset(AVIRIS5_PATH, group='reflectance')
    wvl = np.round(refl_ds['wavelength'].values.astype(float))
    fwhm = refl_ds['fwhm'].values.astype(float)
    refl = refl_ds['reflectance'].assign_coords(
        easting=root['easting'].values, northing=root['northing'].values)
    return wvl, fwhm, refl


def aoi_mean(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    NaN-aware mean spectrum of the masked pixels.

    Parameters
    ----------
    cube : np.ndarray
        Reflectance cube, shape (n_rows, n_cols, n_bands).
    mask : np.ndarray
        Boolean AOI mask, shape (n_rows, n_cols).

    Returns
    -------
    np.ndarray
        Mean spectrum, shape (n_bands,); all NaN if the mask is empty.
    """
    if not mask.any():
        return np.full(cube.shape[-1], np.nan)
    return np.nanmean(cube[mask].astype(float), axis=0)


def check_inputs() -> None:
    """
    Fail fast if required inputs are missing from the working directory.

    Raises
    ------
    FileNotFoundError
        Listing every missing file/directory, or if no ASD file matches a
        name in ``PITS``.
    """
    required = [ASD_DIR, EMIT_PATH, AVIRIS_HDR, AVIRIS5_PATH, AOI_PATH,
                USGS_WAVELENGTH_FILE, f'{REF_DIR}/crowley', f'{REF_DIR}/relab']
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f'Missing inputs (relative to {os.getcwd()}): {", ".join(missing)}. '
            'Run from the repository root; see README.md for the expected layout.')
    found = [k for k in PITS if os.path.exists(f'{ASD_DIR}/{k}.txt')]
    if not found:
        raise FileNotFoundError(
            f'No ASD spectra in {ASD_DIR} match the PITS names (e.g. {next(iter(PITS))}.txt).')
    absent = sorted(set(PITS) - set(found))
    if absent:
        logger.warning('ASD spectra missing for %s; these pits are skipped', ', '.join(absent))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Run the full analysis and write the summary tables to ``OUT_DIR``.

    Outputs
    -------
    band_params.csv
        Decision-tree band positions per pit/source and for AOI-mean spectra
        (see ``compute_band_params``).
    band_params_by_source.csv
        Mean and std of each band parameter across pits, per source.
    class_shares.csv
        Per-class counts/percentages for SAM, decision tree and NNLS.
    nnls_mean_abundance.csv
        Mean NNLS abundance per mineral and source.
    """
    check_inputs()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load inputs -------------------------------------------------------
    asd = load_asd()
    wvl_emit, fwhm_emit, emit_cube_full, emit_refl = load_emit()
    wvl_av, fwhm_av, aviris_data, aviris_crs, aviris_transform = load_aviris()
    wvl_av5, fwhm_av5, av5_refl = load_aviris5()
    aoi = gpd.read_file(AOI_PATH)
    pure_refs = load_reference_spectra()

    # --- Pit spectra: ASD, co-located EMIT pixel, ASD convolved to EMIT ----
    to_wgs = Transformer.from_crs(UTM, 'EPSG:4326', always_xy=True)
    lat_vals = emit_refl.latitude.values
    lon_vals = emit_refl.longitude.values
    pits = {}
    wavelengths_asd = None
    resamplers = {}
    for key, (east, north) in PITS.items():
        if key not in asd:
            continue
        lon_s, lat_s = to_wgs.transform(east, north)
        # Nearest EMIT grid cell (no check that the pit lies inside the scene).
        lat_i = int(np.argmin(np.abs(lat_vals - lat_s)))
        lon_i = int(np.argmin(np.abs(lon_vals - lon_s)))
        emit_pix = emit_cube_full[:, lat_i, lon_i].astype(float)
        df = asd[key]
        asd_wvl = df['Wavelength'].values.astype(float)
        asd_rfl = df['Reflectance'].values.astype(float)
        # Put all pits on the first pit's ASD wavelength grid.
        if wavelengths_asd is None:
            wavelengths_asd = asd_wvl.copy()
        elif len(asd_wvl) != len(wavelengths_asd) or not np.allclose(asd_wvl, wavelengths_asd):
            asd_rfl = np.interp(wavelengths_asd, asd_wvl, asd_rfl, left=np.nan, right=np.nan)
            asd_wvl = wavelengths_asd
        key_b = asd_wvl.tobytes()
        if key_b not in resamplers:
            resamplers[key_b] = BandResampler(asd_wvl, wvl_emit, fwhm1=None, fwhm2=fwhm_emit)
        # Fill gaps before resampling so NaNs don't spread into EMIT bands.
        bad = ~np.isfinite(asd_rfl)
        rfl_filled = asd_rfl.copy()
        if bad.any() and (~bad).sum() > 1:
            rfl_filled[bad] = np.interp(asd_wvl[bad], asd_wvl[~bad], asd_rfl[~bad])
        asd_emit = apply_bad_wavelength_mask(wvl_emit, np.asarray(resamplers[key_b](rfl_filled), dtype=float))
        pits[key] = dict(
            east=east, north=north,
            asd_wvl=asd_wvl, asd_rfl=apply_bad_wavelength_mask(asd_wvl, asd_rfl),
            asd_native_wvl=df['Wavelength'].values.astype(float),
            asd_native_rfl=df['Reflectance'].values.astype(float),
            emit_wvl=wvl_emit, emit_rfl=emit_pix,
            asd_emit_wvl=wvl_emit, asd_emit_rfl=asd_emit,
        )

    # --- Pit spectra: co-located AVIRIS and AVIRIS-5 pixels ----------------
    n_rows, n_cols = aviris_data.shape[:2]
    to_aviris = Transformer.from_crs(UTM, aviris_crs, always_xy=True)
    av5_east = av5_refl.easting.values
    av5_north = av5_refl.northing.values
    for key, pit in pits.items():
        x, y = to_aviris.transform(pit['east'], pit['north'])
        row, col = rowcol(aviris_transform, x, y)
        if 0 <= row < n_rows and 0 <= col < n_cols:
            pix = scale_aviris_reflectance(aviris_data[row, col, :])
            pix[(wvl_av < 350) | (wvl_av > 2500)] = np.nan
            # Give the AVIRIS pit spectrum the same band gaps as its EMIT pixel.
            pix[~nearest_emit_valid_mask(wvl_av, pit['emit_wvl'], pit['emit_rfl'])] = np.nan
            pit['aviris_wvl'] = wvl_av
            pit['aviris_rfl'] = pix
        e_i = int(np.argmin(np.abs(av5_east - pit['east'])))
        n_i = int(np.argmin(np.abs(av5_north - pit['north'])))
        av5_pix = np.asarray(av5_refl.isel(northing=n_i, easting=e_i).values, dtype=float)
        av5_pix[(wvl_av5 < 350) | (wvl_av5 > 2500) | ~np.isfinite(av5_pix) | (av5_pix <= 0)] = np.nan
        av5_pix[bad_wavelength_mask(wvl_av5)] = np.nan
        pit['av5_wvl'] = wvl_av5
        pit['av5_rfl'] = av5_pix

    # --- AOI subsets and inside-polygon masks for each sensor --------------
    mine = aoi.to_crs(UTM)
    bounds = mine.total_bounds
    # AVIRIS read window: AOI plus pits, padded 30 m (plus 50 m more to the east).
    buf, east_extra = 30.0, 50.0
    pit_e = [p['east'] for p in pits.values()]
    pit_n = [p['north'] for p in pits.values()]
    xmin = min(bounds[0], min(pit_e)) - buf
    xmax = max(bounds[2], max(pit_e)) + buf + east_extra
    ymin = min(bounds[1], min(pit_n)) - buf
    ymax = max(bounds[3], max(pit_n)) + buf

    # AVIRIS-5: clip to the AOI bounding box + 30 m, then rasterize the AOI.
    clip = box(bounds[0] - buf, bounds[1] - buf, bounds[2] + buf, bounds[3] + buf)
    av5_view = (av5_refl
                .rio.set_spatial_dims(x_dim='easting', y_dim='northing', inplace=False)
                .rio.write_crs(UTM, inplace=False)
                .rio.clip([mapping(clip)], mine.crs, drop=True, all_touched=True))
    av5_cube = np.moveaxis(av5_view.values.astype(float), 0, -1)
    av5_inside = inside_mask(
        [mapping(g) for g in mine.geometry],
        (av5_view.sizes['northing'], av5_view.sizes['easting']),
        av5_view.rio.transform())

    # AVIRIS: read the window, scale to unit reflectance, rasterize the AOI.
    r0, r1, c0, c1 = window_for_bounds(
        aviris_transform, n_rows, n_cols, xmin, ymin, xmax, ymax)
    av_win = np.asarray(aviris_data[r0:r1, c0:c1, :], dtype=float)
    if np.nanmax(av_win) > 2:
        av_win = av_win / 10000.0
    av_win[:, :, (wvl_av < 350) | (wvl_av > 2500)] = np.nan
    av_inside = inside_mask(
        [mapping(g) for g in aoi.to_crs(aviris_crs).geometry],
        (n_rows, n_cols), aviris_transform)[r0:r1, c0:c1]

    # EMIT: already cropped; rasterize the AOI on the lat/lon grid.
    emit_geo = (emit_refl
                .rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=False)
                .rio.write_crs('EPSG:4326', inplace=False))
    emit_inside = inside_mask(
        [mapping(g) for g in aoi.to_crs('EPSG:4326').geometry],
        (emit_geo.sizes['latitude'], emit_geo.sizes['longitude']),
        emit_geo.rio.transform())
    emit_cube = np.moveaxis(emit_geo.values.astype(float), 0, -1)
    emit_cube[emit_cube == -9999] = np.nan

    # --- Mean reference endmembers on each sensor's band grid --------------
    members_emit = mean_endmember_on_grid(pure_refs, wvl_emit, fwhm_emit)
    members_av5 = mean_endmember_on_grid(pure_refs, wvl_av5, fwhm_av5)
    members_av = mean_endmember_on_grid(pure_refs, wvl_av, fwhm_av)
    members_asd = mean_endmember_on_grid(pure_refs, wavelengths_asd)

    # --- Band parameters: every pit x source, plus AOI-mean spectra --------
    param_rows = []
    for key, pit in pits.items():
        for source, wvl_key, rfl_key in (
            ('asd', 'asd_native_wvl', 'asd_native_rfl'),
            ('emit', 'emit_wvl', 'emit_rfl'),
            ('aviris', 'aviris_wvl', 'aviris_rfl'),
            ('aviris5', 'av5_wvl', 'av5_rfl'),
            ('asd_emit', 'asd_emit_wvl', 'asd_emit_rfl'),
        ):
            if wvl_key not in pit:
                continue
            params = band_params_from_spectrum(pit[wvl_key], pit[rfl_key])
            if not params:
                continue
            param_rows.append(dict(
                category='field', id=key, source=source,
                **{k: params.get(k, np.nan) for k in PARAM_KEYS}))
    means = {
        'EMIT AOI mean': ('emit', wvl_emit, aoi_mean(emit_cube, emit_inside)),
        'AVIRIS AOI mean': ('aviris', wvl_av, aoi_mean(av_win, av_inside)),
        'AVIRIS-5 AOI mean': ('aviris5', wvl_av5, aoi_mean(av5_cube, av5_inside)),
        'ASD to EMIT pit mean': (
            'asd_emit', wvl_emit,
            np.nanmean(np.vstack([p['asd_emit_rfl'] for p in pits.values()]), axis=0)),
    }
    for spec_id, (source, wvl, rfl) in means.items():
        params = band_params_from_spectrum(wvl, rfl)
        if params:
            param_rows.append(dict(
                category='mean', id=spec_id, source=source,
                **{k: params.get(k, np.nan) for k in PARAM_KEYS}))
    band_df = pd.DataFrame(param_rows)
    band_df.to_csv(f'{OUT_DIR}/band_params.csv', index=False, float_format='%.4f')
    print(band_df.to_string(index=False))
    field = band_df[band_df['category'] == 'field']
    summary = []
    for source, sub in field.groupby('source'):
        for key in PARAM_KEYS:
            vals = sub[key].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            summary.append(dict(source=source, parameter=key, n=int(vals.size),
                                mean=float(vals.mean()), std=std))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{OUT_DIR}/band_params_by_source.csv', index=False, float_format='%.4f')
    print(summary_df.to_string(index=False))

    # --- Decision-tree classification --------------------------------------
    def tree_pixel(wvl: np.ndarray) -> Callable[[np.ndarray], int]:
        """Build a per-pixel decision-tree classifier for one sensor."""
        def classify(spec: np.ndarray) -> int:
            spec = np.asarray(spec, dtype=float).copy()
            spec[~np.isfinite(spec) | (spec <= 0)] = np.nan
            return refined_class_from_spectrum(wvl, spec)
        return classify

    av5_tree = map_classes(wvl_av5, av5_cube, av5_inside, tree_pixel(wvl_av5))
    av_tree = map_classes(wvl_av, av_win, av_inside, tree_pixel(wvl_av))
    emit_tree = map_classes(wvl_emit, emit_cube, emit_inside, tree_pixel(wvl_emit))
    asd_emit_tree = [
        refined_class_from_spectrum(p['asd_emit_wvl'], p['asd_emit_rfl'])
        for p in pits.values()
    ]
    asd_native_tree = [
        refined_class_from_spectrum(p['asd_wvl'], p['asd_rfl'])
        for p in pits.values()
    ]
    tree_counts = []
    tree_counts += share_rows(asd_emit_tree, 'Decision tree', 'ASD-EMIT')
    tree_counts += share_rows(emit_tree[emit_inside], 'Decision tree', 'EMIT')
    tree_counts += share_rows(av_tree[av_inside], 'Decision tree', 'AVIRIS')
    tree_counts += share_rows(av5_tree[av5_inside], 'Decision tree', 'AVIRIS-5')
    print(class_count_table(tree_counts).to_string(index=False))

    # --- SAM and NNLS classification ---------------------------------------
    def sam_pixel(wvl: np.ndarray, members: np.ndarray) -> Callable[[np.ndarray], int]:
        """Build a per-pixel SAM classifier for one sensor."""
        def classify(spec: np.ndarray) -> int:
            return sam_class_from_spectrum(wvl, spec, members)
        return classify

    av5_sam = map_classes(wvl_av5, av5_cube, av5_inside, sam_pixel(wvl_av5, members_av5))
    av_sam = map_classes(wvl_av, av_win, av_inside, sam_pixel(wvl_av, members_av))
    emit_sam = map_classes(wvl_emit, emit_cube, emit_inside, sam_pixel(wvl_emit, members_emit))
    asd_sam = [sam_class_from_spectrum(p['asd_wvl'], p['asd_rfl'], members_asd) for p in pits.values()]

    av5_abund = nnls_cube(wvl_av5, av5_cube, av5_inside, members_av5)
    av_abund = nnls_cube(wvl_av, av_win, av_inside, members_av)
    emit_abund = nnls_cube(wvl_emit, emit_cube, emit_inside, members_emit)
    asd_abund = np.vstack([
        nnls_abundances(p['asd_wvl'], p['asd_rfl'], members_asd) for p in pits.values()
    ])
    av5_nnls = class_from_abund_cube(av5_abund)
    av_nnls = class_from_abund_cube(av_abund)
    emit_nnls = class_from_abund_cube(emit_abund)
    asd_nnls = [nnls_class_from_abundance(row) for row in asd_abund]

    # --- Class-share table across all methods and sources ------------------
    share = []
    share += share_rows(asd_sam, 'SAM', 'ASD')
    share += share_rows(emit_sam[emit_inside], 'SAM', 'EMIT')
    share += share_rows(av_sam[av_inside], 'SAM', 'AVIRIS')
    share += share_rows(av5_sam[av5_inside], 'SAM', 'AVIRIS-5')
    share += share_rows(asd_native_tree, 'Decision tree', 'ASD')
    share += share_rows(emit_tree[emit_inside], 'Decision tree', 'EMIT')
    share += share_rows(av_tree[av_inside], 'Decision tree', 'AVIRIS')
    share += share_rows(av5_tree[av5_inside], 'Decision tree', 'AVIRIS-5')
    share += share_rows(asd_nnls, 'NNLS', 'ASD')
    share += share_rows(emit_nnls[emit_inside], 'NNLS', 'EMIT')
    share += share_rows(av_nnls[av_inside], 'NNLS', 'AVIRIS')
    share += share_rows(av5_nnls[av5_inside], 'NNLS', 'AVIRIS-5')
    share_df = pd.DataFrame(share)
    share_df.to_csv(f'{OUT_DIR}/class_shares.csv', index=False, float_format='%.4f')
    print(class_count_table(share).to_string(index=False))

    # --- Mean NNLS abundance per mineral and source ------------------------
    abund_rows = []
    for source, abund, mask in (
        ('ASD', asd_abund, None),
        ('EMIT', emit_abund, emit_inside),
        ('AVIRIS', av_abund, av_inside),
        ('AVIRIS-5', av5_abund, av5_inside),
    ):
        A = abund if mask is None else abund[mask]
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        A = A[np.isfinite(A).all(axis=1)]
        mean = np.mean(A, axis=0) if len(A) else np.full(len(MINERAL_KEYS), np.nan)
        for k, name in enumerate(MINERAL_NAMES):
            abund_rows.append(dict(source=source, mineral=name, n=len(A), mean_abundance=mean[k]))
    abund_df = pd.DataFrame(abund_rows)
    abund_df.to_csv(f'{OUT_DIR}/nnls_mean_abundance.csv', index=False, float_format='%.4f')
    print(abund_df.pivot(index='mineral', columns='source', values='mean_abundance')
          .reindex(MINERAL_NAMES)[['ASD', 'EMIT', 'AVIRIS', 'AVIRIS-5']].to_string())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
