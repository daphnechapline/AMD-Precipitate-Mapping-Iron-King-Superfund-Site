import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import spectral
from spectral import BandResampler
from spectral.algorithms.algorithms import spectral_angles
from scipy import integrate
from scipy.signal import savgol_filter
from pysptools.spectro import SpectrumConvexHullQuotient
from pysptools.abundance_maps.amaps import NNLS
from shapely.geometry import mapping, box
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol
from rasterio.features import geometry_mask
from pyproj import Transformer

warnings.simplefilter('ignore')

DATA = 'data'
ASD_DIR = f'{DATA}/asd'
EMIT_PATH = f'{DATA}/emit_ortho_crop.nc'
AVIRIS_HDR = f'{DATA}/aviris_rfl.hdr'
AVIRIS5_PATH = f'{DATA}/aviris5_rfl.nc'
AOI_PATH = f'{DATA}/IronKing.kml'
REF_DIR = 'ref_spectra'
OUT_DIR = 'outputs'

BAD_WAVELENGTH_WINDOWS = [(1275, 1475), (1750, 1975)]
APPLY_SG_SMOOTH = False
SAM_MAX_ANGLE_RAD = 0.2
NOISE_THRESHOLD = 2e-4
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
UNCONFIRMED_ID = 9

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

SG = {
    'emit': (7, 3),
    'aviris': (13, 3),
    'aviris5': (13, 3),
}

PARAM_KEYS = [
    'b500_center', 'b500_depth', 'b500_area',
    'absorption_edge', 'vis_peak',
    'b900_center', 'b900_depth', 'b900_area', 'WAC', 'b900_start',
    'b2265_center', 'b2265_depth', 'b2265_area',
    'cr_500_600', 'cr_450_550', 'cr_1200_1275', 'cr_600_750',
]


def bad_wavelength_mask(wavelengths):
    wvl = np.asarray(wavelengths, dtype=float)
    bad = np.zeros(len(wvl), dtype=bool)
    for lo, hi in BAD_WAVELENGTH_WINDOWS:
        bad |= (wvl >= lo) & (wvl <= hi)
    return bad


def apply_bad_wavelength_mask(wavelengths, reflectance):
    out = np.asarray(reflectance, dtype=float).copy().reshape(-1)
    out[bad_wavelength_mask(wavelengths)] = np.nan
    return out


def max_normalize_spectrum(rfl):
    out = np.asarray(rfl, dtype=float).copy()
    finite = np.isfinite(out) & (out > 0)
    if not finite.any():
        return out
    peak = float(np.max(out[finite]))
    if peak > 0:
        out = out / peak
    return out


def envi_georef_from_hdr(hdr_path):
    text = open(hdr_path).read()
    m = re.search(r'map info\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)
    if not m:
        raise ValueError(f'No map info in ENVI header: {hdr_path}')
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
    west = map_x - ps_x * ((ref_px - 1.0) + 0.5)
    north = map_y + ps_y * ((ref_py - 1.0) + 0.5)
    return crs, Affine(ps_x, 0.0, west, 0.0, -ps_y, north)


def scale_aviris_reflectance(pix):
    pix = np.asarray(pix, dtype=float).copy().ravel()
    if pix.size and np.nanmax(pix) > 2:
        pix /= 10000.0
    pix[pix < -0.1] = np.nan
    return pix


def nearest_emit_valid_mask(src_wvl, emit_wvl, emit_rfl):
    emit_wvl = np.asarray(emit_wvl, dtype=float)
    src_wvl = np.asarray(src_wvl, dtype=float)
    emit_ok = np.isfinite(emit_rfl) & (np.asarray(emit_rfl, dtype=float) > 0)
    idx = np.abs(emit_wvl[:, None] - src_wvl[None, :]).argmin(axis=0)
    return emit_ok[idx]


def gaussian_resample(src_wvl, src_rfl, dst_wvl, dst_fwhm):
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
    sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
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


def continuum_removal(refl_values, wavelengths):
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


def vis_cr_band_center(wvl, cr, wmin, wmax, min_depth=0.01):
    m = (wvl >= wmin) & (wvl <= wmax) & np.isfinite(wvl) & np.isfinite(cr)
    if m.sum() < 3:
        return np.nan
    w_seg, cr_seg = wvl[m], cr[m]
    i = int(np.argmin(cr_seg))
    if (1.0 - float(cr_seg[i])) <= min_depth:
        return np.nan
    return float(w_seg[i])


def first_cr_unity(wavelengths, cr, wmin, wmax, descending=False):
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


def compute_band_params(cr, wavelengths, reflectance=None, b900_max=1300):
    wavelengths = np.asarray(wavelengths, dtype=float)
    cr = np.asarray(cr, dtype=float)
    valid = ~np.isnan(cr)
    wvl_v, cr_v = wavelengths[valid], cr[valid]
    if len(cr_v) < 11:
        return None

    def store_band(result, prefix, wvl_seg, cr_seg, wmin, wmax, clip=True):
        result[f'{prefix}_center'] = np.nan
        result[f'{prefix}_depth'] = np.nan
        result[f'{prefix}_area'] = np.nan
        if len(wvl_seg) < 3:
            return False
        min_idx = np.argmin(cr_seg)
        depth = 1 - cr_seg[min_idx]
        if depth <= 0.01:
            return False
        band_signal = np.clip(1 - cr_seg, 0, None) if clip else (1 - cr_seg)
        result[f'{prefix}_center'] = wvl_seg[min_idx]
        result[f'{prefix}_depth'] = depth
        result[f'{prefix}_area'] = float(integrate.simpson(band_signal, wvl_seg))
        return True

    b900_min = 720
    bm = (wvl_v >= b900_min) & (wvl_v <= b900_max)
    wav_seg, cr_seg = wvl_v[bm], cr_v[bm]
    if len(wav_seg) < 3:
        return None
    result = {}
    b900_ok = store_band(result, 'b900', wav_seg, cr_seg, b900_min, b900_max, clip=False)
    if b900_ok:
        halfarea = result['b900_area'] / 2
        smaller_wav, smaller_int = wav_seg[0], result['b900_area']
        wac = wav_seg[-1]
        for i, wav in enumerate(wav_seg):
            partial = integrate.simpson(1 - cr_seg[i:], wav_seg[i:])
            if partial < halfarea:
                wac = smaller_wav + (halfarea - smaller_int) / (partial - smaller_int) * (wav - smaller_wav)
                break
            smaller_wav, smaller_int = wav, partial
        result['WAC'] = wac
        result['b900_start'] = first_cr_unity(wvl_v, cr_v, 625, 800, descending=True)
    else:
        result['WAC'] = np.nan
        result['b900_start'] = np.nan
    store_band(result, 'b2265', wvl_v[(wvl_v >= 2230) & (wvl_v <= 2290)],
               cr_v[(wvl_v >= 2230) & (wvl_v <= 2290)], 2230, 2290)
    result['cr_500_600'] = vis_cr_band_center(wvl_v, cr_v, 500, 600)
    result['cr_450_550'] = vis_cr_band_center(wvl_v, cr_v, 450, 550)
    result['cr_1200_1275'] = vis_cr_band_center(wvl_v, cr_v, 1200, 1275)
    result['cr_600_750'] = vis_cr_band_center(wvl_v, cr_v, 600, 750)
    result['absorption_edge'] = first_cr_unity(wvl_v, cr_v, 500, 720)
    result['vis_peak'] = np.nan
    if reflectance is not None:
        reflectance = np.asarray(reflectance, dtype=float)
        vis = (wavelengths > 400) & (wavelengths < 800) & np.isfinite(reflectance)
        if vis.any():
            result['vis_peak'] = wavelengths[vis][np.argmax(reflectance[vis])]
        b500 = (wavelengths >= 380) & (wavelengths <= 700)
        store_band(result, 'b500', wavelengths[b500], cr[b500], 380, 700)
    return result


def band_params_from_spectrum(wvl, rfl, b900_max=1300):
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    if np.isfinite(rfl).sum() < 11:
        return None
    cr = continuum_removal(rfl, wvl)
    params = compute_band_params(cr, wvl, reflectance=rfl, b900_max=b900_max)
    if not params:
        return None
    params['cr'] = cr
    params['wavelengths'] = wvl
    return params


def smooth_spectrum(vals, window, polyorder):
    out = np.asarray(vals, dtype=float).copy()
    if not APPLY_SG_SMOOTH:
        return out
    mask = ~np.isnan(out)
    if mask.sum() >= window:
        out[mask] = savgol_filter(out[mask], window, polyorder)
    return out


def smooth_pixel_spectrum(wvl, rfl, window, polyord):
    wvl = np.asarray(wvl, dtype=float)
    rfl = apply_bad_wavelength_mask(wvl, rfl)
    valid = np.isfinite(rfl) & (rfl > 0)
    if valid.sum() < max(window, 11):
        return None
    out = rfl.copy()
    good, bad = np.where(valid)[0], np.where(~valid)[0]
    if len(good) > 1 and len(bad):
        out[bad] = np.interp(wvl[bad], wvl[good], rfl[good])
    out = smooth_spectrum(out, window, polyord)
    out[~valid] = np.nan
    return out


def cr_absorption_depth(cr, wvl, wmin, wmax, min_points=2):
    wvl = np.asarray(wvl, dtype=float)
    cr = np.asarray(cr, dtype=float)
    m = (wvl >= wmin) & (wvl <= wmax) & np.isfinite(cr)
    if m.sum() < min_points:
        return 0.0, np.nan
    i = int(np.argmin(cr[m]))
    return float(1.0 - cr[m][i]), float(wvl[m][i])


def refined_mineral_class(params):
    b900 = params.get('b900_center', np.nan)
    b900_depth = params.get('b900_depth', np.nan)
    if not np.isfinite(b900) or b900_depth <= 0.01:
        return 0
    cr, wvl = params['cr'], params['wavelengths']
    b900_start = params.get('b900_start', np.nan)
    if 780 <= b900 <= 800:
        d_coq, c_coq = cr_absorption_depth(cr, wvl, 500, 600)
        d_rho, c_rho = cr_absorption_depth(cr, wvl, 450, 550)
        has_coq = np.isfinite(c_coq) and d_coq > 0.01 and 545 < c_coq <= 595
        has_rho = np.isfinite(c_rho) and d_rho > 0.01 and 500 <= c_rho <= 545
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
        d2265 = params.get('b2265_depth', np.nan)
        if np.isfinite(c2265) and np.isfinite(d2265) and d2265 > 0.01 and 2255 <= c2265 <= 2275:
            return 5
        return 0
    if 940 < b900 < 980:
        d_goe, c_goe = cr_absorption_depth(cr, wvl, 600, 750)
        if np.isfinite(c_goe) and d_goe > 0.01 and 625 <= c_goe <= 715:
            return 6
        if np.isfinite(b900_start) and 710 <= b900_start <= 755:
            return 7
        return 0
    if 980 <= b900 <= 1030 and np.isfinite(b900_start) and 750 <= b900_start <= 785:
        return 8
    return 0


def refined_class_from_spectrum(wvl, rfl, window, polyord):
    sm = smooth_pixel_spectrum(wvl, rfl, window, polyord)
    if sm is None:
        return 0
    params = band_params_from_spectrum(wvl, sm)
    if not params:
        return 0
    return refined_mineral_class(params)


def prep_spectrum_for_sam(wvl, rfl):
    rfl = apply_bad_wavelength_mask(wvl, rfl)
    rfl = np.asarray(rfl, dtype=float)
    rfl[~(np.isfinite(rfl) & (rfl > 0))] = np.nan
    return max_normalize_spectrum(rfl)


def sam_class_from_spectrum(wvl, rfl, members):
    pix = prep_spectrum_for_sam(wvl, rfl)
    members = np.asarray(members, dtype=float).copy()
    members[:, bad_wavelength_mask(wvl)] = np.nan
    valid = np.isfinite(pix) & np.all(np.isfinite(members), axis=0)
    if valid.sum() < 10:
        return 0
    pix_n = max_normalize_spectrum(pix[valid])
    mem = np.vstack([max_normalize_spectrum(row) for row in members[:, valid]])
    ang = np.asarray(spectral_angles(pix_n.reshape(1, 1, -1), mem), dtype=float).ravel()
    if not np.all(np.isfinite(ang)):
        return 0
    k = int(np.argmin(ang))
    if SAM_MAX_ANGLE_RAD is not None and ang[k] > SAM_MAX_ANGLE_RAD:
        return 0
    return k + 1


def prep_spectrum_for_unmix(wvl, rfl):
    rfl = apply_bad_wavelength_mask(wvl, rfl)
    rfl = np.asarray(rfl, dtype=float)
    rfl[np.isfinite(rfl) & (rfl < 0)] = 0.0
    rfl[~np.isfinite(rfl)] = np.nan
    return max_normalize_spectrum(rfl)


def nnls_abundances(wvl, specs, members, min_bands=10):
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


def nnls_class_from_abundance(abund):
    abund = np.asarray(abund, dtype=float).ravel()
    if abund.size != len(MINERAL_KEYS) or not np.any(np.isfinite(abund)):
        return 0
    a = np.where(np.isfinite(abund), abund, -np.inf)
    k = int(np.argmax(a))
    if not np.isfinite(abund[k]) or abund[k] <= 0:
        return 0
    return k + 1


def class_from_abund_cube(abund):
    cls = np.zeros(abund.shape[:2], dtype=np.uint8)
    finite = np.any(np.isfinite(abund), axis=2)
    if finite.any():
        a = np.where(np.isfinite(abund), abund, -np.inf)
        k = np.argmax(a, axis=2)
        good = finite & (np.max(a, axis=2) > 0)
        cls[good] = (k[good] + 1).astype(np.uint8)
    return cls


def noise_index(wvl, rfl, lo=500.0, hi=1350.0):
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    m = (wvl >= lo) & (wvl <= hi) & np.isfinite(rfl)
    r = rfl[m]
    if r.size < 20:
        return np.nan
    d2 = r[2:] - 2.0 * r[1:-1] + r[:-2]
    return float(np.median(np.abs(d2)) / np.nanmedian(r))


def has_vnir_coverage(wvl, rfl, lo=750.0, hi=1000.0, min_points=10):
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    return int(((wvl >= lo) & (wvl <= hi) & np.isfinite(rfl)).sum()) >= min_points


def classify_minerals(text):
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


def read_crowley(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['w', 'r'])
    return df['w'].values * 1000.0, df['r'].values.astype(float)


def read_usgs(path, s07_wvl):
    rfl = pd.read_csv(path, skiprows=1, header=None)[0].values.astype(float)
    rfl[rfl < -1] = np.nan
    if len(rfl) != len(s07_wvl):
        return None, None
    return s07_wvl.copy(), rfl


def read_relab(path):
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


def usgs_header_name(path):
    with open(path) as fh:
        header = fh.readline().strip()
    if ':' in header:
        return header.split(':', 1)[-1].strip()
    return os.path.splitext(os.path.basename(path))[0]


def relab_xml_name(tab_path):
    xml = tab_path[:-4] + '.xml'
    sub = name = ''
    if os.path.exists(xml):
        txt = open(xml).read()
        m = re.search(r'<speclib:mineral_subtype>([^<]*)', txt)
        sub = m.group(1).strip() if m else ''
        m = re.search(r'<speclib:specimen_name>([^<]*)', txt)
        name = m.group(1).strip() if m else ''
    return sub, name


def load_reference_spectra():
    s07 = pd.read_csv(
        f'{REF_DIR}/usgs_minerals/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
        skiprows=1, header=None)[0].values.astype(float) * 1000.0
    records = []
    for path in sorted(glob.glob(f'{REF_DIR}/crowley/*.txt')):
        raw = os.path.splitext(os.path.basename(path))[0]
        w, r = read_crowley(path)
        mins = classify_minerals(raw)
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
    kept = []
    for rec in records:
        noisy = (rec['library'] == 'usgs'
                 and np.isfinite(noise_index(rec['wvl'], rec['rfl']))
                 and noise_index(rec['wvl'], rec['rfl']) > NOISE_THRESHOLD)
        if has_vnir_coverage(rec['wvl'], rec['rfl']) and not noisy:
            kept.append(rec)
    return [r for r in kept if r['is_pure'] and r['mineral'] in MINERAL_KEYS]


def mean_endmember_on_grid(pure_refs, dst_wvl, dst_fwhm=None):
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


def window_for_bounds(transform, n_rows, n_cols, xmin, ymin, xmax, ymax):
    rc = [rowcol(transform, x, y) for x, y in (
        (xmin, ymax), (xmax, ymax), (xmin, ymin), (xmax, ymin))]
    rows = [r for r, _ in rc]
    cols = [c for _, c in rc]
    r0, r1 = max(0, min(rows)), min(n_rows, max(rows) + 1)
    c0, c1 = max(0, min(cols)), min(n_cols, max(cols) + 1)
    return r0, r1, c0, c1


def inside_mask(geoms, shape, transform):
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def map_classes(wvl, cube, mask, classify):
    cls = np.zeros(mask.shape, dtype=np.uint8)
    for i, j in zip(*np.where(mask)):
        cls[i, j] = classify(cube[i, j])
    return cls


def nnls_cube(wvl, cube, mask, members):
    abund = np.full(mask.shape + (len(MINERAL_KEYS),), np.nan, dtype=float)
    if mask.any():
        specs = cube[mask].astype(float)
        specs[np.isfinite(specs) & (specs < 0)] = 0.0
        abund[mask] = nnls_abundances(wvl, specs, members)
    return abund


def share_rows(ids, method, source):
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


def class_count_table(rows):
    df = pd.DataFrame(rows)
    names = ['Unclassified'] + MINERAL_NAMES
    out = []
    for (method, source), sub in df.groupby(['method', 'source'], sort=False):
        rec = {'method': method, 'source': source, 'n': int(sub['n_total'].iloc[0])}
        for name in names:
            rec[name] = int(sub.loc[sub['mineral'] == name, 'count'].iloc[0])
        out.append(rec)
    return pd.DataFrame(out)


def jarosite_window_counts(wvl, cube, mask, class_grid, window, polyord):
    n_pass = n_out = n_shallow = n_out_short = n_out_long = n_shallow_in_win = 0
    for i, j in zip(*np.where(mask)):
        spec = np.asarray(cube[i, j], dtype=float).copy()
        spec[~np.isfinite(spec) | (spec <= 0)] = np.nan
        sm = smooth_pixel_spectrum(wvl, spec, window, polyord)
        if sm is None:
            continue
        params = band_params_from_spectrum(wvl, sm)
        if not params:
            continue
        b900 = params.get('b900_center', np.nan)
        d900 = params.get('b900_depth', np.nan)
        if not (np.isfinite(b900) and d900 > 0.01 and 910 <= b900 <= 940):
            continue
        if refined_mineral_class(params) == 5:
            n_pass += 1
            continue
        d_sw, c_sw = cr_absorption_depth(params['cr'], params['wavelengths'], 2230, 2290)
        if np.isfinite(c_sw) and d_sw > 0.01 and not (2255 <= c_sw <= 2275):
            n_out += 1
            if c_sw < 2255:
                n_out_short += 1
            else:
                n_out_long += 1
        else:
            n_shallow += 1
            if np.isfinite(c_sw) and 2255 <= c_sw <= 2275:
                n_shallow_in_win += 1
    n_aoi = int(mask.sum())
    n_unclass = int(np.isin(class_grid[mask], [0, UNCONFIRMED_ID]).sum())
    return dict(
        n_aoi=n_aoi, n_unclass=n_unclass,
        n_window=n_pass + n_out + n_shallow, n_pass=n_pass,
        n_reject=n_out + n_shallow, n_out=n_out,
        n_out_short=n_out_short, n_out_long=n_out_long,
        n_shallow=n_shallow, n_shallow_in_win=n_shallow_in_win,
    )


def load_asd():
    samples = {}
    for path in sorted(glob.glob(f'{ASD_DIR}/*.txt')):
        name = os.path.splitext(os.path.basename(path))[0]
        if 'POW' in name:
            continue
        samples[name] = pd.read_csv(path, sep=r'\s+', skipinitialspace=True)
    return samples


def load_emit():
    ds = xr.open_dataset(EMIT_PATH)
    refl = ds['reflectance']
    if refl.dims != ('wavelengths', 'latitude', 'longitude'):
        refl = refl.transpose('wavelengths', 'latitude', 'longitude')
    data = refl.values.astype(float)
    data[data == -9999] = np.nan
    wvl = ds['wavelengths'].values.astype(float)
    fwhm = ds['fwhm'].values.astype(float)
    return wvl, fwhm, data, refl


def load_aviris():
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


def load_aviris5():
    root = xr.open_dataset(AVIRIS5_PATH)
    refl_ds = xr.open_dataset(AVIRIS5_PATH, group='reflectance')
    wvl = np.round(refl_ds['wavelength'].values.astype(float))
    fwhm = refl_ds['fwhm'].values.astype(float)
    refl = refl_ds['reflectance'].assign_coords(
        easting=root['easting'].values, northing=root['northing'].values)
    return wvl, fwhm, refl


def aoi_mean(cube, mask):
    if not mask.any():
        return np.full(cube.shape[-1], np.nan)
    return np.nanmean(cube[mask].astype(float), axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    asd = load_asd()
    wvl_emit, fwhm_emit, emit_cube_full, emit_refl = load_emit()
    wvl_av, fwhm_av, aviris_data, aviris_crs, aviris_transform = load_aviris()
    wvl_av5, fwhm_av5, av5_refl = load_aviris5()
    aoi = gpd.read_file(AOI_PATH)
    pure_refs = load_reference_spectra()

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
        lat_i = int(np.argmin(np.abs(lat_vals - lat_s)))
        lon_i = int(np.argmin(np.abs(lon_vals - lon_s)))
        emit_pix = emit_cube_full[:, lat_i, lon_i].astype(float)
        df = asd[key]
        asd_wvl = df['Wavelength'].values.astype(float)
        asd_rfl = df['Reflectance'].values.astype(float)
        if wavelengths_asd is None:
            wavelengths_asd = asd_wvl.copy()
        elif len(asd_wvl) != len(wavelengths_asd) or not np.allclose(asd_wvl, wavelengths_asd):
            asd_rfl = np.interp(wavelengths_asd, asd_wvl, asd_rfl, left=np.nan, right=np.nan)
            asd_wvl = wavelengths_asd
        key_b = asd_wvl.tobytes()
        if key_b not in resamplers:
            resamplers[key_b] = BandResampler(asd_wvl, wvl_emit, fwhm1=None, fwhm2=fwhm_emit)
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

    mine = aoi.to_crs(UTM)
    bounds = mine.total_bounds
    buf, east_extra = 30.0, 50.0
    pit_e = [p['east'] for p in pits.values()]
    pit_n = [p['north'] for p in pits.values()]
    xmin = min(bounds[0], min(pit_e)) - buf
    xmax = max(bounds[2], max(pit_e)) + buf + east_extra
    ymin = min(bounds[1], min(pit_n)) - buf
    ymax = max(bounds[3], max(pit_n)) + buf

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

    r0, r1, c0, c1 = window_for_bounds(
        aviris_transform, n_rows, n_cols, xmin, ymin, xmax, ymax)
    av_win = np.asarray(aviris_data[r0:r1, c0:c1, :], dtype=float)
    if np.nanmax(av_win) > 2:
        av_win = av_win / 10000.0
    av_win[:, :, (wvl_av < 350) | (wvl_av > 2500)] = np.nan
    av_inside = inside_mask(
        [mapping(g) for g in aoi.to_crs(aviris_crs).geometry],
        (n_rows, n_cols), aviris_transform)[r0:r1, c0:c1]

    emit_geo = (emit_refl
                .rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=False)
                .rio.write_crs('EPSG:4326', inplace=False))
    emit_inside = inside_mask(
        [mapping(g) for g in aoi.to_crs('EPSG:4326').geometry],
        (emit_geo.sizes['latitude'], emit_geo.sizes['longitude']),
        emit_geo.rio.transform())
    emit_cube = np.moveaxis(emit_geo.values.astype(float), 0, -1)
    emit_cube[emit_cube == -9999] = np.nan

    members_emit = mean_endmember_on_grid(pure_refs, wvl_emit, fwhm_emit)
    members_av5 = mean_endmember_on_grid(pure_refs, wvl_av5, fwhm_av5)
    members_av = mean_endmember_on_grid(pure_refs, wvl_av, fwhm_av)
    members_asd = mean_endmember_on_grid(pure_refs, wavelengths_asd)

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
            params = band_params_from_spectrum(pit[wvl_key], pit[rfl_key], b900_max=1200)
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
        params = band_params_from_spectrum(wvl, rfl, b900_max=1200)
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

    emit_w, emit_p = SG['emit']
    av_w, av_p = SG['aviris']
    av5_w, av5_p = SG['aviris5']

    def tree_pixel(wvl, window, poly):
        def classify(spec):
            spec = np.asarray(spec, dtype=float).copy()
            spec[~np.isfinite(spec) | (spec <= 0)] = np.nan
            return refined_class_from_spectrum(wvl, spec, window, poly)
        return classify

    av5_tree = map_classes(wvl_av5, av5_cube, av5_inside, tree_pixel(wvl_av5, av5_w, av5_p))
    av_tree = map_classes(wvl_av, av_win, av_inside, tree_pixel(wvl_av, av_w, av_p))
    emit_tree = map_classes(wvl_emit, emit_cube, emit_inside, tree_pixel(wvl_emit, emit_w, emit_p))
    asd_emit_tree = [
        refined_class_from_spectrum(p['asd_emit_wvl'], p['asd_emit_rfl'], emit_w, emit_p)
        for p in pits.values()
    ]
    asd_native_tree = [
        refined_class_from_spectrum(p['asd_wvl'], p['asd_rfl'], emit_w, emit_p)
        for p in pits.values()
    ]
    tree_counts = []
    tree_counts += share_rows(asd_emit_tree, 'Decision tree', 'ASD-EMIT')
    tree_counts += share_rows(emit_tree[emit_inside], 'Decision tree', 'EMIT')
    tree_counts += share_rows(av_tree[av_inside], 'Decision tree', 'AVIRIS')
    tree_counts += share_rows(av5_tree[av5_inside], 'Decision tree', 'AVIRIS-5')
    print(class_count_table(tree_counts).to_string(index=False))

    jaro_rows = []
    for source, wvl, cube, mask, grid, window, poly in (
        ('EMIT', wvl_emit, emit_cube, emit_inside, emit_tree, emit_w, emit_p),
        ('AVIRIS', wvl_av, av_win, av_inside, av_tree, av_w, av_p),
        ('AVIRIS-5', wvl_av5, av5_cube, av5_inside, av5_tree, av5_w, av5_p),
    ):
        rec = jarosite_window_counts(wvl, cube, mask, grid, window, poly)
        rec['source'] = source
        jaro_rows.append(rec)
    jaro_df = pd.DataFrame(jaro_rows).set_index('source')
    jaro_df.to_csv(f'{OUT_DIR}/jarosite_window.csv')
    print(jaro_df.to_string())

    def sam_pixel(wvl, members):
        def classify(spec):
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
    main()
