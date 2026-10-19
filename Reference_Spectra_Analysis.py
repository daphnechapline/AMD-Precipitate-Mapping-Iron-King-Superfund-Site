"""
Summarize diagnostic band positions of pure reference spectra for the eight
target AMD minerals.

Loads the Crowley, USGS splib07 and RELAB libraries from ``ref_spectra/``,
continuum-removes each pure spectrum with VNIR coverage, and prints the mean,
+/-1 sigma and +/-2 sigma range of the feature that the decision tree in
``Iron_King_Analysis.refined_mineral_class`` uses for each mineral:

- NIR (~900 nm) band center, all minerals
- 500-600 nm band center (coquimbite) and 450-550 nm band center (rhomboclase)
- NIR band edge/shoulder (ferricopiapite, hematite, schwertmannite,
  ferrihydrite)
- 2230-2290 nm band center (jarosite)
- 600-750 nm band center (goethite)

Run from the repository root:  ``python Reference_Spectra_Analysis.py``.
The script executes at import time and only prints a table (no files written).
"""
import os, re, glob, warnings
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from pysptools.spectro import SpectrumConvexHullQuotient
warnings.simplefilter('ignore')

REF_DIR = 'ref_spectra'
TARGET_MINERALS = ['coquimbite', 'rhomboclase', 'ferricopiapite', 'hematite',
                   'jarosite', 'schwertmannite', 'goethite', 'ferrihydrite']
USGS_WAVELENGTH_FILE = (
    f'{REF_DIR}/usgs_minerals/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')

# Fail fast with a clear message if the reference library is not in place.
_missing = [p for p in (f'{REF_DIR}/crowley', f'{REF_DIR}/relab', USGS_WAVELENGTH_FILE)
            if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError(
        f'Missing reference inputs (relative to {os.getcwd()}): {", ".join(_missing)}. '
        'Run from the repository root; see README.md for the expected layout.')

# NOTE (review): this script and Iron_King_Analysis.py prepare the library
# differently. Here, Crowley "paracoquimbite" is counted as coquimbite, and
# continuum removal does not mask the water-vapor windows or non-positive
# values. The decision-tree windows are therefore derived from slightly
# different spectra than those used for the SAM/NNLS endmembers. USGS spectra
# with a noise index above 2e-4 were removed from the library by hand.


def continuum_removal(refl: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Divide a spectrum by its upper convex hull (non-NaN bands only).

    Parameters
    ----------
    refl : np.ndarray
        Reflectance, shape (n_bands,).
    wavelengths : np.ndarray
        Wavelengths in nm, shape (n_bands,).

    Returns
    -------
    np.ndarray
        Continuum-removed spectrum, NaN where ``refl`` is NaN.
    """
    mask = ~np.isnan(refl)
    chq = SpectrumConvexHullQuotient(refl[mask].tolist(), wavelengths[mask].tolist())
    cr = np.full(len(wavelengths), np.nan)
    cr[mask] = chq.get_continuum_removed_spectrum()
    return cr


def band_center(wvl: np.ndarray, cr: np.ndarray, lo: float, hi: float, min_depth: float = 0.01) -> float:
    """
    Wavelength of the continuum-removed minimum in ``[lo, hi]`` nm.

    Parameters
    ----------
    wvl : np.ndarray
        Wavelengths in nm.
    cr : np.ndarray
        Continuum-removed spectrum.
    lo, hi : float
        Inclusive window in nm.
    min_depth : float, optional
        Minimum depth (1 - CR) required.

    Returns
    -------
    float
        Band center in nm, or NaN if fewer than 3 valid points or too shallow.
    """
    m = (wvl >= lo) & (wvl <= hi) & np.isfinite(wvl) & np.isfinite(cr)
    if m.sum() < 3:
        return np.nan
    w, c = wvl[m], cr[m]
    i = int(np.argmin(c))
    if 1.0 - float(c[i]) <= min_depth:
        return np.nan
    return float(w[i])


def band_edge(wvl: np.ndarray, cr: np.ndarray) -> float:
    """
    Short-wavelength shoulder of the NIR band: first CR = 1 scanning 800 -> 625 nm.

    Matches ``b900_start`` in ``Iron_King_Analysis.compute_band_params``.

    Parameters
    ----------
    wvl : np.ndarray
        Wavelengths in nm.
    cr : np.ndarray
        Continuum-removed spectrum.

    Returns
    -------
    float
        Shoulder wavelength in nm, or NaN if none is found.
    """
    m = (wvl >= 625) & (wvl <= 800)
    w, c = wvl[m], cr[m]
    if w.size < 2:
        return np.nan
    order = np.argsort(-w)
    w, c = w[order], c[order]
    good = np.isfinite(c)
    w, c = w[good], c[good]
    hit = np.where(c >= 1.0)[0]
    return float(w[hit[0]]) if hit.size else np.nan


def spectrum_params(wvl: npt.ArrayLike, rfl: npt.ArrayLike) -> dict[str, Any] | None:
    """
    Continuum-remove a reference spectrum and extract its diagnostic positions.

    Parameters
    ----------
    wvl : array_like
        Wavelengths in nm.
    rfl : array_like
        Reflectance.

    Returns
    -------
    dict or None
        ``b900_center`` (720-1275 nm), ``b900_start`` (only if a NIR band was
        found), ``b2265_center`` (2230-2290 nm), plus ``wavelengths`` and
        ``cr``. None if there are too few valid points or the hull fit fails.
    """
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    if np.isfinite(rfl).sum() < 11:
        return None
    try:
        cr = continuum_removal(rfl, wvl)
    except (IndexError, ValueError):
        return None
    valid = ~np.isnan(cr)
    if valid.sum() < 11 or ((wvl[valid] >= 720) & (wvl[valid] <= 1275)).sum() < 3:
        return None
    center = band_center(wvl, cr, 720, 1275)
    return dict(
        b900_center=center,
        b900_start=band_edge(wvl, cr) if np.isfinite(center) else np.nan,
        b2265_center=band_center(wvl, cr, 2230, 2290),
        wavelengths=wvl, cr=cr,
    )


def mean_std(vals: npt.ArrayLike) -> tuple[float, float, int]:
    """
    Mean, sample standard deviation and count of the finite values.

    Parameters
    ----------
    vals : array_like
        Values (NaNs ignored).

    Returns
    -------
    mean, std : float
        NaN if no finite values; std is 0 for a single value.
    n : int
        Number of finite values.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return float(vals.mean()), std, int(vals.size)


def classify_minerals(text: str) -> set[str]:
    """
    Identify target minerals named in a label (case-insensitive substring match).

    Paracoquimbite is excluded from coquimbite here; ``add_record`` re-labels
    the Crowley paracoquimbite spectrum explicitly.

    Parameters
    ----------
    text : str
        File name, header line or metadata text.

    Returns
    -------
    set of str
        Mineral keys found; a single-element set marks a pure reference.
    """
    t = text.lower()
    found = set()
    if 'ferricopiapite' in t: found.add('ferricopiapite')
    if 'schwertmannite' in t: found.add('schwertmannite')
    if 'ferrihydrite' in t: found.add('ferrihydrite')
    if 'rhomboclase' in t: found.add('rhomboclase')
    if 'coquimbite' in t and 'paracoquimbite' not in t: found.add('coquimbite')
    if 'jarosite' in t: found.add('jarosite')
    if 'goethite' in t: found.add('goethite')
    if 'hematite' in t: found.add('hematite')
    return found


def has_vnir(wvl: npt.ArrayLike, rfl: npt.ArrayLike, lo: float = 750.0, hi: float = 1000.0,
             min_points: int = 10) -> bool:
    """
    True if at least ``min_points`` finite samples lie in ``[lo, hi]`` nm.

    Parameters
    ----------
    wvl : array_like
        Wavelengths in nm.
    rfl : array_like
        Reflectance.
    lo, hi : float, optional
        Window in nm.
    min_points : int, optional
        Required number of finite samples.

    Returns
    -------
    bool
    """
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    return int(((wvl >= lo) & (wvl <= hi) & np.isfinite(rfl)).sum()) >= min_points


def read_crowley(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a Crowley spectrum: two whitespace columns, wavelength (µm) and reflectance.

    Returns
    -------
    wvl : np.ndarray
        Wavelengths in nm.
    rfl : np.ndarray
        Reflectance.
    """
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['w', 'r'])
    return df['w'].values * 1000.0, df['r'].values.astype(float)


# Shared 2151-channel wavelength grid (nm) for USGS splib07 ASD-resampled files.
_s07_wvl = pd.read_csv(
    USGS_WAVELENGTH_FILE,
    skiprows=1, header=None)[0].values.astype(float) * 1000.0


def read_usgs(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read a USGS splib07 ASD-resampled spectrum (one header line, one value per line).

    Returns
    -------
    wvl, rfl : np.ndarray or None
        Wavelengths (copy of the s07 grid) and reflectance with "deleted"
        values (< -1) as NaN; ``(None, None)`` if the channel count differs.
    """
    rfl = pd.read_csv(path, skiprows=1, header=None)[0].values.astype(float)
    rfl[rfl < -1] = np.nan
    if len(rfl) != len(_s07_wvl):
        return None, None
    return _s07_wvl.copy(), rfl


def read_relab(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a RELAB ``.tab`` spectrum (wavelength nm, reflectance); non-numeric lines skipped.

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
                w.append(wi); r.append(ri)
    return np.asarray(w), np.asarray(r)


def usgs_name(path: str) -> str:
    """Sample description from a USGS file's header (text after ':'), else the file stem."""
    with open(path) as fh:
        header = fh.readline().strip()
    return header.split(':', 1)[-1].strip() if ':' in header else os.path.splitext(os.path.basename(path))[0]


def relab_name(path: str) -> tuple[str, str]:
    """
    Mineral subtype and specimen name from the PDS4 ``.xml`` label beside a RELAB file.

    Returns
    -------
    subtype, name : str
        Empty strings if the label or fields are missing.
    """
    xml = path[:-4] + '.xml'
    sub = name = ''
    if os.path.exists(xml):
        txt = open(xml).read()
        m = re.search(r'<speclib:mineral_subtype>([^<]*)', txt)
        sub = m.group(1).strip() if m else ''
        m = re.search(r'<speclib:specimen_name>([^<]*)', txt)
        name = m.group(1).strip() if m else ''
    return sub, name


def add_record(library: str, spec_id: str, text: str, wvl: np.ndarray, rfl: np.ndarray) -> None:
    """
    Classify a reference spectrum by its label and append it to ``records``.

    Parameters
    ----------
    library : str
        ``'crowley'``, ``'usgs'`` or ``'relab'``.
    spec_id : str
        Identifier (file stem or name) reported in the output.
    text : str
        Label text searched for mineral names.
    wvl, rfl : np.ndarray
        Spectrum (wavelength in nm).
    """
    mins = classify_minerals(text)
    # The Crowley paracoquimbite spectrum is used as a coquimbite reference.
    if library == 'crowley' and spec_id.lower() == 'paracoquimbite':
        mins = {'coquimbite'}
    records.append(dict(
        library=library, spec_id=spec_id,
        mineral=(next(iter(mins)) if len(mins) == 1 else None),
        is_pure=(len(mins) == 1), wvl=wvl, rfl=rfl,
    ))


# --- Load all three libraries ------------------------------------------------
records = []
for path in sorted(glob.glob(f'{REF_DIR}/crowley/*.txt')):
    spec_id = os.path.splitext(os.path.basename(path))[0]
    w, r = read_crowley(path)
    add_record('crowley', spec_id, spec_id, w, r)

for path in sorted(glob.glob(f'{REF_DIR}/usgs_minerals/*.txt')):
    if 'Wavelength' in path:
        continue
    w, r = read_usgs(path)
    if w is None:
        continue
    name = usgs_name(path)
    add_record('usgs', os.path.basename(path), name + ' ' + os.path.basename(path), w, r)

for path in sorted(glob.glob(f'{REF_DIR}/relab/*.tab')):
    sub, name = relab_name(path)
    w, r = read_relab(path)
    add_record('relab', os.path.splitext(os.path.basename(path))[0], f'{sub} {name}', w, r)

# --- Band parameters for every spectrum with VNIR coverage -------------------
records = [r for r in records if has_vnir(r['wvl'], r['rfl'])]
for rec in records:
    rec['params'] = spectrum_params(rec['wvl'], rec['rfl'])

rows = []
for rec in records:
    p = rec['params']
    rows.append(dict(
        library=rec['library'], spec_id=rec['spec_id'], mineral=rec['mineral'], is_pure=rec['is_pure'],
        b900_center=p.get('b900_center', np.nan) if p else np.nan,
        b900_start=p.get('b900_start', np.nan) if p else np.nan,
        b2265_center=p.get('b2265_center', np.nan) if p else np.nan,
    ))
band_df = pd.DataFrame(rows)


def column_frame(minerals: list[str], col: str, drop: set[tuple[str, str]] | tuple = ()) -> pd.DataFrame:
    """
    Pure-reference rows for ``minerals`` with ``col`` copied into ``value``.

    Parameters
    ----------
    minerals : list of str
        Minerals to include.
    col : str
        ``band_df`` column to summarize.
    drop : set of (library, spec_id), optional
        Spectra to exclude.

    Returns
    -------
    pd.DataFrame
        ``band_df`` subset plus ``group`` (= mineral), ``value`` and ``use``
        (all True) columns.
    """
    df = band_df[band_df['is_pure'] & band_df['mineral'].isin(minerals)].copy()
    if drop:
        keep = [(a, b) not in drop for a, b in zip(df['library'], df['spec_id'])]
        df = df.loc[keep]
    df['group'] = df['mineral']
    df['value'] = df[col]
    df['use'] = True
    return df


def window_frame(pairs: list[tuple[str, float, float, str]]) -> pd.DataFrame:
    """
    Band centers of pure references in mineral-specific windows.

    Parameters
    ----------
    pairs : list of (mineral, lo, hi, group)
        For each pure spectrum of ``mineral``, the CR band center in
        ``[lo, hi]`` nm is reported under ``group``.

    Returns
    -------
    pd.DataFrame
        Columns ``library``, ``group``, ``value`` and ``use`` (all True).
    """
    rows = []
    for mineral, lo, hi, group in pairs:
        for rec in records:
            if not (rec['is_pure'] and rec['mineral'] == mineral and rec['params']):
                continue
            p = rec['params']
            rows.append(dict(
                library=rec['library'], group=group,
                value=band_center(p['wavelengths'], p['cr'], lo, hi), use=True,
            ))
    return pd.DataFrame(rows)


def stat_rows(feature: str, df: pd.DataFrame, groups: list[str]) -> list[dict[str, Any]]:
    """
    Summary rows (n, mean, +/-1 sigma and +/-2 sigma ranges) per group.

    Parameters
    ----------
    feature : str
        Feature label for the output table.
    df : pd.DataFrame
        Frame with ``group``, ``value`` and ``use`` columns; only rows with
        ``use`` True are summarized.
    groups : list of str
        Groups to report, in order.

    Returns
    -------
    list of dict
        One row per group.
    """
    out = []
    for g in groups:
        vals = df.loc[(df['group'] == g) & df['use'], 'value']
        mean, std, n = mean_std(vals)
        fmt = (lambda k: f'{mean - k * std:.1f}–{mean + k * std:.1f}') if np.isfinite(mean) else (lambda k: '—')
        out.append(dict(feature=feature, mineral=g, n=n,
                        mean=round(mean, 1) if np.isfinite(mean) else np.nan,
                        **{'±1σ': fmt(1), '±2σ': fmt(2)}))
    return out


# --- Per-feature statistics --------------------------------------------------
# Coquimbite spectra whose NIR band center is above 850 nm are treated as
# outliers and excluded from the NIR band-center statistics only.
coq_drop = set(zip(
    band_df.loc[(band_df['mineral'] == 'coquimbite') & (band_df['b900_center'] > 850), 'library'],
    band_df.loc[(band_df['mineral'] == 'coquimbite') & (band_df['b900_center'] > 850), 'spec_id'],
))

summary = []

df = column_frame(TARGET_MINERALS, 'b900_center', coq_drop)
summary += stat_rows('NIR band center', df, list(TARGET_MINERALS))

pairs = [
    ('coquimbite', 500, 600, 'coquimbite', 'Center of absorption band between 500–600 nm'),
    ('rhomboclase', 450, 550, 'rhomboclase', 'Center of absorption band between 450–550 nm'),
]
df = window_frame([(m, lo, hi, g) for m, lo, hi, g, _ in pairs])
for mineral, lo, hi, group, feature in pairs:
    summary += stat_rows(feature, df, [group])

df = column_frame(['ferricopiapite', 'hematite'], 'b900_start')
summary += stat_rows('NIR band edge', df, ['ferricopiapite', 'hematite'])

# Jarosite: spectra with a 2230-2290 nm center below 2250 nm are excluded.
df = column_frame(['jarosite'], 'b2265_center')
df = df.dropna(subset=['value'])
df['use'] = df['value'] >= 2250
summary += stat_rows('Center of absorption band between 2230–2290 nm', df, ['jarosite'])

pairs = [('goethite', 600, 750, 'goethite')]
df = window_frame(pairs)
summary += stat_rows('Center of absorption band between 600–750 nm', df, ['goethite'])

for mineral in ['schwertmannite', 'ferrihydrite']:
    df = column_frame([mineral], 'b900_start')
    summary += stat_rows('NIR band edge', df, [mineral])

print(pd.DataFrame(summary).to_string(index=False))
