import os, re, glob, warnings
import numpy as np
import pandas as pd
from pysptools.spectro import SpectrumConvexHullQuotient
warnings.simplefilter('ignore')

REF_DIR = 'ref_spectra'
TARGET_MINERALS = ['coquimbite', 'rhomboclase', 'ferricopiapite', 'hematite',
                   'jarosite', 'schwertmannite', 'goethite', 'ferrihydrite']

def continuum_removal(refl, wavelengths):
    mask = ~np.isnan(refl)
    chq = SpectrumConvexHullQuotient(refl[mask].tolist(), wavelengths[mask].tolist())
    cr = np.full(len(wavelengths), np.nan)
    cr[mask] = chq.get_continuum_removed_spectrum()
    return cr

def band_center(wvl, cr, lo, hi, min_depth=0.01):
    m = (wvl >= lo) & (wvl <= hi) & np.isfinite(wvl) & np.isfinite(cr)
    if m.sum() < 3:
        return np.nan
    w, c = wvl[m], cr[m]
    i = int(np.argmin(c))
    if 1.0 - float(c[i]) <= min_depth:
        return np.nan
    return float(w[i])

def band_edge(wvl, cr):
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

def spectrum_params(wvl, rfl):
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    if np.isfinite(rfl).sum() < 11:
        return None
    try:
        cr = continuum_removal(rfl, wvl)
    except (IndexError, ValueError):
        return None
    valid = ~np.isnan(cr)
    if valid.sum() < 11 or ((wvl[valid] >= 720) & (wvl[valid] <= 1200)).sum() < 3:
        return None
    center = band_center(wvl, cr, 720, 1200)
    return dict(
        b900_center=center,
        b900_start=band_edge(wvl, cr) if np.isfinite(center) else np.nan,
        b2265_center=band_center(wvl, cr, 2230, 2290),
        wavelengths=wvl, cr=cr,
    )

def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return float(vals.mean()), std, int(vals.size)


def classify_minerals(text):
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

def has_vnir(wvl, rfl, lo=750.0, hi=1000.0, min_points=10):
    wvl = np.asarray(wvl, dtype=float)
    rfl = np.asarray(rfl, dtype=float)
    return int(((wvl >= lo) & (wvl <= hi) & np.isfinite(rfl)).sum()) >= min_points

def read_crowley(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['w', 'r'])
    return df['w'].values * 1000.0, df['r'].values.astype(float)

_s07_wvl = pd.read_csv(
    f'{REF_DIR}/usgs_minerals/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
    skiprows=1, header=None)[0].values.astype(float) * 1000.0

def read_usgs(path):
    rfl = pd.read_csv(path, skiprows=1, header=None)[0].values.astype(float)
    rfl[rfl < -1] = np.nan
    if len(rfl) != len(_s07_wvl):
        return None, None
    return _s07_wvl.copy(), rfl

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
                w.append(wi); r.append(ri)
    return np.asarray(w), np.asarray(r)

def usgs_name(path):
    with open(path) as fh:
        header = fh.readline().strip()
    return header.split(':', 1)[-1].strip() if ':' in header else os.path.splitext(os.path.basename(path))[0]

def relab_name(path):
    xml = path[:-4] + '.xml'
    sub = name = ''
    if os.path.exists(xml):
        txt = open(xml).read()
        m = re.search(r'<speclib:mineral_subtype>([^<]*)', txt)
        sub = m.group(1).strip() if m else ''
        m = re.search(r'<speclib:specimen_name>([^<]*)', txt)
        name = m.group(1).strip() if m else ''
    return sub, name

def add_record(library, spec_id, text, wvl, rfl):
    mins = classify_minerals(text)
    if library == 'crowley' and spec_id.lower() == 'paracoquimbite':
        mins = {'coquimbite'}
    records.append(dict(
        library=library, spec_id=spec_id,
        mineral=(next(iter(mins)) if len(mins) == 1 else None),
        is_pure=(len(mins) == 1), wvl=wvl, rfl=rfl,
    ))

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


def column_frame(minerals, col, drop=()):
    df = band_df[band_df['is_pure'] & band_df['mineral'].isin(minerals)].copy()
    if drop:
        keep = [(a, b) not in drop for a, b in zip(df['library'], df['spec_id'])]
        df = df.loc[keep]
    df['group'] = df['mineral']
    df['value'] = df[col]
    df['use'] = True
    return df

def window_frame(pairs):
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

def stat_rows(feature, df, groups):
    out = []
    for g in groups:
        vals = df.loc[(df['group'] == g) & df['use'], 'value']
        mean, std, n = mean_std(vals)
        fmt = (lambda k: f'{mean - k * std:.1f}–{mean + k * std:.1f}') if np.isfinite(mean) else (lambda k: '—')
        out.append(dict(feature=feature, mineral=g, n=n,
                        mean=round(mean, 1) if np.isfinite(mean) else np.nan,
                        **{'±1σ': fmt(1), '±2σ': fmt(2)}))
    return out

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

