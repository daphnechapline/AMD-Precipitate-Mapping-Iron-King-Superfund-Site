"""
Generate a small synthetic dataset in every input format the analysis expects.

Used to verify an installation end to end without the real field and imagery
data. The spectra are Gaussian absorption features on a sloped continuum and
the results carry no scientific meaning.

Usage (from the repository root)::

    python test/make_synthetic_data.py test/synthetic
    cd test/synthetic
    python ../../Reference_Spectra_Analysis.py
    python ../../Iron_King_Analysis.py        # writes test/synthetic/outputs/
"""
import argparse
import os

import numpy as np
import xarray as xr
import geopandas as gpd
import spectral.io.envi as envi
from shapely.geometry import box
from pyproj import Transformer

# Same pit names/coordinates as Iron_King_Analysis.PITS (UTM 12N meters).
PITS = {f'IK24-PIT-{i:03d}': xy for i, xy in enumerate([
    (384838, 3818515), (384968, 3818473), (384765, 3818461), (385065, 3818422), (384925, 3818328),
    (385095, 3818306), (385018, 3818178), (385256, 3818223), (385428, 3818247), (385323, 3818149)], 1)}

# Mineral -> list of (center nm, width nm, depth) Gaussian absorptions.
FEATURES = {
    'coquimbite': [(790, 90, .25), (560, 25, .10)],
    'rhomboclase': [(790, 90, .25), (520, 25, .10)],
    'ferricopiapite': [(880, 100, .25)],
    'hematite': [(880, 80, .35), (530, 40, .2)],
    'jarosite': [(925, 100, .25), (2265, 15, .15), (435, 15, .1)],
    'goethite': [(960, 110, .3), (670, 40, .1)],
    'schwertmannite': [(960, 120, .25)],
    'ferrihydrite': [(1000, 130, .25)],
}
NAMES = list(FEATURES)


def synthetic_spectrum(wvl: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Reflectance of a weighted mixture of the synthetic mineral features.

    Parameters
    ----------
    wvl : np.ndarray
        Wavelengths in nm.
    weights : np.ndarray
        Weight per mineral, shape (len(NAMES),).

    Returns
    -------
    np.ndarray
        Reflectance on ``wvl``.
    """
    continuum = 0.25 + 0.15 * (wvl - 350) / 2150
    absorption = np.zeros_like(wvl, dtype=float)
    for weight, mineral in zip(weights, NAMES):
        for center, width, depth in FEATURES[mineral]:
            absorption += weight * depth * np.exp(-((wvl - center) / width) ** 2)
    return continuum * (1 - np.clip(absorption, 0, .9))


def random_weights(rng: np.random.Generator) -> np.ndarray:
    """One dominant mineral plus a minor random admixture."""
    weights = np.zeros(len(NAMES))
    weights[rng.integers(len(NAMES))] = rng.uniform(.6, 1.0)
    weights[rng.integers(len(NAMES))] += rng.uniform(0, .3)
    return weights


def random_cube(rng: np.random.Generator, wvl: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Noisy mixture spectra, shape (n_rows, n_cols, n_bands)."""
    cube = np.empty((n_rows, n_cols, wvl.size))
    for i in range(n_rows):
        for j in range(n_cols):
            cube[i, j] = synthetic_spectrum(wvl, random_weights(rng)) + rng.normal(0, 2e-3, wvl.size)
    return cube


def make_dataset(root: str, seed: int = 0) -> None:
    """
    Write ``data/`` and ``ref_spectra/`` under ``root``.

    Parameters
    ----------
    root : str
        Output directory (created if needed).
    seed : int, optional
        Random seed.
    """
    rng = np.random.default_rng(seed)

    def subdir(*parts: str) -> str:
        path = os.path.join(root, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    # ASD field spectra; the POW file exercises the skip rule in load_asd.
    asd_wvl = np.arange(350, 2501, 1.0)
    for key in list(PITS) + ['IK24-PIT-001-POW']:
        rfl = synthetic_spectrum(asd_wvl, random_weights(rng)) + rng.normal(0, 1e-3, asd_wvl.size)
        np.savetxt(os.path.join(subdir('data', 'asd'), f'{key}.txt'), np.c_[asd_wvl, rfl],
                   header='Wavelength Reflectance', comments='', fmt='%.5f')

    east = np.array([p[0] for p in PITS.values()])
    north = np.array([p[1] for p in PITS.values()])
    x0, x1, y0, y1 = east.min() - 150, east.max() + 150, north.min() - 150, north.max() + 150
    to_wgs = Transformer.from_crs('EPSG:32612', 'EPSG:4326', always_xy=True)

    # AOI polygon around the pits.
    aoi = gpd.GeoDataFrame(
        {'Name': ['Iron King']},
        geometry=[box(east.min() - 40, north.min() - 40, east.max() + 40, north.max() + 40)],
        crs='EPSG:32612')
    aoi.to_crs('EPSG:4326').to_file(os.path.join(subdir('data'), 'IronKing.kml'), driver='KML')

    # EMIT: lat/lon grid (~60 m), dims deliberately not in the loader's order,
    # with one -9999 fill pixel.
    emit_wvl = np.linspace(381, 2493, 285)
    lon0, lat1 = to_wgs.transform(x0, y1)
    lon1, lat0 = to_wgs.transform(x1, y0)
    lat = np.arange(lat1, lat0, -0.00055)
    lon = np.arange(lon0, lon1, 0.00065)
    emit = random_cube(rng, emit_wvl, lat.size, lon.size)
    emit[0, 0, :] = -9999
    xr.Dataset(
        {'reflectance': (('latitude', 'longitude', 'wavelengths'), emit.astype('float32'))},
        coords={'latitude': lat, 'longitude': lon, 'wavelengths': emit_wvl,
                'fwhm': ('wavelengths', np.full(285, 8.5))},
    ).to_netcdf(os.path.join(root, 'data', 'emit_ortho_crop.nc'))

    # AVIRIS: ENVI, 15 m UTM, int16 scaled by 10000, bands stored unsorted.
    av_wvl = np.linspace(366, 2496, 224)
    perm = rng.permutation(av_wvl.size)
    n_rows, n_cols = int((y1 - y0) // 15), int((x1 - x0) // 15)
    aviris = (random_cube(rng, av_wvl, n_rows, n_cols) * 10000).astype('int16')[:, :, perm]
    envi.save_image(os.path.join(root, 'data', 'aviris_rfl.hdr'), aviris, force=True, ext='.img', metadata={
        'map info': f'{{UTM, 1.000, 1.000, {x0:.1f}, {y1:.1f}, 15.0, 15.0, 12, North, WGS-84, units=Meters}}',
        'wavelength': [f'{v:.2f}' for v in av_wvl[perm]],
        'fwhm': ['10.0'] * av_wvl.size,
        'wavelength units': 'Nanometers'})

    # AVIRIS-5: root easting/northing + 'reflectance' group, 10 m UTM.
    av5_wvl = np.linspace(380, 2500, 284)
    av5_east = np.arange(x0, x1, 10.0) + 5
    av5_north = np.arange(y1, y0, -10.0) - 5
    av5 = np.moveaxis(random_cube(rng, av5_wvl, av5_north.size, av5_east.size), -1, 0).astype('float32')
    av5_path = os.path.join(root, 'data', 'aviris5_rfl.nc')
    xr.Dataset({'easting': ('easting', av5_east), 'northing': ('northing', av5_north)}).to_netcdf(av5_path)
    xr.Dataset(
        {'reflectance': (('wavelength', 'northing', 'easting'), av5),
         'fwhm': ('wavelength', np.full(av5_wvl.size, 7.5))},
        coords={'wavelength': av5_wvl},
    ).to_netcdf(av5_path, group='reflectance', mode='a')

    # Crowley: two columns, wavelength in µm; includes the paracoquimbite case.
    lab_wvl = np.arange(350, 2501, 2.0)
    for mineral in NAMES + ['paracoquimbite']:
        weights = np.eye(len(NAMES))[NAMES.index('coquimbite' if mineral == 'paracoquimbite' else mineral)]
        np.savetxt(os.path.join(subdir('ref_spectra', 'crowley'), f'{mineral}.txt'),
                   np.c_[lab_wvl / 1000, synthetic_spectrum(lab_wvl, weights)], fmt='%.6f')

    # USGS splib07 ASD-resampled: shared wavelength file + one value per line,
    # with a few "deleted" (-1.23e34) channels.
    s07_wvl = np.linspace(350, 2500, 2151)
    usgs_dir = subdir('ref_spectra', 'usgs_minerals')
    with open(os.path.join(usgs_dir, 's07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt'), 'w') as fh:
        fh.write('Wavelengths\n' + '\n'.join(f'{v / 1000:.6f}' for v in s07_wvl))
    for i, mineral in enumerate(NAMES):
        rfl = synthetic_spectrum(s07_wvl, np.eye(len(NAMES))[i])
        rfl[:5] = -1.23e34
        with open(os.path.join(usgs_dir, f's07_ASD_{mineral.title()}_GDS{i}_ASDFRa_AREF.txt'), 'w') as fh:
            fh.write(f'splib07a Record={i}: {mineral.title()} GDS{i} ASDFRa AREF\n'
                     + '\n'.join(f'{v:.6e}' for v in rfl))

    # RELAB: .tab (wavelength nm, reflectance) + minimal PDS4-style .xml label.
    relab_dir = subdir('ref_spectra', 'relab')
    for i, mineral in enumerate(['jarosite', 'goethite', 'hematite']):
        weights = np.eye(len(NAMES))[NAMES.index(mineral)]
        np.savetxt(os.path.join(relab_dir, f'c1rl{i}.tab'),
                   np.c_[lab_wvl, synthetic_spectrum(lab_wvl, weights)], fmt='%.5f')
        with open(os.path.join(relab_dir, f'c1rl{i}.xml'), 'w') as fh:
            fh.write(f'<x><speclib:mineral_subtype>{mineral.title()}</speclib:mineral_subtype>'
                     f'<speclib:specimen_name>{mineral} sample</speclib:specimen_name></x>')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a synthetic Iron King input dataset.')
    parser.add_argument('root', nargs='?', default='test/synthetic', help='output directory')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    make_dataset(args.root, args.seed)
    print(f'Synthetic dataset written to {args.root}')
