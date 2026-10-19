# AMD Precipitate Mapping: Iron King Mine Superfund Site

Maps iron-bearing acid-mine-drainage (AMD) precipitates at the Iron King Mine –
Humboldt Smelter Superfund site (Dewey-Humboldt, Arizona) by combining field ASD
spectra with EMIT, AVIRIS and AVIRIS-5 imaging spectroscopy.

Eight target minerals are mapped:

| ID | Mineral        | ID | Mineral        |
|----|----------------|----|----------------|
| 1  | Coquimbite     | 5  | Jarosite       |
| 2  | Rhomboclase    | 6  | Goethite       |
| 3  | Ferricopiapite | 7  | Schwertmannite |
| 4  | Hematite       | 8  | Ferrihydrite   |

Class 0 means unclassified.

Every pixel inside the study-area polygon is classified three independent ways:

1. **Decision tree:** rules on continuum-removed absorption-band positions.
2. **Spectral Angle Mapper (SAM):** compared against mean laboratory reference spectra.
3. **Non-negative least squares (NNLS):** unmixing against the same reference endmembers.

## Repository contents

| Path | Purpose |
|------|---------|
| `Iron_King_Analysis.py` | Main pipeline: loads all data, computes band parameters, runs the three classifiers, writes tables to `outputs/`. |
| `Reference_Spectra_Analysis.py` | Prints reference-library statistics (mean, ±1σ, ±2σ) for the diagnostic band positions that the decision-tree windows are based on. |
| `requirements.txt` | Python dependencies. |

## Requirements

- Python 3.10 or newer (tested with Python 3.11 on macOS)
- The packages in `requirements.txt`. All of them ship binary wheels, and GDAL
  comes bundled with `rasterio` and `pyogrio`, so no system GDAL install is
  needed.

The package versions that were tested are numpy 2.4, scipy 1.17, pandas 3.0,
xarray 2026.9, rioxarray 0.19, geopandas 1.2, rasterio 1.4, spectral 0.25 and
pysptools 0.15.

## Installation

```bash
git clone https://github.com/daphnechapline/AMD-Precipitate-Mapping-Iron-King-Superfund-Site.git
cd AMD-Precipitate-Mapping-Iron-King-Superfund-Site

python3 -m venv ~/.venvs/iron-king
source ~/.venvs/iron-king/bin/activate          # Windows: %USERPROFILE%\.venvs\iron-king\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input data

The input data is **not** included in this repository. Put it in the layout
below, relative to the repository root (both scripts use relative paths and
must be run from the directory that holds `data/` and `ref_spectra/`):

```
data/
├── asd/
│   ├── IK24-PIT-001.txt … IK24-PIT-010.txt
├── emit_ortho_crop.nc
├── aviris_rfl.hdr  (+ its binary file, e.g. aviris_rfl or aviris_rfl.img)
├── aviris5_rfl.nc
└── IronKing.kml
ref_spectra/
├── crowley/*.txt
├── usgs_minerals/
│   ├── s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt
│   └── *.txt
└── relab/*.tab  (+ matching *.xml labels)
```

If something is missing, both scripts stop right away and list the missing
paths.

### File formats

| Input | Expected format |
|-------|-----------------|
| `data/asd/<PIT>.txt` | Whitespace-delimited table with a header row that includes `Wavelength` (nm) and `Reflectance` columns. |
| `data/emit_ortho_crop.nc` | EMIT L2A reflectance, orthorectified to a regular lat/lon grid and cropped to the site. Variable `reflectance` with dimensions `wavelengths`, `latitude`, `longitude` (any order). `wavelengths` and `fwhm` (nm) variables. Fill value −9999. This matches the orthorectified output of the LP DAAC `emit_tools` module. |
| `data/aviris_rfl.hdr` | AVIRIS surface reflectance in ENVI format, with `map info` in UTM, `wavelength` (nm) and, optionally, `fwhm` (otherwise 10 nm is assumed). Values stored as reflectance × 10000 are detected and rescaled. |
| `data/aviris5_rfl.nc` | AVIRIS-5 reflectance NetCDF. Root variables `easting` and `northing` (UTM 12N, m). A `reflectance` group with `reflectance` (dimensions: band, `northing`, `easting`), `wavelength` and `fwhm` (nm). |
| `data/IronKing.kml` | Study-area polygon or polygons. |
| `ref_spectra/crowley/*.txt` | Two whitespace-delimited columns: wavelength (µm) and reflectance. The mineral is read from the file name. |
| `ref_spectra/usgs_minerals/*.txt` | USGS splib07 ASD-resampled spectra: one header line (`… Record=N: <sample description>`), then one reflectance value per line on the shared 2151-channel wavelength file. |
| `ref_spectra/relab/*.tab` | RELAB spectra: wavelength (nm) and reflectance columns. The mineral is read from `speclib:mineral_subtype` and `speclib:specimen_name` in the PDS4 `.xml` label beside each file. |


### Reference Data sources

| Input | Source |
|-------|--------|
| USGS library | Kokaly, R.F., et al. (2017). *USGS Spectral Library Version 7 Data.* U.S. Geological Survey data release. https://doi.org/10.5066/F7RR1WDJ (ASD-resampled "s07_ASD" files). |
| RELAB library | Brown University RELAB spectral library, distributed through the NASA PDS Geosciences Node (PDS4 format). |
| Crowley library | Crowley, J.K., et al. (2003). Spectral reflectance properties (0.4–2.5 µm) of secondary Fe-oxide, Fe-hydroxide, and Fe-sulphate-hydrate minerals associated with sulphide-bearing mine wastes. *Geochemistry: Exploration, Environment, Analysis*, 3, 219–228. |

## Usage

Run from the repository root, with the virtual environment activated:

```bash
python Reference_Spectra_Analysis.py   # reference band-position statistics (printed only)
python Iron_King_Analysis.py           # full pipeline, writes outputs/
```

### Outputs (`outputs/`)

| File | Contents |
|------|----------|
| `band_params.csv` | Decision-tree band positions for each pit source (`asd`, `emit`, `aviris`, `aviris5`, and `asd_emit`, which is ASD convolved to EMIT bands), plus the AOI-mean spectrum of each sensor. Columns are `b900_center`, `b900_start`, `b2265_center`, `cr_500_600`, `cr_450_550`, and `cr_600_750`. A center is blank when that band is absent or shallower than 0.01. |
| `band_params_by_source.csv` | Mean and standard deviation of each band parameter across the pits, per source. |
| `class_shares.csv` | Pixel (or sample) counts and percentages per class for every method × source combination. |
| `nnls_mean_abundance.csv` | Mean NNLS abundance per mineral and source. |


## Method summary

**Preprocessing.**
- Bands in the water-vapor windows (1275–1475 nm and 1750–1975 nm) and
  non-positive values are set to NaN.
- Each pit is matched to its nearest EMIT, AVIRIS and AVIRIS-5 pixel.
- ASD spectra are convolved to EMIT bands with `spectral.BandResampler`.
- Reference spectra are convolved to each sensor's bands with Gaussian
  spectral response functions and averaged per mineral to form endmembers.

**Band parameters.**
- Spectra are continuum-removed by dividing by the upper convex hull
  (`pysptools`).
- The recorded parameters are only the positions the decision tree uses: the
  ~900 nm Fe³⁺ band center, its short-wavelength shoulder (`b900_start`), the
  ~2265 nm jarosite band center, and the continuum-removed minima in the
  450–550, 500–600, and 600–750 nm windows. A position is recorded only when
  the band is deeper than 0.01. Depths, areas, and the area-bisecting
  wavelength are not calculated.

**Decision tree** (`refined_mineral_class`). Pixels are first split by the ~900
nm band center, then confirmed with a second feature:

| NIR band center | Class | Second test |
|-----------------|-------|-------------|
| 780–800 nm | Coquimbite | visible band at 545–595 nm, and no rhomboclase band |
| | Rhomboclase | visible band at 500–545 nm, and no coquimbite band |
| 865–895 nm | Ferricopiapite | band shoulder at 660–695 nm |
| | Hematite | band shoulder at 725–755 nm |
| 910–940 nm | Jarosite | 2.2 µm band centered at 2255–2275 nm |
| 940–980 nm | Goethite | band at 625–715 nm |
| | Schwertmannite | otherwise, band shoulder at 710–755 nm |
| 980–1020 nm | Ferrihydrite | band shoulder at 750–785 nm |

These windows are hard-coded. They correspond to the statistics printed by
`Reference_Spectra_Analysis.py` and need to be updated by hand if the
reference library changes.

**SAM.** A pixel gets the class of the endmember with the smallest spectral
angle. If that angle is larger than 0.2 rad, the pixel is left unclassified.

**NNLS.** Abundances are only constrained to be non-negative, not to sum to
one. They are relative weights, not areal fractions. The class is the
mineral with the largest abundance.

### Configuration

The tunable constants are at the top of `Iron_King_Analysis.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `BAD_WAVELENGTH_WINDOWS` | `[(1275, 1475), (1750, 1975)]` | Masked water-vapor windows (nm). |
| `MIN_VALID_BANDS` | EMIT and ASD 11; AVIRIS and AVIRIS-5 13 | Minimum number of valid bands required to classify a spectrum. |
| `SAM_MAX_ANGLE_RAD` | `0.2` | SAM rejection threshold (radians). |
| `NOISE_THRESHOLD` | `2e-4` | USGS reference spectra noisier than this are dropped. |
| `PITS` | 10 pits | Pit name → UTM 12N easting/northing. |

## Known issues and review notes

These are documented, not changed, because changing them would alter results:

- **ENVI georeferencing half-pixel offset.** `envi_georef_from_hdr` treats the
  `map info` tie point as a pixel center. GDAL treats it as the pixel's
  upper-left corner, so the AVIRIS grid here sits 0.5 pixel west and north of
  where GDAL puts it. Confirm which convention the AVIRIS product follows.
- **Inconsistent b900 search window.** Band-parameter tables and reference
  statistics search the ~900 nm band in 720–1200 nm, but decision-tree
  classification searches 720–1300 nm.
- **Library preparation differs between the scripts.**
  `Reference_Spectra_Analysis.py` counts Crowley "paracoquimbite" as
  coquimbite, does not noise-filter USGS spectra, and does not mask
  water-vapor bands before continuum removal. `Iron_King_Analysis.py` does
  all three the other way.
- **Nearest-pixel lookups do not check bounds.** If a pit falls outside the
  EMIT or AVIRIS-5 scene, its nearest edge pixel is used silently.
- **All warnings are suppressed** (`warnings.simplefilter('ignore')`).

## Citation and license

**TODO:** author(s), citation for any associated publication, and a license.
Until a license is added, default copyright applies.
