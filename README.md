# dem_check

Code to evaluate ice-sheet DEMs against imagery.

The `ldcheck` package checks whether the *shape* of each 2-m REMA strip DEM agrees with
Landsat 8/9 panchromatic images. It predicts what each image should look like from the
DEM, using that image's sun elevation and azimuth, and compares the prediction with the
image. DEM areas that the images contradict can then be removed from the filtered 16-m or
32-m DEMs that go into altimetryFit, and DEMs that are mostly contradicted can be listed as
problem DEMs.

The motivation: `dem_filter.py` (altimetryFit) treats very rough or steep surfaces as DEM
errors, which is wrong over active crevasses and rifts, and altimetryFit finds bad DEMs only
indirectly, through iterative outlier editing and repeated fits. Landsat imagery gives an
independent check of the DEM's shape.

## Status (2026-11-15)

- The workflow is built and was tested on 123 DEM segments (122 finished, 1 had only cloudy
  images): 50 DEMs with small residuals in the SPK fits, 42 segments from the SPK problem
  lists, 27 from other regions' problem lists, and 3 from `maybe_not_problem_DEMs.txt`.
- 29 more SPK problem-list segments were deferred and have not been run.
- Development is moving to a Claude instance on an AWS server, which can read Landsat and
  REMA from S3. The code was developed in `~/DEM_regions/SPK/landsat_DEM_check` on Ben's
  workstation; the test results and figures are still there, and the check outputs and
  Landsat chips are in `/Volumes/ice1/ben/REMA/landsat_DEM_check/`.
- Detailed notes, including the development history and performance work, are in
  [`notes/STATUS.md`](notes/STATUS.md).

## Installation

    conda install -c conda-forge gdal numba     # GDAL is best installed with conda
    pip install -e .

The package depends on [altimetryFit](https://github.com/SmithB/altimetryFit) (`dem_filter`),
[pointCollection](https://github.com/SmithB/pointCollection) and
[pointAdvection](https://github.com/SmithB/pointAdvection), and requires numba. If they are
already installed from local clones, use `pip install --no-deps -e .` so that pip does not
replace them. Run the code with the conda environment activated, or GDAL may not find PROJ's
`proj.db`.

Downloading Landsat data needs a USGS ERS login in `~/.netrc` (machine `ers.cr.usgs.gov`).
**Read the ERS warning below before running anything in parallel.**

## Method

Each strip is processed in tiles of about 15 km with a 900-m pad.

1. **Find images.** Landsat 8/9 Collection-2 Level-1 scenes from the USGS STAC (landsatlook,
   collection `landsat-c2l1`) within ±45 days of the DEM, sun elevation above 8°. Up to 6
   acquisitions, ranked by |dt|/20 days + cloud cover/40 %. Rows of one pass are mosaicked.
2. **Download chips.** Windowed COG reads (`/vsicurl/` with an ERS cookie, `GDAL_NUM_THREADS=8`),
   only the 512-px blocks that touch the DEM footprint: pan at 15 m, SWIR 1.6 µm and cirrus at
   60 m. Sun angles are computed (`sun.py`, NOAA algorithm, within 0.05° of the SAA/SZA bands).
3. **Predict the image.** The DEM is read on a 3-m grid aligned with the 15-m pan pixels.
   Direct illumination mu = max(0, n·s) with cast shadows (numba ray march), 5×5 block mean to
   15 m, Gaussian PSF σ 0.75 px. Grid azimuth = geographic azimuth + longitude (EPSG:3031).
4. **Normalize images.** rho′ = M·DN + A; o = HP(rho′) / (LP(rho′) / sin e) with a 200-m
   high-pass, so that under a clear sky o ≈ f_dir · HP(mu).
5. **Advect and register.** Images are sampled at x + v·(t_img − t_DEM) + shift, with v from the
   ITS_LIVE 120-m composite. Masked NCC registration: each image against its prediction
   (global, ±300 m), against a reference image (global and 96-px blocks), and the stack median
   against the DEM prediction (blocks).
6. **Screen clouds.** A 60-m spectral mask (NDSI from pan and SWIR, SWIR and cirrus
   reflectance), then a pairwise screen: texture that two images share is "established", and an
   image that disagrees with it, or adds texture over smooth snow, is cloudy there. Each image
   gets a relative gain and a noise variance.
7. **Pool statistics** over the clear images (Gaussian σ 90 m, weights 1/noise²):
   F = cov(o, p) / (f · var(p)), the fraction of the predicted shading the images confirm;
   SNR; pred_amp; and the unconfirmed amplitude U = pred_amp · √(1 − F). The clear-image gain f
   is estimated from the DEM and clamped to [0.7, 1.0].
8. **Classify** each 15-m cell: 0 no data, 1 no signal (SNR < 3 or pred_amp < 0.005),
   2 consistent (F ≥ 0.5), 3 marginal, 4 inconsistent (F < 0.25), 5 image texture missing from
   the DEM. Output `<strip>_ldcheck.tif` (bands class, F, SNR, r, pred_amp, image_amp,
   unconfirmed_amp, n_clear, DEM_valid_frac, matched_frac), a JSON summary and a PNG.

## Usage

    # check one strip (strip name or 2-m dem file)
    python -m ldcheck.check_dem <strip> --out_dir <qa_dir> [--roi x0,y0,x1,y1] [--save_debug]

    # whole-DEM verdict: ok / bad / unknown (or the reason, with --why)
    python -m ldcheck.flag <qa_dir>/<strip>/<strip>_ldcheck.json --filtered <16m dem_filt.tif> [--why]

    # edited filtered DEM, same bands and grid as the dem_filter output
    python -m ldcheck.edit <2m dem> <ldcheck.tif> <out_dem_filt.tif> --res 16 \
        --standard_file <16m dem_filt.tif> [--min_unconfirmed 0.02]

    # get_pgc_2res.sh plus the check, the edit and the flag (run in a region directory)
    get_pgc_2res_ldcheck.sh <strip .json URL> --res 16

- **Whole-DEM rule** (`flag.py`): bad if ≥ 20 % of the cells kept by the standard 16-m filter
  are class 4; otherwise unknown if less than 20 % of the DEM is seen by a clear image or
  less than 1 % of the kept cells have a shading signal.
- **Edit** (`edit.py`): remove class-4 cells with U ≥ 0.02 (about a 2 % slope error along the
  sun direction), dilated by 30 m; add cells from a relaxed `dem_filter` run (slope_tol 4,
  smooth_tol 40, R_tol 200, 2-px matchtag erosion) where the class is 2. `dem_filter.py` itself
  is not modified.
- **Pipeline** (`get_pgc_2res_ldcheck.sh`): makes the usual `16m/` output unchanged, plus
  `16m_ldcheck/<name>_dem_filt.tif`, and appends whole-DEM failures to
  `16m_ldcheck/problem_DEMs_ldcheck.txt`, which `read_DEM_data` picks up.
- **Batches:** `scripts/make_check_queue.py set.csv qa_dir queue.txt` writes a pboss queue.

## Test results (2026-11-14)

Whole-DEM rule, ≥ 20 % of the standard 16-m filtered DEM inconsistent:

| Set | Flagged |
|---|---|
| 50 low-residual DEMs (incl. 10 rifted Crosson DEMs) | 0 (highest 17.8 %) |
| 42 SPK problem-list segments | 40; the 2 misses had about 0.5 % clear-image cover |
| 27 other regions' problem-list segments | 14 (11 of 21 with ≥ 20 % clear cover) |
| 3 `maybe_not_problem_DEMs.txt` segments | 0 (8–17 %) |

AUC 0.97 against the SPK lists, 0.79 against the other lists. The same statistic over the
whole 2-m DEM does worse (AUC 0.84), because good strips contain cloud tops and unmatched
blobs that dem_filter already removes. Other-list DEMs that pass mostly have shapes that
match the images; their problems are presumably offsets, tilts or long-wavelength warps,
which shading cannot see.

What the edit (class 4 with U ≥ 0.02) would remove, against the altimetryFit fits:

| Set | Fit points removed | Fit edit fraction, removed / kept | Mean \|residual\|, removed / kept |
|---|---|---|---|
| good (40) | 0.7 % | 8.0 % / 0.6 % | 0.158 / 0.032 m |
| good_rough (10) | 2.6 % | 14 % / 3.2 % | 0.241 / 0.049 m |
| maybe_not_SPK (3) | 16 % | 65 % / 7.4 % | 1.17 / 0.078 m |
| problem_SPK (17 in fits) | 53 % | 44 % / 15 % | 1.07 / 0.12 m |

The relaxed filter adds almost nothing at 16 m: dem_filter with the 16m_E4 settings removed
only about 2 % of the cells the images confirm, and the combined output gains 0.01–0.26 % of
cells. At 16 m the benefit is in the removals. The 32-m filter settings are untested.

The per-DEM tables, figures and the fit-point file are in
`~/DEM_regions/SPK/landsat_DEM_check/{results,plots}` (not in the repository).

## Inventory of processed strips and problem lists

`scripts/inventory_rema.py` and `scripts/collect_problem_lists.py` scan
`/Volumes/ice1/ben/REMA` and write to `$LDCHECK_WORK/handoff_lists/`:

- `rema_strips.csv` (one row per strip segment) and `rema_files.csv.gz` (one row per file):
  every 2-m strip file and every `*_dem_filt.tif`, by directory. On 2026-11-15: 148,305 strip
  segments, 29,691 with complete 2-m files, 137,415 with a 16-m and 142,037 with a 32-m
  filtered version, in `fullres/<region>/`, `season_covers_v2/<season>/` and the top-level
  season directories (`annual_covers_v1`).
- `problem_DEMs_unique.csv` (one row per strip segment) and `problem_DEMs_all_entries.csv`
  (one row per list entry): every DEM in every problem list, found case-insensitively,
  including backups, dated versions, `more_problem_DEMs*`, `maybe_not_problem_DEMs.txt` and DEMs
  moved into `problems/` directories. On 2026-11-15: 90 list files, 9,973 entries, 4,345 strip
  segments plus one whole-strip entry. Columns say which lists `altimetryFit.read_DEM_data`
  reads automatically (`problem_DEMs*.txt` next to a `GeoIndex*.h5`) and whether its name
  matching can use each entry.

## Repository layout

- `ldcheck/` — the package: `landsat.py` (STAC search, ERS login, chip downloads), `dem.py`,
  `shading.py`, `sun.py`, `velocity.py`, `stack.py` (normalization, registration, cloud
  screens), `compare.py`, `check_dem.py` (driver), `flag.py`, `edit.py`, `quicklook.py`.
- `scripts/` — pipeline (`get_pgc_2res_ldcheck.sh`), batch queues, test-set construction,
  analysis of the test against the altimetryFit residuals, figures, and the inventory scripts.
  Most have hard-coded paths on Ben's workstation; `LDCHECK_WORK`, `LDCHECK_QA` and
  `LDCHECK_PYTHON` override some of them.
- `test_sets/test_set.csv` — the test DEMs and their sets; `test_set_todo.csv` — remaining runs.
- `notes/STATUS.md` — detailed notes.

## USGS ERS login warning

On 2026-11-14 the first test batch ran 10 workers that each logged in to USGS ERS and forced a
new login on every failed read. That made a few hundred logins in minutes, ERS refused all
logins (HTTP 401) for about 35 minutes, and the account is shared with other Landsat work. The
code now shares one cookie across processes (`~/.cache/ldcheck/ers_cookie.json`), keeps logins
at least 2 minutes apart, stops for 30 minutes after a failed login (`ERSLoginBlocked`) and
allows at most 5 downloads at once. Before a batch, check the cookie file's `fail_time`; if
logins fail, stop rather than retry. Reading Landsat from S3 would avoid ERS altogether.

## Caveats

- Landsat 8/9 only, so DEMs from before 2013 cannot be checked.
- Shading constrains the slope along the sun azimuth only (about the same azimuth for every
  image at a site), and nothing about offsets, tilts or long wavelengths.
- Haze, fog and real surface change between the DEM and image dates can look like DEM errors.
- A strip takes 7–50 minutes, mostly download latency and processing.
- Several paths are hard-coded for Ben's workstation: the 2-m and meta directories and
  `all_REMA_urls.txt` (`ldcheck/dem.py`, `ldcheck/edit.py`), the ITS_LIVE files
  (`ldcheck/velocity.py`), and the chip and output directories (`ldcheck/check_dem.py`).

## Next steps

1. **A cache of trusted Landsat files.** Chips are now cut per strip
   (`chips/<strip>/<acquisition>/`), so a scene is downloaded again for every strip it covers,
   and cloud masks, gains and registration shifts are recomputed for every DEM. The entry
   points are `landsat.get_acquisition_chips` and `stack.load_acquisition`.
2. **Fall back to any good image from the same daylight season** when the ±45-day window gives
   too little clear imagery (`check_dem.check_dem`, around `search_scenes` and
   `select_acquisitions`). Longer time gaps make the chip pad, velocity errors and real surface
   change matter more.
3. **Use pointAdvection advectionGrid objects in place of ITS_LIVE** for advection matching,
   read with `pointCollection.grid.mosaic` from a directory that contains one set of objects
   per year. `check_dem.py` now applies constant-velocity advection, x + v·dt, in four places:
   the chip pad, the tile velocity grid, the advection of the spectral cloud masks, and image
   sampling in `_register_tile`. An advectionGrid gives the position at t_img of the parcel at
   (x, y) at t_DEM through x0, y0 (to the epoch) and x1, y1 (from the epoch).

Also open: running the 29 deferred SPK segments, testing the 32-m filter settings and
crevassed margins, and three decisions for Ben: whether to adopt the 20 % rule (the closest
good DEM was at 17.8 %), the edit threshold (`--min_unconfirmed`), and whether to run the
deferred segments.
