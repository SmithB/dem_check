# Landsat shape check for REMA strip DEMs (`ldcheck`)

Started 2026-11-14 at Ben's request: check whether the *shape* of each 2-m REMA strip DEM
agrees with Landsat 8/9 panchromatic images, so that real rough surfaces (crevasses, rifts)
can be kept and DEM artifacts removed, instead of relying on dem_filter.py's roughness and
slope limits.  Test against DEMs with small altimetryFit residuals and against the
`problem_DEMs*.txt` lists.

2026-11-15: the code moved from `~/DEM_regions/SPK/landsat_DEM_check/code` to this repository
(`~/git_repos/dem_check`, installed with `pip install -e`).  Paths below are relative to the
repository except `results/` and `plots/`, which stay in the work directory
`~/DEM_regions/SPK/landsat_DEM_check` (`$LDCHECK_WORK` in the scripts).  Development is
moving to a Claude instance on an AWS server; see `../README.md` for the summary and next steps.

## Layout

- `ldcheck/` — the package (e.g. `python -m ldcheck.check_dem NAME`)
  - `landsat.py` — USGS STAC search (landsatlook, `landsat-c2l1`, L8/L9), USGS ERS login
    (credentials from `~/.netrc`, as in landsatutilities), windowed COG reads with GDAL
    (`/vsicurl/` + login cookie, `GDAL_NUM_THREADS=8`).  Chips are cut to the DEM bounding
    box (+ advection pad), reading only the 512-px blocks that touch the DEM footprint:
    pan (15 m), SWIR-1.6 and cirrus at 60 m (spectral clouds).  Sun angles: `sun.py`.
  - `dem.py` — finds the 2-m dem/matchtag/bitmask (fixes the SPK/2m symlinks that point to
    `../ASE/2m`), the strip time (PGC json), footprint, reads onto a 3-m grid whose pixel
    edges match the 15-m pan pixels; can download from PGC.
  - `shading.py` — predicted direct illumination mu = max(0, n.s) with cast shadows (numba
    ray march), 5x5 block mean to the 15-m pan grid, plus a Gaussian PSF (sigma 0.75 px:
    the NCC between prediction and image peaks at 0.75-1 px on the Crosson test).
    Sun grid azimuth = geographic azimuth + longitude (EPSG:3031).
  - `velocity.py` — ITS_LIVE 120-m composite (default) or an h5 velocity stack.
  - `stack.py` — calibration (rho' = M*DN + A), advection to the DEM time, normalization
    o = HP(rho') / (LP(rho')/sin e) (units of mu, so o ~ f_dir * HP(mu) for a clear sky),
    masked-NCC registration (global and 96-px blocks), spectral cloud mask
    (NDSI < 0.55, B6 > 0.2, B9 > 0.025), pairwise image-consistency cloud screen.
  - `compare.py` — classification and summaries.
  - `check_dem.py` — the per-DEM driver (CLI); `flag.py` — whole-DEM verdict;
    `edit.py` — standard + relaxed dem_filter outputs combined with the classification;
    `quicklook.py`.
- `scripts/` — `make_test_sets.py`, `make_check_queue.py`, `summarize_tests.py`,
  `compare_filter_class.py`, `collect_fit_points.py`, `scan_fit_DEM_residuals.py`,
  `plot_example.py`, `get_pgc_2res_ldcheck.sh` (get_pgc_2res.sh + check + combine),
  `inventory_rema.py` and `collect_problem_lists.py` (inventory of the processed strips and
  of all problem-DEM lists under /Volumes/ice1/ben/REMA, written to `$LDCHECK_WORK/handoff_lists`).
- Data: `/Volumes/ice1/ben/REMA/landsat_DEM_check/` — `chips/<strip>/<acquisition>/`,
  `qa/<strip>/` (test-set outputs), `qa_dev/` (development runs).

## Method (per strip)

1. Landsat 8/9 L1 acquisitions within +-45 days, sun elevation > 8 deg; up to 6 (`--n_max`), ranked by
   |dt|/20 d + USGS cloud cover/40 %.  Rows of one pass are mosaicked.
2. Tiles of ~15 km (+900 m pad).  For each image: predicted mu for its sun position
   (tile-mean angles from the SAA/SZA bands), image sampled at x + v (t_img - t_DEM) + shift.
3. Registration: (a) global NCC shift of each image against its own prediction (+-300 m);
   (b) images that do not register to the DEM: global shift against the reference image;
   (c) 96-px block shifts of each image against the reference image (relative geolocation,
   velocity errors); (d) block shifts of the stack median against the DEM prediction.
4. Clouds: spectral mask, then pairwise local NCC between images: where two images agree
   (NCC > 0.5) the surface texture is "established", and an image that agrees with none,
   or whose contrast is < 0.5 of the agreeing images, is cloudy; where nothing is
   established, an image with > 3x the variance of the smoothest image is cloudy (cloud
   texture over smooth snow); albedo > 10 % off the clear median is cloudy.  Each image
   gets a relative gain g (contrast relative to the clearest images, DEM-independent) and
   a noise variance (its variance not shared with its best-matching image).
5. Statistics: local (Gaussian sigma 90 m) moments of the high-passed (200 m) image and
   prediction for each clear image, pooled with weights 1/noise^2.  The image gain of a
   clear image, f_clear, is estimated from the DEM over its strongest predicted shading but
   limited to [0.7, 1.0], so that a bad DEM cannot lower it much.
   - F = pooled cov(o, p) / (f var(p)): the fraction of the predicted shading the images
     confirm (albedo variations add noise but no bias);
   - SNR, pred_amp (predicted image std, mu units), unconfirmed_amp U = amp sqrt(1 - F).
6. Classes (15 m): 0 no data, 1 no signal, 2 consistent (F >= 0.5), 3 marginal,
   4 inconsistent (F < 0.25), 5 image texture missing from the DEM.  "Signal" needs
   SNR >= 3 and pred_amp >= 0.005 (about a 0.5 % slope along the sun direction).
7. Outputs: `<strip>_ldcheck.tif` (class, F, SNR, r, pred_amp, image_amp,
   unconfirmed_amp, n_clear, DEM_valid_frac, matched_frac), `.json` summary, `.png`.

## Test (2026-11-14): 123 checked DEM segments

Test sets (`test_sets/test_set.csv`, built by `scripts/make_test_sets.py`):
- **good** (40): from `results/fit_DEM_residuals.csv` (per-DEM residuals in the SPK hybrid
  jitter-stripe fits 2019.5-2021.5): edit fraction <= 2 %, RDE <= 5 cm, >= 5e5 points,
  8 per year 2018-2022 (15 in the first pass, 25 more after);
- **good_rough** (10): DEMs of the Crosson Lagrangian fits (rifted shelf; Ben's October QA
  found their edits follow rift walls), RDE <= 6 cm;
- **problem_SPK**: all 71 segments of the SPK lists (`16m_E4/problem_DEMs.txt`,
  `more_problem_DEMs`) in 22 strips; 42 checked (<= 3 segments per strip; the other 29
  are in `par_run_deferred/`); **maybe_not_SPK** (3): `16m_E4/maybe_not_problem_DEMs.txt`;
- **problem_other** (28): random sample (<= 4 per list) of the other `problem_DEMs*.txt`
  lists under /Volumes/ice1/ben/REMA (continent-wide season-cover lists), Landsat-8 era,
  with local 2-m files.
Outputs: `/Volumes/ice1/ben/REMA/landsat_DEM_check/qa/<strip>/`; tables in `results/`
(`test_merged.csv` = per-DEM everything); figures in `plots/`.  122 finished, 1 had only
cloudy images (`all_cloudy`).  Median 6 images per strip (4 clear), 97 % of a good strip
covered by at least one clear image.  Run time 7-50 min per strip (20 workers).

### Whole-DEM detection

Metric that works: **the fraction of the standard 16-m filtered DEM (16m_E4) that is
class 4** (`kept_frac_incons`; `compare.kept_area_stats`).  Statistics over the whole 2-m DEM
do much worse (AUC 0.84), because good strips often contain cloud tops or unmatched blobs
that dem_filter already removes (e.g. WV03_20181031 seg2).

| kept_frac_incons >= 0.2 | flagged |
|---|---|
| good + good_rough | 0 / 50 (largest 0.178, 0.167, 0.164) |
| problem_SPK | 40 / 42; the 2 misses are WV02_20161027 seg2/seg4 with ~1000 cells and 0.5 % clear-image cover ("unknown" in flag.py) |
| problem_other | 14 / 27 (11 / 21 with >= 20 % clear cover) |
| maybe_not_SPK | 0 / 3 (0.08-0.17), but see the fit residuals below |

AUC 0.97 (SPK lists), 0.79 (other lists).  Most other-list DEMs that pass have a shape
that agrees with the images; their listed problem is presumably of a kind the shading
cannot see (vertical offsets, tilts, long-wavelength warps, time mismatches).
`flag.py` uses this rule (`--filtered`); `plots/test_summary.png`.

### Where the edit would remove data, and what altimetryFit made of it

Fit points of the test DEMs (`results/fit_points_test_set.h5`, from the hybrid fits and
the Crosson 2022.0 fits), split by whether the edit (class 4 with unconfirmed amplitude
>= 0.02) would remove them (`scripts/fit_residuals_removed.py`):

| set | removed share of points | fit edit fraction, removed / kept | mean abs residual (kept by fit), removed / kept |
|---|---|---|---|
| good (40) | 0.7 % | 8.0 % / 0.6 % | 0.158 / 0.032 m |
| good_rough (10) | 2.6 % | 14.2 % / 3.2 % | 0.241 / 0.049 m |
| maybe_not_SPK (3) | 16 % | 65 % / 7.4 % | 1.17 / 0.078 m |
| problem_SPK (17 in fits) | 53 % | 44 % / 15 % | 1.07 / 0.12 m |

So the areas the images contradict are where the fits found the errors, also inside
"good" strips.  With the looser class-4 cut (U >= 0.005) the edit removes a median 2.8 %
(max 18 %) of a good strip; with U >= 0.02 a median 1 % (max 7 %): `edit.py` defaults to
0.02 (`--min_unconfirmed`).  `results/test_kept_U_thresholds.csv`.

### Relaxed filter (keeping rough surfaces)

With the 16m_E4 defaults dem_filter removes little real rough surface in these strips:
only ~2 % of the cells that the images confirm (class 2) were removed by the filter.
The relaxed filter (slope_tol 4, smooth_tol 40, R_tol 200, 2-pixel matchtag erosion)
adds 0.1-1.3 % more cells, of which only 7-12 % are confirmed and 28-44 % contradicted, so the combination adds 0.01-0.26 % (`results/test_edit_round1.csv`,
19 strips, run with the older class-4 cut).  At 16 m the gain from Landsat is in the
removals; the restore step may matter more for the 32-m filter (slope_tol 0.5, R_tol 10)
or for heavily crevassed margins, untested.

### Examples
- `plots/example_problem_WV02_20200212_seg11.png`: the north half of the strip is DEM
  artefacts (contradicted), the south half matches the images; the list marks the whole
  segment bad.
- `plots/example_good_crosson_rifts_WV01_20191116.png`: Crosson rifts, F ~1 (predicted and
  observed shading agree in detail); the smooth shelf has F ~0 but no signal (DEM striping
  below the amplitude cut).

### Caveats
- Landsat 8/9 only (2013-); sun elevation > 8 deg; images within +-45 days.
- Shading constrains the slope along the sun azimuth only (grid azimuth ~310-320 deg at
  SPK for all images), and nothing about offsets or long wavelengths.
- Thin uniform haze, persistent fog and surface change between the DEM and the images
  (calving, rift opening) can look like DEM errors; the pairwise cloud screen and the noise
  weighting reduce but do not remove this.
- The check is slow: downloads (landsatlook) and ~100 s per 15-km tile; a strip takes
  7-50 min.  A scene-level cache would be the next speed-up for production.

## USGS ERS login incident (2026-11-14 ~17:10)

The first 10-worker test batch logged in to ERS from every process, with forced re-logins
on every failed read.  ERS refused the logins within minutes (every task: "no EROS_SSO
cookie"; one later manual attempt: HTTP 401 "Unable to authenticate").  Batch stopped.
The login is now shared by all processes through `~/.cache/ldcheck/ers_cookie.json`
(mode 600, flock), logins are >= 2 min apart, a failed login blocks further attempts for
30 min (`ERSLoginBlocked`, the DEM ends with status `login_blocked`), and at most 5
downloads run at once (`download_slot`).

## Running it

    python -m ldcheck.check_dem <strip name or 2-m dem file> --out_dir <qa dir> [--save_debug] [--roi x0,y0,x1,y1]
    python -m ldcheck.flag <qa dir>/<strip>/<strip>_ldcheck.json [--why]      # bad / ok / unknown
    python -m ldcheck.edit <2-m dem> <ldcheck.tif> <out_dem_filt.tif> --res 16 --standard_file <16m file>

`scripts/get_pgc_2res_ldcheck.sh URL --res 16` is get_pgc_2res.sh with the check added: it
makes the usual `16m/` output unchanged, plus `16m_ldcheck/<name>_dem_filt.tif` (standard
samples minus inconsistent areas, plus relaxed samples in consistent areas) and appends
whole-DEM failures to `16m_ldcheck/problem_DEMs_ldcheck.txt` (read_DEM_data reads
`problem_DEMs*.txt` next to the GeoIndex).  Tested end-to-end on
WV03_20200320 seg2 in the scratchpad (standard 12438 cells, 8081 removed, DEM listed).

Batches: `scripts/make_check_queue.py set.csv qa_dir queue.txt` -> pboss (tmux `ldcheck`).
Each queue line sets NUMBA/OMP threads to 4.

## Performance

- Processing: ~1-2 min per 15-km tile with 4 threads (illumination with shadows for 6-8 sun
  positions, registration, cloud screen); ~100 s per tile with 20 workers on this machine.
  Three early bottlenecks are fixed: reading the whole strip bitmask for every tile, spline-
  filtering the whole strip chip for every image sample (now cropped to the tile), and
  full-resolution sun-angle and cloud maps (now computed sun position, 60-m cloud mask).
- Downloads: GDAL reads COG tiles one request at a time (0.7 s latency each), so a chip
  read was latency-bound (~0.3 MB/s); `GDAL_NUM_THREADS=8` makes GDAL fetch tiles in
  parallel (60x60 km pan window: 34 s -> 7 s).  5 download slots shared by all workers.  Pan chips are 1-30 MB; the cloud
  bands (SWIR 1.6, cirrus) are read at 60 m from the COG overviews; the sun angles are
  computed (`sun.py`, NOAA algorithm; agrees with the SAA/SZA bands to < 0.05 deg), so the
  angle bands are no longer downloaded; the snow index uses the pan band (agrees with the
  green-band index on 99.4-100 % of pixels).  A strip needs ~1.5 min of downloads.
- For production over a whole region, a scene-level cache (each scene cut once to the
  region and shared by all strips) would cut downloads by the number of strips per scene.
