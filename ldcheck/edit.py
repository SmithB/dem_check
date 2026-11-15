"""
Use the Landsat check to edit the filtered (16-m / 32-m) DEMs.

Two filtered versions of a strip are combined with the ldcheck classification:
  standard : dem_filter.py with the usual arguments (e.g. filter_defaults_16m_E4.txt)
  relaxed  : dem_filter.py with looser roughness / slope limits and a smaller
             matchtag erosion, so that real rough surfaces (crevasses, rifts) survive
The output keeps
  - standard samples, except where the Landsat check says the DEM is inconsistent
    with the images (class 4 with unconfirmed amplitude >= min_unconfirmed, default
    0.02, i.e. about a 2 % slope error along the sun direction; dilated by dilate_m);
  - relaxed samples that the standard filter removed, where the Landsat check says
    the DEM is consistent with the images (class 2).
The output has the same two bands as dem_filter (z, error estimate) on the same grid.
"""
import os
import re
import json
import shutil
import argparse
import numpy as np
import scipy.ndimage as snd
from osgeo import gdal, gdalconst

gdal.UseExceptions()

RELAXED_ARGS = {16: ['--smooth_scale=8', '--smooth_tol=40', '--slope_tol=4', '--R_tol=200',
                     '--target_resolution=16', '--geolocation_error=0', '--pgc_masks', '--error_RMS_scale=4'],
                32: ['--smooth_scale=16', '--smooth_tol=40', '--slope_tol=4', '--R_tol=200',
                     '--target_resolution=32', '--geolocation_error=0', '--pgc_masks']}


def read_args_file(fname):
    with open(fname) as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith('#')]


def run_dem_filter(dem_file, out_file, args, matchtag_erode=None, pgc_url_file=None):
    """
    Run altimetryFit's dem_filter.filter_dem in-process.  With matchtag_erode (2-m
    pixels), the matchtag erosion (normally the decimation factor, 8 at 16 m) is
    replaced by an erosion of that size; dem_filter.py itself is not changed.
    Note: dem_filter finds the strip name with a greedy regex over the whole path, so
    no directory in the path of dem_file may contain 'SETSM_'.
    """
    if 'SETSM_' in os.path.dirname(os.path.abspath(dem_file)):
        raise ValueError(f'dem_filter cannot find the pgc masks for {dem_file}: a directory name contains SETSM_')
    import scripts.dem_filter as dfm
    args = [a for a in args if not a.startswith('--pgc_url_file')]
    if pgc_url_file is not None:
        args = args + [f'--pgc_url_file={pgc_url_file}']
    elif '--pgc_masks' in args:
        args = args + ['--pgc_url_file=/Volumes/ice1/ben/REMA/index/all_REMA_urls.txt']
    import sys
    argv_save = sys.argv
    orig = dfm.mask_pgc
    try:
        sys.argv = ['dem_filter.py', dem_file, out_file] + list(args)
        if matchtag_erode is not None:
            def mask_pgc(this_bounds, mask, pgc_subs, dec):
                return orig(this_bounds, mask, pgc_subs, max(2, int(matchtag_erode)))
            dfm.mask_pgc = mask_pgc
        a = dfm.parse_input_args(sys.argv)
        dfm.filter_dem(a)
    finally:
        dfm.mask_pgc = orig
        sys.argv = argv_save
    return out_file


def crop_2m(files, bounds, out_dir):
    """Copy a window of the 2-m dem/matchtag/bitmask into out_dir (same names), for tests."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for k, f in files.items():
        dst = os.path.join(out_dir, os.path.basename(f))
        gdal.Translate(dst, f, options=gdal.TranslateOptions(projWin=[bounds[0], bounds[3], bounds[2], bounds[1]],
                                                             creationOptions=['COMPRESS=LZW', 'TILED=YES']))
        out[k] = dst
    return out


def _read(fname):
    ds = gdal.Open(fname)
    bands = [ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32) for i in range(ds.RasterCount)]
    return bands, ds.GetGeoTransform(), ds.GetProjection()


def class_on_grid(ldcheck_tif, gt, shape, dilate_m=30., min_unconfirmed=0.02):
    """
    ldcheck class (nearest neighbour) on a DEM grid.  Cells in class 4 whose
    unconfirmed amplitude is below min_unconfirmed are set to class 3 (marginal), so
    that only clearly contradicted areas are removed; the remaining class-4 areas are
    dilated by dilate_m.
    """
    ny, nx = shape
    x0, y1 = gt[0], gt[3]
    bounds = [x0, y1 + gt[5] * ny, x0 + gt[1] * nx, y1]
    src = gdal.Open(ldcheck_tif)
    names = {src.GetRasterBand(i + 1).GetDescription(): i + 1 for i in range(src.RasterCount)}
    out = {}
    for name in ('class', 'unconfirmed_amp'):
        ds = gdal.Warp('', ldcheck_tif, options=gdal.WarpOptions(
            format='MEM', outputBounds=bounds, width=nx, height=ny, resampleAlg='near',
            srcBands=[names[name]], dstBands=[1], srcNodata=np.nan, dstNodata=np.nan))
        out[name] = ds.GetRasterBand(1).ReadAsArray()
    cls = np.nan_to_num(out['class'], nan=0).astype(np.uint8)
    weak = (cls == 4) & ~(out['unconfirmed_amp'] >= min_unconfirmed)
    cls[weak] = 3
    if dilate_m > 0:
        n = int(np.ceil(dilate_m / abs(gt[1])))
        if n > 0:
            bad = snd.binary_dilation(cls == 4, structure=np.ones((2 * n + 1, 2 * n + 1), bool))
            cls[bad & (cls != 4)] = 4
    return cls


def combine(std_file, relaxed_file, ldcheck_tif, out_file, dilate_m=30., min_unconfirmed=0.02):
    """Combine standard and relaxed filtered DEMs with the ldcheck classification."""
    (zs, es), gt, proj = _read(std_file)
    have_rel = relaxed_file is not None and os.path.isfile(relaxed_file)
    if have_rel:
        (zr, er), gtr, _ = _read(relaxed_file)
        if gtr != gt or zr.shape != zs.shape:
            raise ValueError('standard and relaxed grids differ')
    cls = class_on_grid(ldcheck_tif, gt, zs.shape, dilate_m=dilate_m, min_unconfirmed=min_unconfirmed)
    keep_std = np.isfinite(zs) & (cls != 4)
    add = np.zeros(zs.shape, bool)
    if have_rel:
        add = ~np.isfinite(zs) & np.isfinite(zr) & (cls == 2)
    z = np.full(zs.shape, np.nan, np.float32)
    e = np.full(zs.shape, np.nan, np.float32)
    z[keep_std], e[keep_std] = zs[keep_std], es[keep_std]
    if have_rel:
        z[add], e[add] = zr[add], er[add]
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(out_file, zs.shape[1], zs.shape[0], 2, gdal.GDT_Float32,
                    options=['COMPRESS=LZW', 'TILED=YES', 'PREDICTOR=3'])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for i, b in enumerate((z, e)):
        ds.GetRasterBand(i + 1).WriteArray(b)
        ds.GetRasterBand(i + 1).SetNoDataValue(np.nan)
    ds.SetMetadataItem('ldcheck_file', os.path.basename(ldcheck_tif))
    ds.SetMetadataItem('ldcheck_standard_file', os.path.basename(std_file))
    if have_rel:
        ds.SetMetadataItem('ldcheck_relaxed_file', os.path.basename(relaxed_file))
    ds = None
    stats = dict(n_standard=int(np.isfinite(zs).sum()), n_removed=int((np.isfinite(zs) & (cls == 4)).sum()),
                 n_added=int(add.sum()), n_out=int(np.isfinite(z).sum()),
                 n_relaxed=int(np.isfinite(zr).sum()) if have_rel else 0,
                 n_relaxed_only=int((~np.isfinite(zs) & np.isfinite(zr)).sum()) if have_rel else 0,
                 n_relaxed_only_by_class={int(c): int((~np.isfinite(zs) & np.isfinite(zr) & (cls == c)).sum())
                                          for c in range(6)} if have_rel else {})
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description='edit a filtered strip DEM with the Landsat check')
    p.add_argument('dem_2m', help='2-m dem file (its matchtag and bitmask must be next to it)')
    p.add_argument('ldcheck_tif')
    p.add_argument('out_file')
    p.add_argument('--res', type=int, default=16)
    p.add_argument('--standard_file', help='existing standard filtered DEM (else it is made)')
    p.add_argument('--standard_args', default='/Volumes/ice1/ben/REMA/fullres/SPK/filter_defaults_16m_E4.txt')
    p.add_argument('--matchtag_erode', type=int, default=2)
    p.add_argument('--dilate_m', type=float, default=30.)
    p.add_argument('--min_unconfirmed', type=float, default=0.02,
                   help='remove class-4 cells only where the unconfirmed amplitude (mu units) is at least this')
    p.add_argument('--keep_intermediate', action='store_true')
    a = p.parse_args(argv)
    work = os.path.splitext(a.out_file)[0] + '_work'
    os.makedirs(work, exist_ok=True)
    std = a.standard_file
    if std is None:
        std = os.path.join(work, 'standard.tif')
        run_dem_filter(a.dem_2m, std, read_args_file(a.standard_args))
    rel = os.path.join(work, 'relaxed.tif')
    run_dem_filter(a.dem_2m, rel, RELAXED_ARGS[a.res], matchtag_erode=a.matchtag_erode)
    stats = combine(std, rel, a.ldcheck_tif, a.out_file, dilate_m=a.dilate_m, min_unconfirmed=a.min_unconfirmed)
    with open(os.path.splitext(a.out_file)[0] + '_edit.json', 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(stats)
    if not a.keep_intermediate:
        shutil.rmtree(work)


if __name__ == '__main__':
    main()
