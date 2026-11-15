#! /usr/bin/env python
"""
Compare the standard dem_filter output with the Landsat classification.

usage: compare_filter_class.py summary.csv qa_dir out.csv

For each checked DEM with a standard filtered 16-m file (fullres/*/16m*/), count
the 16-m cells by Landsat class, split into cells the filter kept and cells it
removed (DEM valid at 15 m, but no filtered value).  Removed cells in class 2
(consistent) are rough surfaces that the images confirm; kept cells in class 4
are the ones the Landsat edit would remove.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from osgeo import gdal

from ldcheck import dem

gdal.UseExceptions()
F16_DIRS = sorted(glob.glob('/Volumes/ice1/ben/REMA/fullres/*/16m*')) + \
    sorted(glob.glob('/Volumes/ice1/ben/REMA/season_covers_v2/*/16m'))
F32_DIRS = sorted(glob.glob('/Volumes/ice1/ben/REMA/fullres/*/32m')) + \
    sorted(glob.glob('/Volumes/ice1/ben/REMA/season_covers_v2/*/32m')) + \
    sorted(glob.glob('/Volumes/ice1/ben/REMA/20??_*'))


def find_16m(base):
    """The standard filtered DEM: 16 m if there is one, else 32 m."""
    for res, dirs in ((16, F16_DIRS), (32, F32_DIRS)):
        name = base.replace('_2m_', f'_{res}m_') + '_dem_filt.tif'
        for d in dirs:
            f = os.path.join(d, name)
            if os.path.isfile(f):
                return f
    return None


def on_grid(tif, band, gt, shape, alg='near'):
    ny, nx = shape
    b = [gt[0], gt[3] + gt[5] * ny, gt[0] + gt[1] * nx, gt[3]]
    ds = gdal.Warp('', tif, options=gdal.WarpOptions(format='MEM', outputBounds=b, width=nx, height=ny,
                                                     resampleAlg=alg, srcBands=[band], dstBands=[1]))
    return ds.GetRasterBand(1).ReadAsArray()


def main():
    summ, qa_dir, out_csv = sys.argv[1:4]
    d = pd.read_csv(summ)
    rows = []
    for _, r in d[d.status == 'done'].iterrows():
        f16 = find_16m(r.DEM)
        if f16 is None:
            continue
        ds = gdal.Open(f16)
        gt = ds.GetGeoTransform()
        z = ds.GetRasterBand(1).ReadAsArray()
        tif = os.path.join(qa_dir, r.DEM, r.DEM + '_ldcheck.tif')
        cls = on_grid(tif, 1, gt, z.shape)
        vf = on_grid(tif, 9, gt, z.shape, alg='bilinear')   # band 9: DEM_valid_frac
        valid = np.isfinite(vf) & (vf >= 0.9) & (cls > 0)
        kept = np.isfinite(z) & valid
        rem = valid & ~np.isfinite(z)
        row = dict(DEM=r.DEM, set=r.set, file16=f16, n_valid=int(valid.sum()), n_kept=int(kept.sum()),
                   n_removed=int(rem.sum()))
        for c, name in [(1, 'nosig'), (2, 'cons'), (3, 'marg'), (4, 'incons'), (5, 'imgonly')]:
            row[f'kept_{name}'] = int((kept & (cls == c)).sum())
            row[f'removed_{name}'] = int((rem & (cls == c)).sum())
        sig_k = row['kept_cons'] + row['kept_marg'] + row['kept_incons']
        row['kept_frac_signal'] = sig_k / max(row['n_kept'], 1)
        row['kept_incons_of_signal'] = row['kept_incons'] / sig_k if sig_k else np.nan
        row['kept_frac_incons'] = row['kept_incons'] / max(row['n_kept'], 1)
        sig_r = row['removed_cons'] + row['removed_marg'] + row['removed_incons']
        row['removed_cons_of_signal'] = row['removed_cons'] / sig_r if sig_r else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    g = out.groupby('set')[[c for c in out.columns if c.startswith(('n_', 'kept_', 'removed_'))]].sum()
    print(g.T)


if __name__ == '__main__':
    main()
