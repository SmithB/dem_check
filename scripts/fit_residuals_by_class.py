#! /usr/bin/env python
"""
altimetryFit residuals of the test DEMs, split by Landsat class.

usage: fit_residuals_by_class.py fit_points.h5 summary.csv qa_dir out.csv

For each DEM with fit points and a finished check, the class (and F) is sampled
at each fit point (nearest 15-m pixel).  Per DEM and class: the (weighted) number
of points, the fraction edited by the fit, the robust spread (RDE) and mean |r|
of the kept residuals.
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
from osgeo import gdal

gdal.UseExceptions()


def wpct(x, w, q):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return np.interp(np.array(q) / 100 * cw[-1], cw, x[i])


def main():
    fp, summ, qa_dir, out_csv = sys.argv[1:5]
    s = pd.read_csv(summ).set_index('DEM')
    rows = []
    with h5py.File(fp, 'r') as h:
        for base in h.keys():
            if base not in s.index or s.loc[base, 'status'] != 'done':
                continue
            g = h[base]
            x, y, r, ed, w = (g[k][:] for k in ('x', 'y', 'r', 'edited', 'weight'))
            ds = gdal.Open(os.path.join(qa_dir, base, base + '_ldcheck.tif'))
            gt = ds.GetGeoTransform()
            cls = ds.GetRasterBand(1).ReadAsArray()
            col = ((x - gt[0]) / gt[1]).astype(int)
            row = ((y - gt[3]) / gt[5]).astype(int)
            inside = (col >= 0) & (col < cls.shape[1]) & (row >= 0) & (row < cls.shape[0])
            c = np.zeros(x.size, int)
            c[inside] = cls[row[inside], col[inside]]
            for k in range(6):
                ii = c == k
                if ii.sum() < 20:
                    continue
                kept = ii & (ed == 0)
                p16, p84 = wpct(r[kept], w[kept], [16, 84]) if kept.sum() > 10 else (np.nan, np.nan)
                rows.append(dict(DEM=base, set=s.loc[base, 'set'], cls=k, N=float(w[ii].sum()),
                                 edit_frac=float(w[ii & (ed > 0)].sum() / w[ii].sum()),
                                 RDE=(p84 - p16) / 2, mean_abs_r=float(np.sum(w[kept] * np.abs(r[kept])) / np.sum(w[kept]))
                                 if kept.any() else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    # pooled by set and class
    agg = d.assign(NE=d.N * d.edit_frac).groupby(['set', 'cls']).agg(N=('N', 'sum'), NE=('NE', 'sum'),
                                                                       RDE_median=('RDE', 'median'),
                                                                       mean_abs_r_median=('mean_abs_r', 'median'),
                                                                       n_DEMs=('DEM', 'count'))
    agg['edit_frac'] = agg.NE / agg.N
    print(agg.drop(columns='NE').round(3).to_string())


if __name__ == '__main__':
    main()
