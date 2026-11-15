#! /usr/bin/env python
"""
altimetryFit residuals of the fit points that the Landsat edit would remove (class 4 with
unconfirmed amplitude >= U_min) vs the points it keeps, by test set.

usage: fit_residuals_removed.py fit_points.h5 merged.csv qa_dir out.csv [U_min]
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
from osgeo import gdal
gdal.UseExceptions()


def main():
    fp, merged, qa, out_csv = sys.argv[1:5]
    umin = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
    m = pd.read_csv(merged).set_index('DEM')
    rows = []
    with h5py.File(fp, 'r') as h:
        for base in h.keys():
            if base not in m.index or m.loc[base, 'status'] != 'done':
                continue
            g = h[base]
            x, y, r, ed, w = (g[k][:] for k in ('x', 'y', 'r', 'edited', 'weight'))
            ds = gdal.Open(os.path.join(qa, base, base + '_ldcheck.tif'))
            gt = ds.GetGeoTransform()
            cls = ds.GetRasterBand(1).ReadAsArray()
            U = ds.GetRasterBand(7).ReadAsArray()
            c = ((x - gt[0]) / gt[1]).astype(int)
            rr = ((y - gt[3]) / gt[5]).astype(int)
            ins = (c >= 0) & (c < cls.shape[1]) & (rr >= 0) & (rr < cls.shape[0])
            rem = np.zeros(x.size, bool)
            rem[ins] = (cls[rr[ins], c[ins]] == 4) & (U[rr[ins], c[ins]] >= umin)
            for name, sel in (('removed', rem), ('kept', ~rem)):
                if w[sel].sum() == 0:
                    continue
                k = sel & (ed == 0)
                rows.append(dict(DEM=base, set=m.loc[base, 'set'], group=name, N=w[sel].sum(),
                                 N_edit=w[sel & (ed > 0)].sum(),
                                 sum_abs_r=np.sum(w[k] * np.abs(r[k])), N_kept=w[k].sum()))
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    a = d.groupby(['set', 'group'])[['N', 'N_edit', 'sum_abs_r', 'N_kept']].sum()
    a['frac_of_points'] = a.N / a.groupby(level=0).N.transform('sum')
    a['edit_frac'] = a.N_edit / a.N
    a['mean_abs_r_kept'] = a.sum_abs_r / a.N_kept
    a['n_DEMs'] = d.groupby(['set', 'group']).DEM.nunique()
    pd.set_option('display.width', 200)
    print(a[['n_DEMs', 'N', 'frac_of_points', 'edit_frac', 'mean_abs_r_kept']].round(4).to_string())


if __name__ == '__main__':
    main()
