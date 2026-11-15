#! /usr/bin/env python
"""Kept-area fraction that is inconsistent with an unconfirmed amplitude above several thresholds."""
import sys, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from osgeo import gdal
gdal.UseExceptions()
QA = '/Volumes/ice1/ben/REMA/landsat_DEM_check/qa'
TH = [0.005, 0.01, 0.02, 0.05]


def one(args):
    base, f16 = args
    tif = os.path.join(QA, base, base + '_ldcheck.tif')
    ds = gdal.Open(f16)
    gt = ds.GetGeoTransform()
    z = ds.GetRasterBand(1).ReadAsArray()
    ny, nx = z.shape
    b = [gt[0], gt[3] + gt[5] * ny, gt[0] + gt[1] * nx, gt[3]]
    out = {}
    for band, name in ((1, 'cls'), (7, 'U'), (9, 'vf')):
        d = gdal.Warp('', tif, options=gdal.WarpOptions(format='MEM', outputBounds=b, width=nx, height=ny,
                                                        resampleAlg='near', srcBands=[band], dstBands=[1]))
        out[name] = d.GetRasterBand(1).ReadAsArray()
    kept = np.isfinite(z) & (out['vf'] >= 0.9) & (out['cls'] > 0)
    row = dict(DEM=base, n_kept=int(kept.sum()))
    for t in TH:
        row[f'kept_incons_U{t}'] = float((kept & (out['cls'] == 4) & (out['U'] >= t)).sum() / max(kept.sum(), 1))
    return row


def main():
  d = pd.read_csv('results/test_merged.csv')

  d = d[(d.status == 'done') & d.file16.notna()]
  with Pool(16) as p:
      rows = p.map(one, list(zip(d.DEM, d.file16)))
  r = pd.DataFrame(rows).merge(d[['DEM', 'set']], on='DEM')
  r.to_csv('results/test_kept_U_thresholds.csv', index=False)
  good = r[r.set.isin(['good', 'good_rough'])]
  for t in TH:
      c = f'kept_incons_U{t}'
      print(f'U >= {t}: good median {good[c].median():.4f} max {good[c].max():.3f} | ' +
            ' | '.join(f'{s} median {r[r.set == s][c].median():.3f}' for s in ['maybe_not_SPK', 'problem_SPK', 'problem_other']))


if __name__ == '__main__':
    main()
