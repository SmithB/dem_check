#! /usr/bin/env python
"""
Per-DEM residual statistics from altimetryFit tile outputs.

For every tile file in the given fit directories, read data/{sensor, z, z_est,
sigma, three_sigma_edit, x, y} and meta/sensors, and accumulate, for each DEM
(sensor names ending in _dem_filt.tif):
    N            number of points
    N_edit       number of points with three_sigma_edit == False
    sum_abs_r    sum of |z - z_est| over kept points  (for the mean |r|)
    r_kept       a subsample of kept residuals (for the median and RDE)
    tiles        number of tiles
The output is a csv with one row per DEM per fit directory.

usage: scan_fit_DEM_residuals.py out.csv fit_dir1 [fit_dir2 ...]
"""
import sys
import os
import glob
import numpy as np
import h5py
import pandas as pd
from multiprocessing import Pool

MAX_R_PER_TILE = 2000


def scan_tile(fname):
    rows = {}
    try:
        with h5py.File(fname, 'r') as h:
            sensors = {int(k.split('_')[1]): v for k, v in h['meta/sensors'].attrs.items()}
            s = h['data/sensor'][:]
            z = h['data/z'][:]
            ze = h['data/z_est'][:]
            keep = np.asarray(h['data/three_sigma_edit'][:]).astype(bool)
            sig = h['data/sigma'][:]
    except Exception as e:
        print(f'{fname}: {e}', file=sys.stderr)
        return rows
    r = z - ze
    rng = np.random.default_rng(abs(hash(os.path.basename(fname))) % 2**32)
    for ss in np.unique(s):
        name = sensors.get(int(ss), '')
        if not name.endswith('.tif'):
            continue
        ii = s == ss
        rk = r[ii & keep]
        if rk.size > MAX_R_PER_TILE:
            rk = rng.choice(rk, MAX_R_PER_TILE, replace=False)
        rows[name] = dict(N=int(ii.sum()), N_edit=int((ii & ~keep).sum()),
                          sum_abs_r=float(np.abs(r[ii & keep]).sum()),
                          sum_sigma=float(sig[ii].sum()),
                          r_kept=rk)
    return rows


def rde(x):
    if x.size < 10:
        return np.nan
    p = np.percentile(x, [16, 84])
    return (p[1] - p[0]) / 2


def main():
    out_file = sys.argv[1]
    fit_dirs = sys.argv[2:]
    out = []
    for fit_dir in fit_dirs:
        files = sorted(glob.glob(os.path.join(fit_dir, 'E*_N*.h5')))
        print(f'{fit_dir}: {len(files)} tiles', flush=True)
        acc = {}
        with Pool(16) as p:
            for rows in p.imap_unordered(scan_tile, files, chunksize=8):
                for name, d in rows.items():
                    a = acc.setdefault(name, dict(N=0, N_edit=0, sum_abs_r=0., sum_sigma=0., r=[], tiles=0))
                    a['N'] += d['N']
                    a['N_edit'] += d['N_edit']
                    a['sum_abs_r'] += d['sum_abs_r']
                    a['sum_sigma'] += d['sum_sigma']
                    a['r'].append(d['r_kept'])
                    a['tiles'] += 1
        for name, a in acc.items():
            r = np.concatenate(a['r']) if a['r'] else np.zeros(0)
            n_kept = a['N'] - a['N_edit']
            out.append(dict(DEM=name, fit_dir=fit_dir, tiles=a['tiles'], N=a['N'],
                            edit_frac=a['N_edit'] / max(a['N'], 1),
                            mean_abs_r=a['sum_abs_r'] / max(n_kept, 1),
                            median_r=np.median(r) if r.size else np.nan,
                            RDE_r=rde(r),
                            mean_sigma=a['sum_sigma'] / max(a['N'], 1)))
    pd.DataFrame(out).to_csv(out_file, index=False)
    print(f'wrote {len(out)} rows to {out_file}')


if __name__ == '__main__':
    main()
