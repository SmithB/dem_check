#! /usr/bin/env python
"""
Collect the fit residuals of the test DEMs from altimetryFit tile outputs.

usage: collect_fit_points.py test_set.csv out.h5 fit_dir1 [fit_dir2 ...]

For every DEM of the test set, writes a group with x, y, r (= z - z_est), edited
(three_sigma_edit == False), sigma and the fit-directory index, from all tiles.
Every edited point is kept; kept points are subsampled to MAX_KEPT per DEM per tile.
"""
import os
import sys
import glob
import numpy as np
import h5py
import pandas as pd
from multiprocessing import Pool

from ldcheck import dem

WANT = None
MAX_KEPT = 3000


def init(want):
    global WANT
    WANT = want


def scan(args):
    fname, ifit = args
    out = {}
    try:
        with h5py.File(fname, 'r') as h:
            sensors = {int(k.split('_')[1]): v for k, v in h['meta/sensors'].attrs.items()}
            hits = {s: dem.strip_base(v) for s, v in sensors.items() if v.endswith('.tif')}
            hits = {s: b for s, b in hits.items() if b in WANT}
            if not hits:
                return out
            s = h['data/sensor'][:]
            sel = np.isin(s, list(hits))
            if not sel.any():
                return out
            x, y, z, ze = (h[f'data/{k}'][:][sel] for k in ('x', 'y', 'z', 'z_est'))
            keep = np.asarray(h['data/three_sigma_edit'][:][sel]).astype(bool)
            sig = h['data/sigma'][:][sel]
            ss = s[sel]
    except Exception as e:
        print(fname, e, file=sys.stderr)
        return out
    rng = np.random.default_rng(abs(hash(os.path.basename(fname))) % 2**32)
    for sn, b in hits.items():
        ii = ss == sn
        # keep every edited point and up to MAX_KEPT kept points per tile
        jj = np.flatnonzero(ii & keep)
        w = np.ones(ss.size)
        if jj.size > MAX_KEPT:
            ii = ii & ~keep
            ii[rng.choice(jj, MAX_KEPT, replace=False)] = True
            w[keep] = jj.size / MAX_KEPT
        if ii.any():
            out[b] = np.c_[x[ii], y[ii], z[ii] - ze[ii], ~keep[ii], sig[ii], np.full(ii.sum(), ifit), w[ii]]
    return out


def main():
    csv, out_file = sys.argv[1:3]
    fit_dirs = sys.argv[3:]
    want = set(pd.read_csv(csv).DEM)
    tasks = []
    for i, fd in enumerate(fit_dirs):
        tasks += [(f, i) for f in sorted(glob.glob(os.path.join(fd, 'E*_N*.h5')))]
    acc = {}
    with Pool(16, initializer=init, initargs=(want,)) as p:
        for res in p.imap_unordered(scan, tasks, chunksize=8):
            for b, a in res.items():
                acc.setdefault(b, []).append(a)
    with h5py.File(out_file, 'w') as h:
        h.attrs['fit_dirs'] = fit_dirs
        for b, parts in acc.items():
            a = np.concatenate(parts)
            g = h.create_group(b)
            for j, k in enumerate(['x', 'y', 'r', 'edited', 'sigma', 'fit', 'weight']):
                g.create_dataset(k, data=a[:, j], compression='gzip')
    print(f'{len(acc)} DEMs with fit points written to {out_file}')


if __name__ == '__main__':
    main()
