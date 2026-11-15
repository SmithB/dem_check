#! /usr/bin/env python
"""
Run the edit step (relaxed dem_filter + combination with the Landsat class) for the
checked test DEMs that have a standard 16-m file, and tabulate what changes.

usage: run_edit_tests.py summary.csv qa_dir out_dir out.csv [n_procs] [sets]
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ldcheck import dem, edit

F16_DIRS = sorted(glob.glob('/Volumes/ice1/ben/REMA/fullres/*/16m*'))


def find_16m(base):
    name = base.replace('_2m_', '_16m_') + '_dem_filt.tif'
    for d in F16_DIRS:
        f = os.path.join(d, name)
        if os.path.isfile(f):
            return f
    return None


def one(args):
    base, qa_dir, out_dir = args
    std = find_16m(base)
    files = dem.find_2m_files(base)
    if std is None or files is None:
        return dict(DEM=base, status='no_files')
    # (no 'SETSM_' in the directory name: dem_filter's get_pgc_masks finds the strip
    # name in the full path with a greedy regex)
    work = os.path.join(out_dir, base.replace('SETSM_', ''))
    os.makedirs(work, exist_ok=True)
    rel = os.path.join(work, 'relaxed_16m.tif')
    out = os.path.join(work, os.path.basename(std))
    try:
        if not os.path.isfile(rel):
            # dem_filter wants the matchtag/bitmask next to the dem: link them into the work dir
            for f in files.values():
                lnk = os.path.join(work, os.path.basename(f))
                if not os.path.exists(lnk):
                    os.symlink(f, lnk)
            edit.run_dem_filter(os.path.join(work, os.path.basename(files['dem'])), rel,
                                edit.RELAXED_ARGS[16], matchtag_erode=2)
        stats = edit.combine(std, rel, os.path.join(qa_dir, base, base + '_ldcheck.tif'), out)
    except Exception as e:
        return dict(DEM=base, status=f'error: {e}')
    with open(os.path.join(work, 'edit.json'), 'w') as fh:
        json.dump(stats, fh, indent=1)
    row = dict(DEM=base, status='done', standard_file=std)
    row.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
    for c, n in stats.get('n_relaxed_only_by_class', {}).items():
        row[f'relaxed_only_cls{c}'] = n
    return row


def main():
    summ, qa_dir, out_dir, out_csv = sys.argv[1:5]
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    sets = sys.argv[6].split(',') if len(sys.argv) > 6 else None
    d = pd.read_csv(summ)
    d = d[d.status == 'done']
    if sets:
        d = d[d.set.isin(sets)]
    with Pool(nproc) as p:
        rows = p.map(one, [(b, qa_dir, out_dir) for b in d.DEM], chunksize=1)
    out = pd.DataFrame(rows).merge(d[['DEM', 'set']], on='DEM')
    out.to_csv(out_csv, index=False)
    ok = out[out.status == 'done']
    g = ok.groupby('set')[[c for c in ok.columns if c.startswith('n_') or c.startswith('relaxed_only')]].sum()
    print(g.T)


if __name__ == '__main__':
    main()
