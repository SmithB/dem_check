#! /usr/bin/env python
"""
Inventory of the REMA strip DEMs that have been downloaded and processed under the REMA
tree: every 2-m strip file (dem, matchtag, bitmask) and every dem_filter output
(*_dem_filt.tif), with the directory it is in.  get_pgc_2res.sh deletes the 2-m files
after filtering, so for most strips the filtered files are the only record.

Skipped: fit outputs (fits/, jitter_fits/), par_run*, velocity and xy01_grids directories,
thumbs, meta (PGC json), index, masks, mosaics, unity, landsat_DEM_check.

usage: inventory_rema.py [REMA_root] [out_dir]
writes
  rema_files.csv.gz : one row per file: strip, res, kind, dir, collection, area, subdir,
                      is_link, link_ok, size_bytes, mtime
  rema_strips.csv   : one row per strip segment (2-m base name): which 2-m files exist, and
                      the directories that hold 16-m / 32-m / other filtered versions
Collections: fullres (area = region), season_covers_v2 (area = season), annual_covers_v1
(the season directories at the top of the tree, e.g. 2018_late), other.
"""
import os
import re
import sys
import csv
import gzip
import time
from collections import defaultdict

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/ice1/ben/REMA')
WORK = os.environ.get('LDCHECK_WORK', '/home/ben/DEM_regions/SPK/landsat_DEM_check')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, 'handoff_lists')
SKIP_TOP = {'landsat_DEM_check', 'index', 'masks', 'mosaics', 'vel_est', 'xy01_grids', 'unity', 'meta', 'mask_checks'}
SKIP_ANY = {'fits', 'jitter_fits', '.ipynb_checkpoints', 'xy01_grids', 'velocity', 'Velocity', 'thumbs',
            'shifted_thumbs', 'meta', 'SPK_CS2_grids', 'fft_128px', 'fft_32px'}
NAME = re.compile(r'^(SETSM_s2s\d+_\w\w\d\d_\d{8}_[0-9A-F]+_[0-9A-F]+)_(\d+)m_(lsf_)?seg(\d+)_(dem|matchtag|bitmask|dem_filt)\.tif$')
SEASON = re.compile(r'^\d{4}_(early|late)$')


def classify(reldir):
    p = reldir.split(os.sep)
    if p[0] == 'fullres' and len(p) >= 2:
        return 'fullres', p[1], '/'.join(p[2:])
    if p[0] == 'season_covers_v2':
        if len(p) >= 2 and SEASON.match(p[1]):
            return 'season_covers_v2', p[1], '/'.join(p[2:])
        return 'season_covers_v2', '', '/'.join(p[1:])
    if SEASON.match(p[0]):
        return 'annual_covers_v1', p[0], '/'.join(p[1:])
    return 'other', '', reldir


def main():
    t0 = time.time()
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        reldir = os.path.relpath(dirpath, ROOT)
        if reldir == '.':
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOP and not d.startswith('par_run')]
        else:
            dirnames[:] = [d for d in dirnames if d not in SKIP_ANY and not d.startswith('par_run')]
        coll = area = sub = None
        for f in filenames:
            m = NAME.match(f)
            if m is None:
                continue
            if coll is None:
                coll, area, sub = classify(reldir)
            path = os.path.join(dirpath, f)
            is_link = os.path.islink(path)
            try:
                st = os.stat(path)
                ok, size, mt = True, st.st_size, time.strftime('%Y-%m-%d', time.localtime(st.st_mtime))
            except OSError:
                ok, size, mt = False, '', ''
            rows.append(dict(strip=f'{m.group(1)}_2m_{m.group(3) or ""}seg{m.group(4)}', res=int(m.group(2)),
                             kind=m.group(5), dir=reldir, collection=coll, area=area, subdir=sub,
                             is_link=is_link, link_ok=ok if is_link else '', size_bytes=size, mtime=mt))
        if len(rows) and len(rows) % 20000 < len(filenames):
            print(f'{time.time() - t0:.0f} s: {len(rows)} files ({reldir})', flush=True)
    os.makedirs(OUT, exist_ok=True)
    rows.sort(key=lambda r: (r['strip'], r['res'], r['kind'], r['dir']))
    cols = ['strip', 'res', 'kind', 'dir', 'collection', 'area', 'subdir', 'is_link', 'link_ok', 'size_bytes', 'mtime']
    with gzip.open(os.path.join(OUT, 'rema_files.csv.gz'), 'wt', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r['strip']].append(r)
    srows = []
    for s, rr in sorted(by.items()):
        two = defaultdict(set)
        filt = defaultdict(set)
        for r in rr:
            usable = (not r['is_link']) or r['link_ok']
            if r['res'] == 2 and usable:
                two[r['dir']].add(r['kind'])
            elif r['kind'] == 'dem_filt' and usable:
                filt[r['res']].add(r['dir'])
        complete_2m = sorted(d for d, k in two.items() if {'dem', 'matchtag', 'bitmask'} <= k)
        srows.append(dict(
            strip=s, date=s.split('_')[3], sensor=s.split('_')[2],
            has_2m=bool(complete_2m), dirs_2m=';'.join(complete_2m),
            partial_2m=';'.join(sorted(d for d in two if d not in complete_2m)),
            n_16m=len(filt.get(16, ())), dirs_16m=';'.join(sorted(filt.get(16, ()))),
            n_32m=len(filt.get(32, ())), dirs_32m=';'.join(sorted(filt.get(32, ()))),
            other_res=';'.join(f'{k}m:{d}' for k in sorted(filt) if k not in (16, 32) for d in sorted(filt[k])),
            broken_links=sum(1 for r in rr if r['is_link'] and not r['link_ok'])))
    with open(os.path.join(OUT, 'rema_strips.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print(f'{time.time() - t0:.0f} s: {len(rows)} files, {len(srows)} strips; '
          f'{sum(r["has_2m"] for r in srows)} with complete 2-m files, '
          f'{sum(r["n_16m"] > 0 for r in srows)} with 16 m, {sum(r["n_32m"] > 0 for r in srows)} with 32 m')


if __name__ == '__main__':
    main()
