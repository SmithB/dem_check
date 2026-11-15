#! /usr/bin/env python
"""
Collect every DEM that has been marked as a problem anywhere under the REMA tree.

Sources (case-insensitive): every file whose name contains 'problem' (problem_DEMs*.txt,
problem_dems.txt, more_problem_DEMs*, maybe_not_problem_DEMs.txt, dated versions, editor
backups ending in '~' or wrapped in '#'), every text file in a directory whose name
contains 'problem' (problem_DEMs/, problem_DEM_backup/), and DEM files that were moved
into a directory called 'problems'.

usage: collect_problem_lists.py [REMA_root] [out_dir] [rema_strips.csv]
writes
  problem_DEMs_all_entries.csv : one row per listed entry (strip, source file, line, comment, ...)
  problem_DEMs_unique.csv      : one row per strip segment (or strip pattern), with all its
                                 sources and comments
With rema_strips.csv (from inventory_rema.py), pattern entries are expanded to the strip
segments on disk that they match (column 'matches').

Columns
  strip                 2-m base name (SETSM_..._2m_[lsf_]segN), whatever resolution was listed.
                        Entries that name a whole strip or use a wildcard ('..._32m', '..._*',
                        '..._*_lsf_seg3_...') get a pattern such as SETSM_..._2m_* (is_pattern)
  listed_res            resolution in the listed name (m); empty for patterns
  source                path of the list, relative to REMA_root
  kind                  list | maybe_not | compiled (in a problem_DEMs/ directory) | backup |
                        editor_backup ('~' or '#...#' files) | moved (file in a 'problems' directory)
  read_by_altimetryFit  the list's name matches 'problem_DEMs*.txt' and the directory has a
                        GeoIndex*.h5, so altimetryFit.read_DEM_data reads it for fits that use
                        that GeoIndex (lists with other names are used only if a fit passes
                        them explicitly, e.g. with problem_DEM_file)
  effective_in_fits     read_by_altimetryFit, and read_DEM_data's name conversion
                        (get_2m_filename) turns the entry into a '..._2m_..._dem.tif' name, which is
                        what it compares with.  Whole-strip and wildcard entries, and names
                        without the '_dem_filt.tif' / '_dem.tif' ending, are not effective.
  commented_out         the line starts with '#' (read_DEM_data ignores it)
"""
import os
import re
import sys
import csv
import fnmatch
from collections import defaultdict

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/ice1/ben/REMA')
WORK = os.environ.get('LDCHECK_WORK', '/home/ben/DEM_regions/SPK/landsat_DEM_check')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, 'handoff_lists')
STRIPS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, 'rema_strips.csv')
SKIP = {'landsat_DEM_check', '.ipynb_checkpoints', 'par_run', 'jitter_fits', 'fits', 'xy01_grids', 'vel_est', 'unity',
        'index'}
NAME = re.compile(r'(SETSM_s2s\d+_\w\w\d\d_\d{8}_[0-9A-F]+_[0-9A-F]+)_(\d+)m_(lsf_)?seg(\d+)')
STRIP = re.compile(r'(SETSM_s2s\d+_\w\w\d\d_\d{8}_[0-9A-F]+_[0-9A-F]+)(\S*)')
FIT_RE = re.compile(r'(.*)_\d+m_(.*)')


def fit_name(entry):
    """altimetryFit.read_DEM_data.get_2m_filename, or None where it raises."""
    m = FIT_RE.search(os.path.basename(entry))
    if m is None:
        return None
    return '_2m_'.join(m.groups()).replace('_filt.tif', '.tif')


def kind_of(rel, base, in_problem_dir):
    if base.endswith('~') or (base.startswith('#') and base.endswith('#')):
        return 'editor_backup'
    if 'backup' in rel.lower():
        return 'backup'
    if 'maybe_not' in base.lower():
        return 'maybe_not'
    if in_problem_dir and os.path.basename(os.path.dirname(rel)).lower() == 'problem_dems':
        return 'compiled'
    return 'list'


def parse_entry(text):
    """(strip, listed_res, is_pattern) for the DEM named in text, or None."""
    m = NAME.search(text)
    if m:
        return f'{m.group(1)}_2m_{m.group(3) or ""}seg{m.group(4)}', int(m.group(2)), False
    m = STRIP.search(text)
    if m is None:
        return None
    rest = m.group(2)
    seg = re.search(r'seg(\d+)', rest)
    if seg:
        return f'{m.group(1)}_2m_{"lsf_" if "lsf_" in rest else ""}seg{seg.group(1)}', '', True
    return f'{m.group(1)}_2m_*', '', True


def main():
    rows = []
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        reldir = os.path.relpath(dirpath, ROOT)
        dname = os.path.basename(dirpath).lower()
        in_problem_dir = 'problem' in dname
        has_gi = any(f.startswith('GeoIndex') and f.endswith('.h5') for f in filenames)
        if dname == 'problems':
            for f in sorted(filenames):
                p = parse_entry(f)
                if p:
                    rows.append(dict(strip=p[0], listed_res=p[1], is_pattern=p[2], listed_name=f,
                                     source=os.path.join(reldir, f), line=0, kind='moved',
                                     read_by_altimetryFit=False, effective_in_fits=False, commented_out=False,
                                     comment=''))
            continue
        for f in sorted(filenames):
            if 'problem' not in f.lower() and not (in_problem_dir and f.endswith('.txt')):
                continue
            if f.endswith(('.ipynb', '.py', '.sh', '.tif', '.h5')):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, ROOT)
            kind = kind_of(rel, f, in_problem_dir)
            read_by_fit = fnmatch.fnmatchcase(f, 'problem_DEMs*.txt') and has_gi
            n_files += 1
            with open(path, errors='replace') as fh:
                for i, line in enumerate(fh, 1):
                    line = line.replace('\r', '').rstrip('\n')
                    p = parse_entry(line)
                    if p is None:
                        continue
                    commented = line.lstrip().startswith('#')
                    comment = line.split('#', 1)[1].strip() if ('#' in line and not commented) else ''
                    entry = line.split('#')[0].strip() if not commented else line.strip()
                    fn = fit_name(entry) if not commented else None
                    rows.append(dict(strip=p[0], listed_res=p[1], is_pattern=p[2], listed_name=entry,
                                     source=rel, line=i, kind=kind, read_by_altimetryFit=read_by_fit,
                                     effective_in_fits=bool(read_by_fit and fn and fn.endswith('_dem.tif')
                                                            and '*' not in fn),
                                     commented_out=commented, comment=comment))
    os.makedirs(OUT, exist_ok=True)
    cols = ['strip', 'listed_res', 'is_pattern', 'kind', 'read_by_altimetryFit', 'effective_in_fits', 'commented_out',
            'source', 'line', 'listed_name', 'comment']
    rows.sort(key=lambda r: (r['strip'], r['source'], r['line']))
    with open(os.path.join(OUT, 'problem_DEMs_all_entries.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)

    on_disk = []
    if os.path.isfile(STRIPS):
        with open(STRIPS) as fh:
            on_disk = [r['strip'] for r in csv.DictReader(fh)]
    by = defaultdict(list)
    for r in rows:
        by[r['strip']].append(r)
    urows = []
    for s, rr in sorted(by.items()):
        live = [r for r in rr if not r['commented_out']]
        kinds = sorted({r['kind'] for r in live})
        comments = []
        for r in live:
            if r['comment'] and r['comment'] not in comments:
                comments.append(r['comment'])
        pattern = rr[0]['is_pattern']
        matches = ''
        if pattern and on_disk:
            pat = s.replace('_2m_*', '_2m_*seg*') if s.endswith('_2m_*') else s
            matches = ';'.join(x for x in on_disk if fnmatch.fnmatchcase(x, pat))
        urows.append(dict(
            strip=s, is_pattern=pattern, date=s.split('_')[3], sensor=s.split('_')[2],
            n_entries=len(live), kinds=';'.join(kinds),
            read_by_altimetryFit=any(r['read_by_altimetryFit'] for r in live),
            effective_in_fits=any(r['effective_in_fits'] for r in live),
            in_current_list=any(r['kind'] in ('list', 'compiled', 'moved') for r in live),
            maybe_not=any(r['kind'] == 'maybe_not' for r in live),
            only_in_backups=bool(live) and all(r['kind'] in ('backup', 'editor_backup') for r in live),
            listed_res=';'.join(str(x) for x in sorted({r['listed_res'] for r in live if r['listed_res'] != ''})),
            sources=';'.join(sorted({r['source'] for r in live})),
            comments=' | '.join(comments), matches=matches))
    with open(os.path.join(OUT, 'problem_DEMs_unique.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(urows[0].keys()))
        w.writeheader()
        w.writerows(urows)
    print(f'{n_files} list files, {len(rows)} entries, {len(urows)} strips or patterns '
          f'({sum(u["is_pattern"] for u in urows)} patterns)')
    for k in ('read_by_altimetryFit', 'effective_in_fits', 'in_current_list', 'maybe_not', 'only_in_backups'):
        print(f'  {k}: {sum(bool(u[k]) for u in urows)}')


if __name__ == '__main__':
    main()
