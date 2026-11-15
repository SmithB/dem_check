#! /usr/bin/env python
"""
Build the test sets for the Landsat DEM check.

  problem_SPK   : every DEM in the SPK problem lists (16m_E4/problem_DEMs.txt, more_problem_DEMs)
  maybe_not_SPK : 16m_E4/maybe_not_problem_DEMs.txt
  problem_other : a random sample from the problem_DEMs*.txt lists of the other regions
                  (fullres/*/16m*/, season_covers_v2/*/, REMA year dirs), Landsat-8 era only
  good          : DEMs with small residuals in the SPK hybrid fits (2019.5-2021.5)
  good_rough    : DEMs used in the Crosson Lagrangian fits (rifted shelf), judged fine in the
                  Crosson DEM QA (edits follow rift walls), with moderate residuals

Writes test_sets/test_set.csv (DEM, set, source_list, comment).
"""
import os, re, glob, sys
import numpy as np
import pandas as pd
from ldcheck import dem

REMA = '/Volumes/ice1/ben/REMA'
WORK = os.environ.get('LDCHECK_WORK', '/home/ben/DEM_regions/SPK/landsat_DEM_check')
rng = np.random.default_rng(20261114)
name_re = re.compile(r'(SETSM_s2s041_\w\w\d\d_(\d{8})_[0-9A-F]+_[0-9A-F]+_\d+m_(?:lsf_)?seg\d+)')


def read_list(f):
    out = []
    with open(f, errors='ignore') as fh:
        for line in fh:
            line = line.replace('\r', '')
            m = name_re.search(line)
            if m is None:
                continue
            comment = line.split('#', 1)[1].strip() if '#' in line else ''
            out.append((dem.strip_base(m.group(1)), int(m.group(2)[:4]), int(m.group(2)[4:6]), comment))
    return out


rows = []
spk = f'{REMA}/fullres/SPK'
for f, setname in [(f'{spk}/16m_E4/problem_DEMs.txt', 'problem_SPK'), (f'{spk}/more_problem_DEMs', 'problem_SPK'),
                   (f'{spk}/16m_E4/maybe_not_problem_DEMs.txt', 'maybe_not_SPK')]:
    for base, y, m, c in read_list(f):
        rows.append(dict(DEM=base, set=setname, source_list=f, comment=c))
spk_names = {r['DEM'] for r in rows}

# other regions
other_lists = [f for f in glob.glob(f'{REMA}/**/problem_DEMs*.txt', recursive=True)
               if '/SPK/' not in f and 'backup' not in f and not f.endswith('~')]
cands = {}
for f in other_lists:
    for base, y, m, c in read_list(f):
        if base in spk_names or y < 2013 or (y == 2013 and m < 6):
            continue
        cands.setdefault(base, (f, c))
names = sorted(cands)
have = [n for n in names if dem.find_2m_files(n) is not None]
print(f'other-region problem DEMs: {len(names)}, with local 2-m files: {len(have)}')
# sample, stratified by source list
by_list = {}
for n in have:
    by_list.setdefault(cands[n][0], []).append(n)
pick = []
for f, ns in sorted(by_list.items()):
    k = min(len(ns), 4)
    pick += list(rng.choice(ns, k, replace=False))
pick = sorted(set(pick))
for n in pick:
    rows.append(dict(DEM=n, set='problem_other', source_list=cands[n][0], comment=cands[n][1]))

# good DEMs from the fits
fr = pd.read_csv(os.path.join(WORK, 'results', 'fit_DEM_residuals.csv'))
fr['base'] = [dem.strip_base(d) for d in fr.DEM]
fr['year'] = fr.base.str.extract(r'_(\d{4})\d{4}_')[0].astype(int)
hyb = fr[fr.fit_dir.str.contains('hybrid')]
g = hyb.groupby('base').agg(N=('N', 'sum'), edit=('edit_frac', 'max'), RDE=('RDE_r', 'max'), year=('year', 'first'))
good = g[(g.edit <= 0.02) & (g.RDE <= 0.05) & (g.N >= 5e5) & ~g.index.isin(spk_names)]
print(f'good candidates: {len(good)}')
pick = []
for y, gg in good.groupby('year'):
    pick += list(rng.choice(gg.index.values, min(len(gg), 8), replace=False))
for n in sorted(pick):
    rows.append(dict(DEM=n, set='good', source_list='fits/AA_200m_25m_qyr_hybrid_jitter_stripe',
                     comment=f"N={int(good.loc[n, 'N'])} edit={good.loc[n, 'edit']:.3f} RDE={good.loc[n, 'RDE']:.3f}"))
cro = fr[fr.fit_dir.str.contains('lag_crosson')]
cro = cro[(cro.N >= 3e5) & (cro.RDE_r <= 0.06) & ~cro.base.isin(spk_names) & ~cro.base.isin(pick)]
print(f'good_rough candidates: {len(cro)}')
for _, rr in cro.sample(min(len(cro), 10), random_state=1).iterrows():
    rows.append(dict(DEM=rr.base, set='good_rough', source_list='fits/AA_200m_25m_qyr_lag_crosson/2022.0/prelim',
                     comment=f"N={int(rr.N)} edit={rr.edit_frac:.3f} RDE={rr.RDE_r:.3f}"))
df = pd.DataFrame(rows).drop_duplicates('DEM')
df['date'] = df.DEM.str.extract(r'_(\d{8})_')[0]
df['has_2m'] = [dem.find_2m_files(n) is not None for n in df.DEM]
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'test_sets', 'test_set.csv'), index=False)
print(df.groupby('set').agg(n=('DEM', 'count'), has_2m=('has_2m', 'sum')))
