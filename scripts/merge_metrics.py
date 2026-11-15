#! /usr/bin/env python
"""Merge test_summary.csv with test_filter_vs_class.csv and compute AUCs for the kept-area metrics.

usage: merge_metrics.py results/test  (reads <p>_summary.csv, <p>_filter_vs_class.csv; writes <p>_merged.csv, <p>_auc_kept.csv)
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from summarize_tests import auc

p = sys.argv[1]
s = pd.read_csv(p + '_summary.csv')
f = pd.read_csv(p + '_filter_vs_class.csv').drop(columns=['set'])
d = s.merge(f, on='DEM', how='left')
d['res'] = d.file16.str.extract(r'_(\d+)m_')[0]
d.to_csv(p + '_merged.csv', index=False)
ok = d[d.status == 'done']
good = ok[ok.set.isin(['good', 'good_rough'])]
bad = ok[ok.set.isin(['problem_SPK', 'problem_other'])]
rows = []
for m in ['kept_frac_incons', 'kept_incons_of_signal', 'frac_inconsistent', 'frac_inconsistent_of_signal',
          'U_frac_gt_0.01', 'one_minus_median_F']:
    g, b = good[m].dropna(), bad[m].dropna()
    rows.append(dict(metric=m, n_good=len(g), n_bad=len(b), AUC=auc(b, g),
                     AUC_SPK=auc(ok[ok.set == 'problem_SPK'][m].dropna(), g),
                     AUC_other=auc(ok[ok.set == 'problem_other'][m].dropna(), g),
                     good_max=g.max(), good_p90=np.percentile(g, 90) if len(g) else np.nan,
                     bad_p10=np.percentile(b, 10) if len(b) else np.nan, bad_median=b.median()))
r = pd.DataFrame(rows)
r.to_csv(p + '_auc_kept.csv', index=False)
pd.set_option('display.width', 200)
print(r.round(3).to_string())
for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    print(f'kept_frac_incons >= {t}: good flagged {np.mean(good.kept_frac_incons >= t):.2f} '
          f'({(good.kept_frac_incons >= t).sum()}/{good.kept_frac_incons.notna().sum()}), '
          f'problem_SPK {np.mean(ok[ok.set == "problem_SPK"].kept_frac_incons >= t):.2f}, '
          f'problem_other {np.mean(ok[ok.set == "problem_other"].kept_frac_incons >= t):.2f}')
