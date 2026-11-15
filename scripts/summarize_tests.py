#! /usr/bin/env python
"""
Summarize the Landsat DEM-check test runs.

usage: summarize_tests.py test_set.csv qa_dir out_prefix

Writes <out_prefix>_summary.csv (one row per DEM), <out_prefix>_auc.csv (how well each
metric separates the good from the problem DEMs), and <out_prefix>_metrics.png.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ldcheck import dem

METRICS = ['frac_inconsistent_of_signal', 'frac_bad_of_signal', 'frac_inconsistent', 'U_frac_gt_0.01',
           'U_frac_gt_0.02', 'one_minus_median_F', 'one_minus_dem_peak', 'frac_image_only']


def row_for(name, qa_dir):
    base = dem.strip_base(name)
    f = os.path.join(qa_dir, base, base + '_ldcheck.json')
    if not os.path.isfile(f):
        return dict(DEM=base, status='missing')
    with open(f) as fh:
        s = json.load(fh)
    out = dict(DEM=base, status=s.get('status'))
    for k in ['frac_signal', 'frac_inconsistent', 'frac_consistent', 'frac_marginal', 'frac_image_only',
              'frac_no_signal', 'frac_inconsistent_of_signal', 'frac_bad_of_signal', 'median_F_signal',
              'U_frac_gt_0.01', 'U_frac_gt_0.02', 'U_p90_signal', 'f_clear_dem', 'f_clear',
              'frac_with_clear_image', 'frac_matched', 'valid_area_km2', 'run_time_s',
              'frac_signal_matched', 'frac_signal_unmatched', 'frac_inconsistent_of_signal_matched',
              'frac_inconsistent_of_signal_unmatched']:
        out[k] = s.get(k, np.nan)
    acq = s.get('acquisitions', [])
    out['n_images'] = len(acq)
    out['n_images_clear'] = int(sum(a.get('clear_frac', 0) > 0.1 for a in acq))
    out['min_abs_dt_days'] = min([abs(a['dt_days']) for a in acq]) if acq else np.nan
    peaks, reg_ok, ntiles = [], [], 0
    for t in s.get('tiles', []):
        if not t.get('images'):
            continue
        ntiles += 1
        reg_ok.append(t['stack']['dem_reg_ok'])
        pk = [d.get('g_dem_peak', np.nan) for d in t['reg'] if d.get('clear_frac', 0) > 0.2]
        pk = [p for p in pk if p is not None and np.isfinite(p)]
        if pk:
            peaks.append(np.max(pk))
    out['n_tiles'] = ntiles
    out['frac_tiles_reg_ok'] = float(np.mean(reg_ok)) if reg_ok else np.nan
    out['dem_peak_median'] = float(np.median(peaks)) if peaks else np.nan
    out['one_minus_median_F'] = 1 - out['median_F_signal'] if out['median_F_signal'] is not None else np.nan
    out['one_minus_dem_peak'] = 1 - out['dem_peak_median']
    return out


def auc(pos, neg):
    """Probability that a random problem DEM scores higher than a random good DEM."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def main():
    csv, qa_dir, prefix = sys.argv[1:4]
    ts = pd.read_csv(csv)
    rows = []
    for _, t in ts.iterrows():
        r = row_for(t.DEM, qa_dir)
        r.update(set=t.set, comment=t.get('comment', ''))
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(prefix + '_summary.csv', index=False)
    print(d.groupby(['set', 'status']).size().unstack(fill_value=0))
    ok = d[d.status == 'done']
    good = ok[ok.set.isin(['good', 'good_rough'])]
    bad = ok[ok.set.isin(['problem_SPK', 'problem_other'])]
    res = []
    for m in METRICS:
        res.append(dict(metric=m, AUC=auc(bad[m], good[m]),
                        AUC_SPK=auc(ok[ok.set == 'problem_SPK'][m], good[m]),
                        AUC_other=auc(ok[ok.set == 'problem_other'][m], good[m]),
                        good_median=np.nanmedian(good[m]), good_p90=np.nanpercentile(good[m].dropna(), 90) if good[m].notna().any() else np.nan,
                        bad_median=np.nanmedian(bad[m])))
    res = pd.DataFrame(res)
    res.to_csv(prefix + '_auc.csv', index=False)
    print(res.round(3).to_string())
    sets = ['good', 'good_rough', 'maybe_not_SPK', 'problem_SPK', 'problem_other']
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, m in zip(axs.ravel(), METRICS):
        for i, sname in enumerate(sets):
            v = ok[ok.set == sname][m].dropna().values
            ax.plot(i + 0.15 * np.random.default_rng(i).standard_normal(v.size), v, 'o', ms=4, alpha=0.6)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s.replace('_', '\n') for s in sets], fontsize=8)
        a = res.set_index('metric').loc[m, 'AUC']
        ax.set_title(f'{m}  (AUC {a:.2f})', fontsize=9)
    fig.savefig(prefix + '_metrics.png', dpi=110)


if __name__ == '__main__':
    main()
