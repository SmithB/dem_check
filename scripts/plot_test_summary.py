#! /usr/bin/env python
"""
Figure of the test results: per-set distributions of the whole-DEM metrics, and
ROC curves (problem vs good DEMs).

usage: plot_test_summary.py results/test_merged.csv out.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SETS = ['good', 'good_rough', 'maybe_not_SPK', 'problem_SPK', 'problem_other']
LABELS = ['good\n(fits)', 'good, rough\n(Crosson)', 'maybe not\n(SPK list)', 'problem\n(SPK lists)',
          'problem\n(other lists)']
COLORS = ['#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#e87ba4']
PANELS = [('kept_frac_incons', 'fraction of the standard filtered\nDEM that is inconsistent', 'linear', 0.2),
          ('frac_inconsistent_of_signal', 'whole 2-m DEM: inconsistent fraction\nof the area with shading signal', 'linear', 0.3),
          ('kept_frac_signal', 'fraction of the filtered DEM\nwith a shading signal', 'log', None),
          ('frac_with_clear_image', 'fraction of the DEM seen by\nat least one clear image', 'linear', None)]


def roc(pos, neg):
    thr = np.unique(np.r_[pos, neg])[::-1]
    tpr = [(pos >= t).mean() for t in thr]
    fpr = [(neg >= t).mean() for t in thr]
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]


def main():
    d = pd.read_csv(sys.argv[1])
    d = d[d.status == 'done']
    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    rng = np.random.default_rng(0)
    for i, (m, label, scale, thr) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        for j, (s, c) in enumerate(zip(SETS, COLORS)):
            v = d.loc[d.set == s, m].values.astype(float)
            v = v[np.isfinite(v)]
            if scale == 'log':
                v = np.maximum(v, 1e-4)
            ax.plot(j + 0.12 * rng.standard_normal(v.size), v, 'o', ms=6, color=c, alpha=0.75,
                    mec='white', mew=0.5)
            if v.size:
                ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, '-', color='#333333', lw=1.5)
        if thr is not None:
            ax.axhline(thr, color='#888888', lw=1, ls='--')
        ax.set_yscale(scale)
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis='y', color='#e5e5e5', lw=0.6)
        ax.set_axisbelow(True)
    good = d[d.set.isin(['good', 'good_rough'])]
    bad = d[d.set.isin(['problem_SPK', 'problem_other'])]
    ax = fig.add_subplot(gs[1, 0:2])
    for m, c, lab in [('kept_frac_incons', '#2a78d6', 'inconsistent fraction of the filtered DEM'),
                      ('frac_inconsistent_of_signal', '#eb6834', 'whole DEM: inconsistent fraction of signal area'),
                      ('U_frac_gt_0.01', '#1baf7a', 'whole DEM: fraction with unconfirmed amp > 0.01')]:
        p, n = bad[m].dropna().values, good[m].dropna().values
        if p.size and n.size:
            x, y = roc(p, n)
            auc = np.trapezoid(y, x)
            ax.plot(x, y, '-', color=c, lw=2, label=f'{lab} (AUC {auc:.2f})')
    ax.plot([0, 1], [0, 1], ':', color='#999999', lw=1)
    ax.set_xlabel('fraction of good DEMs flagged')
    ax.set_ylabel('fraction of problem-list DEMs flagged')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title(f'whole-DEM detection: {len(bad)} problem-list DEMs vs {len(good)} good DEMs', fontsize=10)
    ax.grid(color='#e5e5e5', lw=0.6)
    ax = fig.add_subplot(gs[1, 2:4])
    for j, (s, c) in enumerate(zip(SETS, COLORS)):
        dd = d[d.set == s]
        ax.plot(np.maximum(dd.kept_frac_signal, 1e-3), np.maximum(dd['kept_frac_incons'], 1e-4), 'o', color=c, ms=7,
                alpha=0.8, mec='white', mew=0.5, label=LABELS[j].replace('\n', ' '))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('fraction of the filtered DEM with a shading signal')
    ax.set_ylabel('fraction of the filtered DEM that is inconsistent')
    ax.axhline(0.2, color='#888888', lw=1, ls='--')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(color='#e5e5e5', lw=0.6)
    fig.savefig(sys.argv[2], dpi=110)


if __name__ == '__main__':
    main()
