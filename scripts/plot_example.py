#! /usr/bin/env python
"""
Example figure for one checked DEM (needs a --save_debug run).

usage: plot_example.py qa_dir strip_base out.png [x0,y0,x1,y1 (km)] [image index]

Panels: predicted illumination (high-passed) for one clear image, the image
(normalized, high-passed, advected and registered), the class map, F, and the
standard 16-m filtered DEM coverage if it exists.
"""
import os
import sys
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from osgeo import gdal

from ldcheck.quicklook import CLASS_COLORS, CLASS_LABELS, read_bands

qa_dir, base, out = sys.argv[1:4]
win = [float(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] != 'all' else None
kimg = int(sys.argv[5]) if len(sys.argv) > 5 else None
d = os.path.join(qa_dir, base)
B, ext = read_bands(os.path.join(d, base + '_ldcheck.tif'))
with h5py.File(os.path.join(d, base + '_ldcheck_debug.h5'), 'r') as h:
    keys = [k.decode() if isinstance(k, bytes) else k for k in h.attrs['keys']]
    clear = h['clear'][:]
    if kimg is None:
        kimg = int(np.argmax(clear.reshape(clear.shape[0], -1).sum(axis=1)))
    o = h['o'][kimg]
    p = h['p'][kimg]
    ck = clear[kimg]
ny, nx = o.shape
x = np.linspace(ext[0], ext[1], nx + 1)
y = np.linspace(ext[3], ext[2], ny + 1)
if win is None:
    sl = (slice(None), slice(None))
else:
    c0, c1 = np.searchsorted(x, win[0]), np.searchsorted(x, win[2])
    r0, r1 = np.searchsorted(-y, -win[3]), np.searchsorted(-y, -win[1])
    sl = (slice(r0, r1), slice(c0, c1))
e = [x[sl[1]][0] if win else ext[0], x[sl[1]][-1] if win else ext[1],
     y[sl[0]][-1] if win else ext[2], y[sl[0]][0] if win else ext[3]]
s = np.nanstd(p[sl])
fig, axs = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True, sharex=True, sharey=True)
axs[0, 0].imshow(p[sl], extent=e, cmap='gray', vmin=-2 * s, vmax=2 * s)
axs[0, 0].set_title(f'predicted illumination (HP), sun of {keys[kimg]}')
oo = np.where(ck, o, np.nan)[sl]
so = np.nanstd(oo)
axs[0, 1].imshow(o[sl], extent=e, cmap='gray', vmin=-2 * so, vmax=2 * so)
axs[0, 1].contour(np.linspace(e[0], e[1], o[sl].shape[1]), np.linspace(e[3], e[2], o[sl].shape[0]),
                  (~ck[sl]).astype(float), [0.5], colors='c', linewidths=0.6)
axs[0, 1].set_title(f'{keys[kimg]} (normalized, HP); cyan: not clear')
cmap = ListedColormap(CLASS_COLORS)
im = axs[1, 0].imshow(B['class'][sl], extent=e, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 6.5), 6),
                      interpolation='nearest')
cb = fig.colorbar(im, ax=axs[1, 0], ticks=range(6), shrink=0.7)
cb.ax.set_yticklabels(CLASS_LABELS)
axs[1, 0].set_title('class')
im = axs[1, 1].imshow(B['F'][sl], extent=e, cmap='RdYlGn', vmin=0, vmax=1.2, interpolation='nearest')
fig.colorbar(im, ax=axs[1, 1], shrink=0.7, label='F')
axs[1, 1].set_title('F (fraction of the predicted shading confirmed)')
for ax in axs.ravel():
    ax.set_aspect('equal')
fig.suptitle(base, fontsize=10)
fig.savefig(out, dpi=100)
