"""Quicklook figure for an ldcheck output GeoTIFF."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from osgeo import gdal

CLASS_COLORS = ['#ffffff', '#d9d9d9', '#1a9850', '#fee08b', '#d73027', '#4575b4']
CLASS_LABELS = ['no data', 'no signal', 'consistent', 'marginal', 'inconsistent', 'image only']


def read_bands(tif):
    ds = gdal.Open(tif)
    gt = ds.GetGeoTransform()
    out = {}
    for i in range(ds.RasterCount):
        b = ds.GetRasterBand(i + 1)
        out[b.GetDescription()] = b.ReadAsArray()
    ext = [gt[0] / 1e3, (gt[0] + gt[1] * ds.RasterXSize) / 1e3,
           (gt[3] + gt[5] * ds.RasterYSize) / 1e3, gt[3] / 1e3]
    return out, ext


def quicklook(tif, png, summary=None):
    B, ext = read_bands(tif)
    # crop to the data
    rows = np.flatnonzero((B['class'] > 0).any(axis=1))
    cols = np.flatnonzero((B['class'] > 0).any(axis=0))
    if rows.size and cols.size:
        sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        dx = (ext[1] - ext[0]) / B['class'].shape[1]
        ext = [ext[0] + cols[0] * dx, ext[0] + (cols[-1] + 1) * dx, ext[3] - (rows[-1] + 1) * dx, ext[3] - rows[0] * dx]
        B = {k: v[sl] for k, v in B.items()}
    ny, nx = B['class'].shape
    wide = nx > 1.5 * ny
    if wide:
        fig, axs = plt.subplots(3, 1, figsize=(12, min(3 * 12 * ny / nx + 2, 16)), constrained_layout=True,
                                sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(16, min(16 / 3 * ny / nx + 1.5, 12)), constrained_layout=True,
                                sharex=True, sharey=True)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), 6)
    im = axs[0].imshow(B['class'], extent=ext, cmap=cmap, norm=norm, interpolation='nearest')
    cb = fig.colorbar(im, ax=axs[0], ticks=range(6), shrink=0.8)
    cb.ax.set_yticklabels(CLASS_LABELS)
    axs[0].set_title('class')
    im = axs[1].imshow(B['F'], extent=ext, cmap='RdYlGn', vmin=0, vmax=1.2, interpolation='nearest')
    fig.colorbar(im, ax=axs[1], shrink=0.8, label='F (confirmed fraction of predicted shading)')
    im = axs[2].imshow(np.log10(B['SNR']), extent=ext, cmap='viridis', vmin=0, vmax=1.5, interpolation='nearest')
    fig.colorbar(im, ax=axs[2], shrink=0.8, label='log10 SNR of predicted shading')
    for ax in axs:
        ax.set_aspect('equal')
    if summary is not None:
        s = summary
        fig.suptitle(f"{s['dem']}\n{len(s.get('acquisitions', []))} images, "
                     f"incons. {100*s.get('frac_inconsistent_of_signal', np.nan):.1f}% of signal area, "
                     f"median F {s.get('median_F_signal', np.nan):.2f}, signal {100*s.get('frac_signal', np.nan):.0f}% of DEM",
                     fontsize=9)
    fig.savefig(png, dpi=110)
    plt.close(fig)
