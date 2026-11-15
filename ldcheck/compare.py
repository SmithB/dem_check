"""
DEM-vs-image statistics and classification.

For clear image k, with p_k the high-passed predicted illumination and o_k the
normalized high-passed image (both in units of mu), local Gaussian-weighted
moments give
    b_k = cov(o_k, p_k) / var(p_k)          (local gain)
    r_k = cov / sqrt(var(p_k) var(o_k))     (local correlation)
For a correct DEM b_k ~ f_k, the image gain (about the direct-beam fraction, a
property of the image, estimated over the whole strip).  The pooled statistic
    F = sum_k cov_k / sum_k f_k var(p_k)
is the fraction of the predicted shading that the images confirm.  Surface
albedo variations add variance to o but no covariance with p, so F is not
biased by them (r is).  F is only meaningful where the predicted shading is
large compared with the image noise / change floor:
    SNR = sqrt(sum_k f_k^2 var(p_k) / sum_k noise_k^2)

Classes (uint8):
    0  no data (no DEM or no clear image)
    1  no signal: the DEM predicts little shading, and the images show little texture
    2  consistent: SNR >= snr_min and F >= F_good
    3  marginal: SNR >= snr_min and F_bad <= F < F_good
    4  inconsistent: SNR >= snr_min and F < F_bad  (DEM structure not in the images)
    5  image texture missing from the DEM: SNR < snr_min but the (stack-consistent)
       image texture is well above the noise
"""
import numpy as np
import scipy.ndimage as snd
from .stack import local_moments

CLASS_NAMES = {0: 'no_data', 1: 'no_signal', 2: 'consistent', 3: 'marginal',
               4: 'inconsistent', 5: 'image_only'}


def image_moments(o, p, w, sigma):
    """Local moments of image o and prediction p with weight w."""
    W, voo, vpp, vop = local_moments(o, p, w, sigma)
    return dict(W=W, voo=voo, vpp=vpp, vop=vop)


def estimate_gain(vop, vpp, W, frac_top=0.3, min_W=0.7):
    """
    Image gain f from the pixels with the strongest predicted signal:
    f = sum(cov) / sum(var_p) over pixels in the top frac_top of var_p.
    Returns f, and the correlation-weighted number of pixels used.
    """
    good = (W > min_W) & np.isfinite(vop) & np.isfinite(vpp)
    if good.sum() < 100:
        return np.nan, 0
    thr = np.percentile(vpp[good], 100 * (1 - frac_top))
    sel = good & (vpp >= thr)
    return float(vop[sel].sum() / vpp[sel].sum()), int(sel.sum())


def classify(F, SNR, img_snr, n_clear, amp=None, img_amp=None, snr_min=3., amp_min=0.005,
             F_good=0.5, F_bad=0.25, img_snr_min=4.):
    """
    Signal: SNR >= snr_min and the predicted image amplitude (mu units) >= amp_min.
    Image-only texture: no signal, but the image amplitude >= 1.5*amp_min and
    img_snr >= img_snr_min, seen in at least two clear images.
    """
    cls = np.zeros(F.shape, dtype=np.uint8)
    has = n_clear > 0
    cls[has] = 1
    sig = has & (SNR >= snr_min) & np.isfinite(F)
    if amp is not None:
        sig &= amp >= amp_min
    cls[sig & (F >= F_good)] = 2
    cls[sig & (F < F_good) & (F >= F_bad)] = 3
    cls[sig & (F < F_bad)] = 4
    img = has & ~sig & (img_snr >= img_snr_min) & (n_clear >= 2)
    if img_amp is not None:
        img &= img_amp >= 1.5 * amp_min
    cls[img] = 5
    return cls


def summarize(cls, F, SNR, valid):
    """Area fractions of each class over the valid DEM area, and pooled stats."""
    out = {}
    nv = max(int(valid.sum()), 1)
    for k, name in CLASS_NAMES.items():
        out[f'frac_{name}'] = float(((cls == k) & valid).sum() / nv)
    sig = valid & ((cls == 2) | (cls == 3) | (cls == 4))
    ns = max(int(sig.sum()), 1)
    out['frac_signal'] = float(sig.sum() / nv)
    out['frac_inconsistent_of_signal'] = float(((cls == 4) & valid).sum() / ns)
    out['frac_bad_of_signal'] = float((((cls == 4) | (cls == 3)) & valid).sum() / ns)
    out['median_F_signal'] = float(np.nanmedian(F[sig])) if sig.any() else np.nan
    return out


def kept_area_stats(ldcheck_tif, filtered_file, min_valid=0.9):
    """
    Landsat classes on the cells of a filtered DEM (16 m or 32 m dem_filter output).

    Returns dict: n_valid (cells with DEM data at 15 m), n_kept (cells the filter kept),
    kept_frac_signal, kept_frac_incons (inconsistent / kept), kept_incons_of_signal,
    removed_cons_of_signal (consistent / removed cells with signal), and counts by class.
    """
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(filtered_file)
    gt = ds.GetGeoTransform()
    z = ds.GetRasterBand(1).ReadAsArray()
    ny, nx = z.shape
    b = [gt[0], gt[3] + gt[5] * ny, gt[0] + gt[1] * nx, gt[3]]

    def on_grid(band, alg):
        d = gdal.Warp('', ldcheck_tif, options=gdal.WarpOptions(format='MEM', outputBounds=b, width=nx, height=ny,
                                                                 resampleAlg=alg, srcBands=[band], dstBands=[1]))
        return d.GetRasterBand(1).ReadAsArray()
    src = gdal.Open(ldcheck_tif)
    names = {src.GetRasterBand(i + 1).GetDescription(): i + 1 for i in range(src.RasterCount)}
    cls = on_grid(names['class'], 'near')
    vf = on_grid(names['DEM_valid_frac'], 'bilinear')
    valid = np.isfinite(vf) & (vf >= min_valid) & (cls > 0)
    kept = np.isfinite(z) & valid
    rem = valid & ~np.isfinite(z)
    out = dict(n_valid=int(valid.sum()), n_kept=int(kept.sum()), n_removed=int(rem.sum()))
    for c, name in [(1, 'nosig'), (2, 'cons'), (3, 'marg'), (4, 'incons'), (5, 'imgonly')]:
        out[f'kept_{name}'] = int((kept & (cls == c)).sum())
        out[f'removed_{name}'] = int((rem & (cls == c)).sum())
    sig_k = out['kept_cons'] + out['kept_marg'] + out['kept_incons']
    out['kept_frac_signal'] = sig_k / max(out['n_kept'], 1)
    out['kept_frac_incons'] = out['kept_incons'] / max(out['n_kept'], 1)
    out['kept_incons_of_signal'] = out['kept_incons'] / sig_k if sig_k else float('nan')
    sig_r = out['removed_cons'] + out['removed_marg'] + out['removed_incons']
    out['removed_cons_of_signal'] = out['removed_cons'] / sig_r if sig_r else float('nan')
    return out
