"""
The Landsat image stack for one DEM tile.

Every acquisition is
  1. calibrated to TOA reflectance (no sun-angle correction), rho' = M*DN + A,
  2. advected to the DEM time (sampled at x + v*(t_img - t_DEM) + shift),
  3. normalized: o = (rho' - LP(rho')) / (LP(rho') / sin(e)), so that o is in
     the units of the direct illumination mu (for a Lambertian surface with
     albedo LP(rho')/sin(e), o ~ f_dir * HP(mu), f_dir = direct-beam fraction).
Images are registered to each other and, as a stack, to the DEM prediction,
with masked normalized cross-correlation (NCC) in blocks.  Clouds (and other
image-to-image changes) are found from the local consistency of each image with
the median of the others.
"""
import numpy as np
import scipy.ndimage as snd
import scipy.fft as sfft
from osgeo import gdal
from .shading import masked_gaussian

gdal.UseExceptions()
L15 = 15.


# ---------------------------------------------------------------- image I/O
def read_chip_array(fname, bounds=None):
    """Read a chip, optionally cropped to bounds (pixel edges on the chip's grid)."""
    ds = gdal.Open(fname)
    gt = ds.GetGeoTransform()
    if bounds is None:
        return ds.GetRasterBand(1).ReadAsArray(), gt
    c0 = int(round((bounds[0] - gt[0]) / gt[1]))
    r0 = int(round((bounds[3] - gt[3]) / gt[5]))
    nc = int(round((bounds[2] - bounds[0]) / gt[1]))
    nr = int(round((bounds[1] - bounds[3]) / gt[5]))
    a = ds.GetRasterBand(1).ReadAsArray(c0, r0, nc, nr)
    return a, (gt[0] + c0 * gt[1], gt[1], 0., gt[3] + r0 * gt[5], 0., gt[5])


def read_on_grid(fname, gt, shape, order=0):
    """
    Read a (coarser) chip and sample it at the pixel centers of the grid (gt, shape):
    order 0 = nearest, 1 = bilinear.  Zero (no data) becomes NaN.
    """
    ds = gdal.Open(fname)
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    g2 = ds.GetGeoTransform()
    a[a == 0] = np.nan
    x = gt[0] + gt[1] * (np.arange(shape[1]) + 0.5)
    y = gt[3] + gt[5] * (np.arange(shape[0]) + 0.5)
    cols = (x - g2[0]) / g2[1] - 0.5
    rows = (y - g2[3]) / g2[5] - 0.5
    if order == 0:
        ci = np.clip(np.round(cols).astype(int), 0, a.shape[1] - 1)
        ri = np.clip(np.round(rows).astype(int), 0, a.shape[0] - 1)
        return a[np.ix_(ri, ci)]
    R, C = np.meshgrid(rows, cols, indexing='ij')
    good = np.isfinite(a)
    fill = np.where(good, a, np.nanmean(a) if good.any() else 0.)
    v = snd.map_coordinates(fill, [R, C], order=1, mode='nearest')
    m = snd.map_coordinates(good.astype(np.float32), [R, C], order=1, mode='nearest')
    v[m < 0.5] = np.nan
    return v


def sun_angle_maps(meta, gt, shape, dec=32):
    """Sun elevation and azimuth on a chip grid, from the scene-center time (sun.py)."""
    import datetime
    import pyproj
    from .sun import sun_position
    t = datetime.datetime.fromisoformat(meta['date'] + 'T' + meta['time'].replace('Z', '')[:15]).replace(
        tzinfo=datetime.timezone.utc)
    rows = np.arange(0, shape[0] + dec, dec)
    cols = np.arange(0, shape[1] + dec, dec)
    x = gt[0] + gt[1] * (cols + 0.5)
    y = gt[3] + gt[5] * (rows + 0.5)
    X, Y = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(3031, 4326, always_xy=True).transform(X, Y)
    el, az = sun_position(t, lat, lon)
    R, C = np.meshgrid(np.arange(shape[0]) / dec, np.arange(shape[1]) / dec, indexing='ij')
    out = []
    for a in (el, np.cos(np.radians(az)), np.sin(np.radians(az))):
        out.append(snd.map_coordinates(a, [R, C], order=1, mode='nearest'))
    return out[0].astype(np.float32), np.degrees(np.arctan2(out[2], out[1])).astype(np.float32)


def spectral_cloud_mask(ch, gt, shape, sun_elev, pan_refl=None, ndsi_min=0.55, b6_max=0.2, b9_max=0.025):
    """
    True where a scene is cloudy by its spectrum: snow has a high NDSI (visible vs
    1.6-um SWIR) and is dark in the 1.38-um cirrus band; water clouds are bright at
    1.6 um and high clouds are bright at 1.38 um.  Reflectances are sun-corrected.
    The visible band is the green band if it was downloaded, else the pan band
    (pan_refl: rho' on the chip grid, smoothed to 60 m here).
    Returns None when the bands are not available.
    """
    m = ch['meta']
    if not all(b in ch['files'] for b in ('swir16', 'cirrus')) or 'band_refl' not in m:
        return None
    sin_e = np.sin(np.radians(sun_elev))
    R = {}
    for name, b in (('swir16', '6'), ('cirrus', '9')):
        dn = read_on_grid(ch['files'][name], gt, shape, order=0)
        mult, add = m['band_refl'][b]
        R[name] = (mult * dn + add) / sin_e
    if 'green' in ch['files']:
        mult, add = m['band_refl']['3']
        vis = (mult * read_on_grid(ch['files']['green'], gt, shape, order=0) + add) / sin_e
    elif pan_refl is not None:
        vis, _ = masked_gaussian(pan_refl, 2., np.isfinite(pan_refl))
        vis = vis / sin_e
    else:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        ndsi = (vis - R['swir16']) / (vis + R['swir16'])
        cloud = (ndsi < ndsi_min) | (R['swir16'] > b6_max) | (R['cirrus'] > b9_max)
    # grow by 4 pixels (60 m) to cover edges
    cloud = snd.binary_dilation(cloud, iterations=4)
    return cloud


def scene_time(meta):
    import datetime
    return datetime.datetime.fromisoformat(meta['date'] + 'T' + meta['time'].replace('Z', '')[:15]).replace(
        tzinfo=datetime.timezone.utc)


def spectral_cloud_mask60(ch, gt, shape, sun_elev, pan_refl, ndsi_min=0.55, b6_max=0.2, b9_max=0.025):
    """
    Spectral cloud mask on the 60-m grid made of 4x4 blocks of the pan chip (see
    spectral_cloud_mask for the tests).  Returns (mask, gt60, shape60), or None.
    """
    m = ch['meta']
    if not all(b in ch['files'] for b in ('swir16', 'cirrus')) or 'band_refl' not in m:
        return None
    ny, nx = shape[0] // 4, shape[1] // 4
    gt60 = (gt[0], gt[1] * 4, 0., gt[3], 0., gt[5] * 4)
    p = pan_refl[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4)
    with np.errstate(invalid='ignore'):
        vis = np.nanmean(p, axis=(1, 3))
    sin_e = np.sin(np.radians(sun_elev))
    R = {}
    for name, b in (('swir16', '6'), ('cirrus', '9')):
        dn = read_on_grid(ch['files'][name], gt60, (ny, nx), order=0)
        mult, add = m['band_refl'][b]
        R[name] = (mult * dn + add) / sin_e
    vis = vis / sin_e
    with np.errstate(invalid='ignore', divide='ignore'):
        ndsi = (vis - R['swir16']) / (vis + R['swir16'])
        cloud = (ndsi < ndsi_min) | (R['swir16'] > b6_max) | (R['cirrus'] > b9_max)
    cloud = snd.binary_dilation(cloud, iterations=1) & np.isfinite(vis)
    return cloud, gt60


def load_acquisition(chips, bounds=None, spectral=True, spectral_kw=None):
    """
    Mosaic the scenes of one acquisition (all pan chips are on the same 15-m grid).
    Returns dict: rho (rho' = M*DN + A, float32, NaN = no data), gt, time (mean scene
    time, for the sun position), scloud (spectral cloud mask on a 60-m grid, or None),
    gt60.
    """
    rho = None
    scloud, gt60 = None, None
    times = []
    for ch in chips:
        dn, gt = read_chip_array(ch['files']['pan'], bounds)
        m = ch['meta']
        r = (m['refl_mult'] * dn.astype(np.float32) + m['refl_add']).astype(np.float32)
        r[(dn == 0) | (dn >= 65535)] = np.nan
        del dn
        times.append(scene_time(m))
        if spectral:
            sc = spectral_cloud_mask60(ch, gt, r.shape, m['sun_elev'], r, **(spectral_kw or {}))
            if sc is not None:
                scloud = sc[0] if scloud is None else (scloud | sc[0])
                gt60 = sc[1]
        if rho is None:
            rho = r
        else:
            both = np.isfinite(rho) & np.isfinite(r)
            rho[both] = 0.5 * (rho[both] + r[both])
            only = ~np.isfinite(rho) & np.isfinite(r)
            rho[only] = r[only]
    t0 = min(times)
    tmean = t0 + sum((t - t0 for t in times), t0 - t0) / len(times)
    return dict(rho=rho, gt=gt, time=tmean, scloud=scloud, gt60=gt60)


def sun_at(t, x, y):
    """Sun elevation and geographic azimuth at EPSG:3031 point(s) x, y at time t."""
    import pyproj
    from .sun import sun_position
    lon, lat = pyproj.Transformer.from_crs(3031, 4326, always_xy=True).transform(x, y)
    return sun_position(t, lat, lon)


# ---------------------------------------------------------------- resampling
def sample_image(img, gt, x, y, order=3, margin=8):
    """
    Sample img (15-m chip, geotransform gt) at map coordinates x, y (2-D arrays).
    Only the window of the chip around the requested points (+ margin pixels) is
    gap-filled and spline-filtered.  Returns the samples (NaN outside / in gaps).
    """
    col = (x - gt[0]) / gt[1] - 0.5
    row = (y - gt[3]) / gt[5] - 0.5
    ny, nx = img.shape
    c0 = int(max(np.floor(np.nanmin(col)) - margin, 0))
    c1 = int(min(np.ceil(np.nanmax(col)) + margin + 1, nx))
    r0 = int(max(np.floor(np.nanmin(row)) - margin, 0))
    r1 = int(min(np.ceil(np.nanmax(row)) + margin + 1, ny))
    if c1 <= c0 or r1 <= r0:
        return np.full(x.shape, np.nan, np.float32)
    sub = img[r0:r1, c0:c1]
    col = col - c0
    row = row - r0
    good = np.isfinite(sub)
    if not good.any():
        return np.full(x.shape, np.nan, np.float32)
    if order > 1:
        fill, _ = masked_gaussian(sub, 2., good, min_weight=0.)
        fill = np.where(good, sub, np.nan_to_num(fill, nan=float(np.nanmean(sub))))
    else:
        fill = np.where(good, sub, 0)
    v = snd.map_coordinates(fill.astype(np.float64), [row, col], order=order, mode='constant', cval=np.nan,
                            prefilter=True)
    m = snd.map_coordinates(good.astype(np.float32), [row, col], order=1, mode='constant', cval=0)
    v[m < 0.99] = np.nan
    return v.astype(np.float32)


def high_pass(a, sigma_px, mask=None):
    """a - masked Gaussian low-pass; also returns the low-pass."""
    if mask is None:
        mask = np.isfinite(a)
    lp, w = masked_gaussian(a, sigma_px, mask, min_weight=0.05)
    hp = a - lp
    hp[~mask] = np.nan
    return hp, lp


def normalize_image(rho, sun_elev, sigma_px, mask=None):
    """o = HP(rho') / (LP(rho') / sin(e)); also returns the albedo estimate LP/sin(e)."""
    hp, lp = high_pass(rho, sigma_px, mask)
    albedo = lp / np.sin(np.radians(sun_elev))
    o = hp / albedo
    return o.astype(np.float32), albedo.astype(np.float32)


# ---------------------------------------------------------------- registration
def masked_ncc(a, b, ma, mb, max_shift):
    """
    Masked normalized cross-correlation (Padfield, 2012) of a(x) with b(x + s)
    for integer shifts s = (dr, dc), |dr|, |dc| <= max_shift.
    Returns ncc[(2*max_shift+1), (2*max_shift+1)], indexed [dr+max_shift, dc+max_shift],
    and the overlap pixel count.
    """
    a = np.where(ma, a, 0).astype(np.float32)
    b = np.where(mb, b, 0).astype(np.float32)
    ma = ma.astype(np.float32)
    mb = mb.astype(np.float32)
    shape = [a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1]
    fshape = [sfft.next_fast_len(s) for s in shape]

    def F(x):
        return sfft.rfft2(x, fshape)

    def xc(fa, fb):
        # sum_x A(x) B(x+s)
        return sfft.irfft2(np.conj(fa) * fb, fshape)

    Fa, Fb, Fma, Fmb = F(a), F(b), F(ma), F(mb)
    Fa2, Fb2 = F(a * a), F(b * b)
    n = xc(Fma, Fmb)
    sab = xc(Fa, Fb)
    sa = xc(Fa, Fmb)
    sb = xc(Fma, Fb)
    sa2 = xc(Fa2, Fmb)
    sb2 = xc(Fma, Fb2)
    # pick shifts -max..max; xc result index k corresponds to shift k (mod fshape)
    idx_r = np.arange(-max_shift, max_shift + 1) % fshape[0]
    idx_c = np.arange(-max_shift, max_shift + 1) % fshape[1]
    sel = np.ix_(idx_r, idx_c)
    n, sab, sa, sb, sa2, sb2 = [q[sel] for q in (n, sab, sa, sb, sa2, sb2)]
    n = np.round(n)
    with np.errstate(invalid='ignore', divide='ignore'):
        num = sab - sa * sb / n
        den = np.sqrt(np.maximum(sa2 - sa ** 2 / n, 0) * np.maximum(sb2 - sb ** 2 / n, 0))
        ncc = num / den
    ncc[(n < 0.5 * ma.sum()) | ~np.isfinite(ncc)] = np.nan
    return ncc, n


def peak_subpixel(ncc, max_shift):
    """Location (dr, dc) and value of the NCC peak, refined by a quadratic fit."""
    if not np.any(np.isfinite(ncc)):
        return np.nan, np.nan, np.nan, False
    k = np.nanargmax(ncc)
    ir, ic = np.unravel_index(k, ncc.shape)
    peak = ncc[ir, ic]
    on_edge = ir in (0, ncc.shape[0] - 1) or ic in (0, ncc.shape[1] - 1)
    dr, dc = float(ir), float(ic)
    if not on_edge:
        c = ncc[ir - 1:ir + 2, ic - 1:ic + 2]
        if np.all(np.isfinite(c)):
            denr = c[0, 1] - 2 * c[1, 1] + c[2, 1]
            denc = c[1, 0] - 2 * c[1, 1] + c[1, 2]
            if denr < 0:
                dr += np.clip(0.5 * (c[0, 1] - c[2, 1]) / denr, -0.5, 0.5)
            if denc < 0:
                dc += np.clip(0.5 * (c[1, 0] - c[1, 2]) / denc, -0.5, 0.5)
    return dr - max_shift, dc - max_shift, peak, not on_edge


def global_shift(a, b, ma, mb, max_shift=20, decimate=1):
    """
    Shift (dr, dc), in 15-m pixels, such that b(x + s) matches a(x); with the
    peak NCC.  A first pass on 2x decimated fields covers large shifts.
    """
    if decimate > 1:
        sl = (slice(None, None, decimate), slice(None, None, decimate))
        a1, b1, ma1, mb1 = (snd.uniform_filter(np.nan_to_num(a), decimate)[sl],
                            snd.uniform_filter(np.nan_to_num(b), decimate)[sl], ma[sl], mb[sl])
        ncc, _ = masked_ncc(a1, b1, ma1, mb1, max_shift // decimate + 1)
        dr, dc, pk, ok = peak_subpixel(ncc, max_shift // decimate + 1)
        if not np.isfinite(dr):
            return np.nan, np.nan, np.nan, False
        return dr * decimate, dc * decimate, pk, ok
    ncc, _ = masked_ncc(a, b, ma, mb, max_shift)
    return peak_subpixel(ncc, max_shift)


def block_shifts(a, b, ma, mb, block=96, max_shift=6, min_valid=0.4, min_peak=0.25, overlap=False):
    """
    NCC shifts of b against a in blocks (stride block, or block/2 with overlap=True).
    Returns block-center rows, cols and arrays dr, dc, peak (NaN where rejected).
    """
    ny, nx = a.shape
    step = block // 2 if overlap else block
    r0s = np.arange(0, max(ny - block, 0) + 1, step)
    c0s = np.arange(0, max(nx - block, 0) + 1, step)
    if r0s.size == 0:
        r0s = np.array([0])
    if c0s.size == 0:
        c0s = np.array([0])
    DR = np.full((r0s.size, c0s.size), np.nan)
    DC = DR.copy()
    PK = DR.copy()
    for i, r0 in enumerate(r0s):
        for j, c0 in enumerate(c0s):
            sl = (slice(r0, r0 + block), slice(c0, c0 + block))
            mab = ma[sl] & np.isfinite(a[sl])
            if mab.mean() < min_valid or (mb[sl].mean() < min_valid):
                continue
            ncc, _ = masked_ncc(a[sl], b[sl], mab, mb[sl] & np.isfinite(b[sl]), max_shift)
            dr, dc, pk, ok = peak_subpixel(ncc, max_shift)
            if ok and pk >= min_peak:
                DR[i, j], DC[i, j], PK[i, j] = dr, dc, pk
    rows = r0s + block / 2 - 0.5
    cols = c0s + block / 2 - 0.5
    return rows, cols, DR, DC, PK


def smooth_shift_field(rows, cols, DR, DC, PK, shape, default=(0., 0.), max_dev=1.0, smooth_blocks=1.0):
    """
    Robustly clean block shifts (reject blocks more than max_dev pixels from the
    median of their neighbours), fill gaps, smooth, and interpolate to every pixel.
    Returns dr, dc fields (pixels) and the fraction of good blocks.
    """
    good = np.isfinite(DR)
    if good.sum() == 0:
        return np.full(shape, default[0]), np.full(shape, default[1]), 0.
    for A in (DR, DC):
        med = snd.generic_filter(np.where(good, A, np.nan), np.nanmedian, size=3, mode='nearest')
        good &= np.abs(A - med) <= max_dev
    if good.sum() == 0:
        return np.full(shape, default[0]), np.full(shape, default[1]), 0.
    w = PK.copy()
    w[~good] = 0
    out = []
    for A, dflt in ((DR, default[0]), (DC, default[1])):
        A0 = np.where(good, A, 0.)
        sw = snd.gaussian_filter(w, smooth_blocks, mode='nearest')
        sa = snd.gaussian_filter(w * A0, smooth_blocks, mode='nearest')
        # wide fill for gaps
        sw2 = snd.gaussian_filter(w, 4 * smooth_blocks + 2, mode='nearest')
        sa2 = snd.gaussian_filter(w * A0, 4 * smooth_blocks + 2, mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            S = np.where(sw > 0.05 * w.max(), sa / sw, sa2 / sw2)
        S[~np.isfinite(S)] = np.nanmedian(A[good])
        # interpolate to pixels (bilinear on the block-center grid, nearest beyond)
        rr = np.clip(np.interp(np.arange(shape[0]), rows, np.arange(rows.size)), 0, rows.size - 1)
        cc = np.clip(np.interp(np.arange(shape[1]), cols, np.arange(cols.size)), 0, cols.size - 1)
        RR, CC = np.meshgrid(rr, cc, indexing='ij')
        out.append(snd.map_coordinates(S, [RR, CC], order=1, mode='nearest'))
    return out[0], out[1], good.mean()


# ---------------------------------------------------------------- consistency / clouds
def nanmedian0(a):
    """Fast NaN-aware median along axis 0 (NaN where all are NaN)."""
    a = np.asarray(a)
    if a.shape[0] == 1:
        return a[0].copy()
    srt = np.sort(a, axis=0)            # NaNs sort last
    n = np.isfinite(a).sum(axis=0)
    lo = np.clip((n - 1) // 2, 0, a.shape[0] - 1)
    hi = np.clip(n // 2, 0, a.shape[0] - 1)
    out = 0.5 * (np.take_along_axis(srt, lo[None], 0)[0] + np.take_along_axis(srt, hi[None], 0)[0])
    out[n == 0] = np.nan
    return out.astype(a.dtype)


def local_moments(a, b, w, sigma):
    """Gaussian-weighted local means, variances and covariance of a and b with weights w."""
    a0 = np.where(w > 0, np.nan_to_num(a), 0)
    b0 = np.where(w > 0, np.nan_to_num(b), 0)
    W = snd.gaussian_filter(w.astype(np.float32), sigma, mode='constant', truncate=3.)
    with np.errstate(invalid='ignore', divide='ignore'):
        ma = snd.gaussian_filter(w * a0, sigma, mode='constant', truncate=3.) / W
        mb = snd.gaussian_filter(w * b0, sigma, mode='constant', truncate=3.) / W
        vaa = snd.gaussian_filter(w * a0 * a0, sigma, mode='constant', truncate=3.) / W - ma ** 2
        vbb = snd.gaussian_filter(w * b0 * b0, sigma, mode='constant', truncate=3.) / W - mb ** 2
        vab = snd.gaussian_filter(w * a0 * b0, sigma, mode='constant', truncate=3.) / W - ma * mb
    return W, np.maximum(vaa, 0), np.maximum(vbb, 0), vab


def image_consistency(O, ALB, masks, sigma=8., min_w=0.3):
    """
    For each image k: local NCC with the median of the other images, its local
    contrast relative to the others, and its albedo relative to the others.
    O: list of normalized images (NaN = missing), ALB: albedo (LP/sin e) images.
    Returns lists of (ncc, contrast_ratio, albedo_ratio) maps; NaN where undefined.
    """
    n = len(O)
    out = []
    Ostack = np.stack([np.where(m, o, np.nan) for o, m in zip(O, masks)])
    Astack = np.stack([np.where(m, a, np.nan) for a, m in zip(ALB, masks)])
    for k in range(n):
        others = [j for j in range(n) if j != k]
        if not others:
            out.append((None, None, None))
            continue
        with np.errstate(all='ignore'):
            med = np.nanmedian(Ostack[others], axis=0) if len(others) > 1 else Ostack[others[0]]
            amed = np.nanmedian(Astack[others], axis=0) if len(others) > 1 else Astack[others[0]]
        w = (masks[k] & np.isfinite(med) & np.isfinite(O[k])).astype(np.float32)
        W, vkk, vmm, vkm = local_moments(O[k], med, w, sigma)
        with np.errstate(invalid='ignore', divide='ignore'):
            ncc = vkm / np.sqrt(vkk * vmm)
            # contrast ratio against the typical single other image (the median of
            # several images is smoother than one image, so compare with each)
            vo = []
            for j in others:
                wj = (masks[j] & masks[k] & np.isfinite(O[j])).astype(np.float32)
                Wj, vjj, _, _ = local_moments(O[j], O[j], wj, sigma)
                vo.append(np.where(Wj > min_w, vjj, np.nan))
            vo = np.nanmedian(np.stack(vo), axis=0) if len(vo) > 1 else vo[0]
            contrast = np.sqrt(vkk / vo)
            alb_lp, _ = masked_gaussian(ALB[k] / amed, sigma, w > 0)
        bad = W < min_w
        for q in (ncc, contrast, alb_lp):
            q[bad] = np.nan
        out.append((ncc.astype(np.float32), contrast.astype(np.float32), alb_lp.astype(np.float32)))
    return out


def cloud_screen(O, ALB, M, S=None, sigma=8., ncc_min=0.4, ncc_est=0.5, tex_min=0.002, contrast_min=0.5,
                 excess_tex=3., alb_tol=0.1, noise_sigma=18., min_w=0.3):
    """
    DEM-independent screening of the registered, normalized images of one tile,
    from pairwise image-to-image consistency.

    For every pair (j, k) of images: the local NCC (Gaussian window sigma, pixels).
    For image k: best_ncc[k] = its highest NCC with any other image.
    The surface texture is 'established' where some pair agrees (NCC > ncc_est).
    Image k is cloudy where
      - it is spectrally cloudy (S[k] False), or
      - the surface is established but k agrees with no other image (best_ncc < ncc_min), or
      - the surface is established and k agrees, but its contrast is less than
        contrast_min times that of the agreeing images (thin cloud / haze), or
      - the surface is not established and k has much more texture than the
        smoothest image (variance > excess_tex * min variance + tex_min^2; cloud
        texture over smooth snow), or
      - its albedo (LP(rho')/sin e) differs from the spectrally clear median by > alb_tol.
    noise2[k] = v_k (1 - best_ncc^2) / 1.5, smoothed: the variance of k not shared
    with the best-matching image (noise + change + residual misregistration).
    G[k]: the tile contrast of k relative to the agreeing images; g = G / median of
    the upper half of the G's (about 1 for the clearest images).

    Returns dict with lists clear[k], noise2[k], best_ncc[k], arrays G, g.
    """
    n = len(O)
    if S is None:
        S = [np.ones(o.shape, bool) for o in O]
    out = dict(clear=[], noise2=[], ncc=[], g=np.ones(n), G=np.full(n, np.nan))
    valid = [m & s & np.isfinite(o) for o, m, s in zip(O, M, S)]
    if n == 1:
        out['clear'] = [valid[0]]
        out['noise2'] = [np.full(O[0].shape, np.nan, np.float32)]
        out['ncc'] = [np.full(O[0].shape, np.nan, np.float32)]
        return out
    shape = O[0].shape
    V = []
    for k in range(n):
        W, v, _, _ = local_moments(O[k], O[k], valid[k].astype(np.float32), sigma)
        v = np.where(W > min_w, v, np.nan).astype(np.float32)
        V.append(v)
    best = [np.full(shape, -1., np.float32) for _ in range(n)]
    for j in range(n):
        for k in range(j + 1, n):
            w = (valid[j] & valid[k]).astype(np.float32)
            if w.sum() < 100:
                continue
            W, vjj, vkk, vjk = local_moments(O[j], O[k], w, sigma)
            with np.errstate(invalid='ignore', divide='ignore'):
                ncc = np.where(W > min_w, vjk / np.sqrt(vjj * vkk), np.nan)
            ncc = np.nan_to_num(ncc, nan=-1.).astype(np.float32)
            np.maximum(best[j], ncc, out=best[j])
            np.maximum(best[k], ncc, out=best[k])
    best = [np.where(b > -1, b, np.nan) for b in best]
    Bst = np.stack(best)
    Vst = np.stack(V)
    with np.errstate(invalid='ignore'):
        established = np.nanmax(np.where(np.isfinite(Bst), Bst, -1), axis=0) > ncc_est
        agree = Bst > ncc_est
        ref_var = nanmedian0(np.where(agree, Vst, np.nan))
        v_min = np.nanmin(np.where(np.isfinite(Vst), Vst, np.inf), axis=0)
    v_min[~np.isfinite(v_min)] = np.nan
    # albedo against the spectrally clear median
    Ast = np.stack([np.where(v, a, np.nan) for a, v in zip(ALB, valid)])
    amed = nanmedian0(Ast)
    for k in range(n):
        b, v = best[k], V[k]
        with np.errstate(invalid='ignore', divide='ignore'):
            cloudy = established & ~(b >= ncc_min)
            cloudy |= established & (b >= ncc_min) & (v < contrast_min ** 2 * ref_var)
            cloudy |= ~established & (v > excess_tex * v_min + tex_min ** 2) & ~(b >= 0.5)
            alb_r, _ = masked_gaussian(np.where(valid[k], ALB[k] / amed, np.nan), sigma, valid[k])
            cloudy |= np.abs(alb_r - 1) > alb_tol
        clear = valid[k] & ~cloudy
        resid = np.where(np.isfinite(b) & np.isfinite(v), v * (1 - np.clip(b, 0, 1) ** 2) / 1.5, np.nan)
        noise2, _ = masked_gaussian(resid, noise_sigma, np.isfinite(resid) & clear)
        sel = established & (b >= ncc_est) & np.isfinite(ref_var) & np.isfinite(v)
        if sel.sum() > 500:
            out['G'][k] = float(np.sqrt(v[sel].sum() / ref_var[sel].sum()))
        out['clear'].append(clear)
        out['noise2'].append(noise2.astype(np.float32))
        out['ncc'].append(b.astype(np.float32))
    Gs = out['G'][np.isfinite(out['G'])]
    if Gs.size:
        top = Gs[Gs >= np.median(Gs)]
        out['g'] = np.where(np.isfinite(out['G']), out['G'] / np.median(top), np.nan)
    # images with very low contrast everywhere are hazy
    for k in range(n):
        if np.isfinite(out['g'][k]) and out['g'][k] < 0.3:
            out['clear'][k] = np.zeros(shape, bool)
    return out
