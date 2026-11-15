"""
Predicted direct-beam illumination of a DEM for a given sun position.

mu = max(0, n . s) for the surface normal n and the unit vector s toward the sun,
set to zero in cast shadows.  For a flat surface mu = sin(sun elevation).

Directions: the grid azimuth psi of the sun (clockwise from +y in EPSG:3031) is
the geographic azimuth plus the longitude (x = rho sin(lon), y = rho cos(lon)).
"""
import numpy as np
import numba
import scipy.ndimage as snd

W3 = 3.
L15 = 15.
SUB = 5     # 3-m pixels per 15-m pixel


def grid_azimuth(sun_az_geo, x, y):
    """Sun azimuth relative to grid north (+y), degrees, for EPSG:3031 points."""
    lon = np.degrees(np.arctan2(x, y))
    return np.mod(sun_az_geo + lon, 360.)


@numba.njit(parallel=True, cache=True)
def _cast_shadow(z, zup, step_c, step_r, tan_e, dx, max_steps):
    """
    True where the ray toward the sun from a pixel hits the surface.
    step_c, step_r: column/row increments of one step (|step| <= 1 pixel)
    zup: an upper bound on z in the neighbourhood (to stop marching early)
    """
    ny, nx = z.shape
    shadow = np.zeros((ny, nx), dtype=numba.boolean)
    step_len = dx * np.sqrt(step_c * step_c + step_r * step_r)
    rise = step_len * tan_e
    for r in numba.prange(ny):
        for c in range(nx):
            z0 = z[r, c]
            if not np.isfinite(z0):
                continue
            n_max = int((zup[r, c] - z0) / rise) + 1
            if n_max > max_steps:
                n_max = max_steps
            rr = float(r)
            cc = float(c)
            h = z0
            for k in range(n_max):
                rr += step_r
                cc += step_c
                h += rise
                ir = int(rr + 0.5)
                ic = int(cc + 0.5)
                if ir < 0 or ir >= ny or ic < 0 or ic >= nx:
                    break
                zz = z[ir, ic]
                if np.isfinite(zz) and zz > h:
                    shadow[r, c] = True
                    break
    return shadow


@numba.njit(parallel=True, cache=True)
def _mu_kernel(z, dx, sx, sy, sz):
    """cos(incidence) with central differences (one-sided at edges / next to NaNs)."""
    ny, nx = z.shape
    mu = np.empty((ny, nx), dtype=np.float32)
    for r in numba.prange(ny):
        for c in range(nx):
            z0 = z[r, c]
            if not np.isfinite(z0):
                mu[r, c] = np.nan
                continue
            # x derivative (columns)
            zl = z[r, c - 1] if c > 0 else np.nan
            zr = z[r, c + 1] if c < nx - 1 else np.nan
            if np.isfinite(zl) and np.isfinite(zr):
                zx = (zr - zl) / (2 * dx)
            elif np.isfinite(zr):
                zx = (zr - z0) / dx
            elif np.isfinite(zl):
                zx = (z0 - zl) / dx
            else:
                zx = 0.
            # y derivative (rows go toward -y)
            zu = z[r - 1, c] if r > 0 else np.nan
            zd = z[r + 1, c] if r < ny - 1 else np.nan
            if np.isfinite(zu) and np.isfinite(zd):
                zy = (zu - zd) / (2 * dx)
            elif np.isfinite(zu):
                zy = (zu - z0) / dx
            elif np.isfinite(zd):
                zy = (z0 - zd) / dx
            else:
                zy = 0.
            m = (-zx * sx - zy * sy + sz) / np.sqrt(1. + zx * zx + zy * zy)
            mu[r, c] = m if m > 0 else 0.
    return mu


@numba.njit(parallel=True, cache=True)
def _block_mean_kernel(a, n, min_count):
    ny, nx = a.shape
    ny1, nx1 = ny // n, nx // n
    out = np.full((ny1, nx1), np.nan, dtype=np.float32)
    frac = np.zeros((ny1, nx1), dtype=np.float32)
    for i in numba.prange(ny1):
        for j in range(nx1):
            s = 0.
            cnt = 0
            for ii in range(i * n, i * n + n):
                for jj in range(j * n, j * n + n):
                    v = a[ii, jj]
                    if np.isfinite(v):
                        s += v
                        cnt += 1
            frac[i, j] = cnt / (n * n)
            if cnt >= min_count:
                out[i, j] = s / cnt
    return out, frac


def _zup(z, dx, max_shadow_dist, dec=20):
    """Upper bound of z within max_shadow_dist (coarse maximum filter)."""
    zf = np.where(np.isfinite(z), z, -1.e4).astype(np.float32)
    ny, nx = z.shape
    pad_r, pad_c = (-ny) % dec, (-nx) % dec
    zc = np.pad(zf, ((0, pad_r), (0, pad_c)), constant_values=-1.e4)
    zc = zc.reshape(zc.shape[0] // dec, dec, zc.shape[1] // dec, dec).max(axis=(1, 3))
    n_win = int(2 * max_shadow_dist / (dec * dx)) + 3
    zc = snd.maximum_filter(zc, size=n_win, mode='nearest')
    return zc, dec


@numba.njit(parallel=True, cache=True)
def _cast_shadow2(z, zc, dec, step_c, step_r, tan_e, dx, max_steps):
    """Like _cast_shadow, with the coarse upper-bound grid zc (block size dec)."""
    ny, nx = z.shape
    shadow = np.zeros((ny, nx), dtype=numba.boolean)
    step_len = dx * np.sqrt(step_c * step_c + step_r * step_r)
    rise = step_len * tan_e
    for r in numba.prange(ny):
        for c in range(nx):
            z0 = z[r, c]
            if not np.isfinite(z0):
                continue
            zu = zc[r // dec, c // dec]
            if zu <= z0:
                continue
            n_max = int((zu - z0) / rise) + 1
            if n_max > max_steps:
                n_max = max_steps
            rr = float(r)
            cc = float(c)
            h = z0
            for k in range(n_max):
                rr += step_r
                cc += step_c
                h += rise
                ir = int(rr + 0.5)
                ic = int(cc + 0.5)
                if ir < 0 or ir >= ny or ic < 0 or ic >= nx:
                    break
                zz = z[ir, ic]
                if zz > h:
                    shadow[r, c] = True
                    break
    return shadow


def illumination(z, sun_elev, psi, dx=W3, shadows=True, max_shadow_dist=2000., zup=None):
    """
    Direct illumination mu (0..1) on the grid of z (rows decreasing in y).

    sun_elev : degrees; psi : grid azimuth of the sun, degrees
    zup: optional (zc, dec) from _zup, reused between sun positions
    Returns mu (NaN where z is NaN), and the shadow mask.
    """
    e = np.radians(sun_elev)
    p = np.radians(psi)
    sx, sy, sz = np.cos(e) * np.sin(p), np.cos(e) * np.cos(p), np.sin(e)
    z = np.ascontiguousarray(z, dtype=np.float32)
    mu = _mu_kernel(z, float(dx), sx, sy, sz)
    shadow = None
    if shadows:
        if zup is None:
            zup = _zup(z, dx, max_shadow_dist)
        zc, dec = zup
        sc, sr = np.sin(p), -np.cos(p)
        m = max(abs(sc), abs(sr))
        sc, sr = sc / m, sr / m
        max_steps = int(max_shadow_dist / (dx * np.hypot(sc, sr))) + 1
        shadow = _cast_shadow2(z, zc, dec, sc, sr, np.tan(e), float(dx), max_steps)
        mu[shadow] = 0.
    return mu, shadow


def block_mean(a, n=SUB, min_frac=0.6):
    """n x n block mean of a (NaN-aware); NaN where less than min_frac of the block is valid."""
    out, frac = _block_mean_kernel(np.ascontiguousarray(a, dtype=np.float32), n, int(np.ceil(min_frac * n * n)))
    return out, frac


def masked_gaussian(a, sigma, mask=None, min_weight=0.2):
    """Normalized (NaN-aware) Gaussian filter."""
    if mask is None:
        mask = np.isfinite(a)
    w = snd.gaussian_filter(mask.astype(np.float32), sigma, mode='constant', cval=0, truncate=3.)
    f = snd.gaussian_filter(np.where(mask, a, 0).astype(np.float32), sigma, mode='constant', cval=0, truncate=3.)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = w > min_weight
    out[ok] = f[ok] / w[ok]
    return out, w


def predicted_image(z3, sun_elev, psi, psf_sigma=0.5, shadows=True, zup=None):
    """
    The predicted 15-m direct illumination for a 3-m DEM whose pixel edges match
    the 15-m pan pixels.

    psf_sigma: extra Gaussian blur (15-m pixels) on top of the 15-m box (the OLI PSF).
    Returns mu15 (NaN where the DEM is mostly missing), the shadow fraction, and the
    DEM valid fraction, all on the 15-m grid.
    """
    mu, shadow = illumination(z3, sun_elev, psi, shadows=shadows, zup=zup)
    mu15, valid_frac = block_mean(mu)
    if shadow is not None:
        sh15, _ = block_mean(np.where(np.isfinite(mu), shadow.astype(np.float32), np.nan))
    else:
        sh15 = np.zeros_like(mu15)
    if psf_sigma > 0:
        good = np.isfinite(mu15)
        mu15b, _ = masked_gaussian(mu15, psf_sigma, good, min_weight=0.5)
        mu15b[~good] = np.nan
        mu15 = mu15b
    return mu15, sh15, valid_frac
