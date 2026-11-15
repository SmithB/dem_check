"""
Check one REMA strip DEM against Landsat 8/9 pan images.

Steps
  1. find the 2-m DEM (or download it), its time and footprint
  2. find Landsat acquisitions within +-max_dt_days, download windowed chips
  3. tile by tile (15 km, padded):
       - read the DEM onto a 3-m grid, predict the 15-m illumination for each acquisition
       - advect each image to the DEM time, register images to each other and the
         stack to the DEM, find clouds from image-to-image consistency
       - local moments of each clear image against its prediction
  4. per-image gains, pooled F / SNR, classification, summary, outputs

Outputs (in out_dir/<strip base>/)
  <base>_ldcheck.tif   15-m pixels, EPSG:3031; bands: class, F, SNR, r, pred_amp, image_amp,
                       unconfirmed_amp, n_clear, DEM_valid_frac, matched_frac
  <base>_ldcheck.json  summary (acquisitions, gains, registration, class fractions)
  <base>_ldcheck.png   quicklook
"""
import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import pyproj
import shapely.geometry
from osgeo import gdal

from . import landsat, dem, shading, stack, compare
from .velocity import VelocityField

gdal.UseExceptions()
L15 = 15.

DEFAULTS = dict(max_dt_days=45., n_max=6, min_sun_elev=8., max_cloud=95.,
                tile_size=15000., tile_pad=900., hp_sigma=200., stat_sigma=6.,
                psf_sigma=0.75, max_global_shift=20, block=96, block_max_shift=6,
                velocity='itslive', velocity_stack=None,
                chip_root='/Volumes/ice1/ben/REMA/landsat_DEM_check/chips',
                snr_min=3., F_good=0.5, F_bad=0.25, cloud_ncc_min=0.4,
                tex_min=0.002, noise_floor=0.0005, single_image_noise=0.002, amp_min=0.005,
                f_clear_default=0.85, f_clear_min=0.7, f_clear_max=1.0,
                save_debug=False, roi=None)


def select_acquisitions(acqs, n_max):
    """Prefer acquisitions close in time and with low USGS cloud cover."""
    for a in acqs:
        a['score'] = abs(a['dt_days']) / 20. + a['cloud_cover'] / 40.
    acqs = sorted(acqs, key=lambda a: a['score'])
    return acqs[:n_max]


def _write_tif(fname, bands, bounds, names, nodata=np.nan):
    ny, nx = bands[0].shape
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(fname, nx, ny, len(bands), gdal.GDT_Float32,
                    options=['COMPRESS=DEFLATE', 'TILED=YES', 'PREDICTOR=3'])
    ds.SetGeoTransform((bounds[0], L15, 0, bounds[3], 0, -L15))
    ds.SetProjection(pyproj.CRS.from_epsg(3031).to_wkt())
    for i, (b, n) in enumerate(zip(bands, names)):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(b.astype(np.float32))
        band.SetDescription(n)
        band.SetNoDataValue(nodata)
    ds = None


def _tile_sun(el, az, sl):
    e = np.nanmean(el[sl])
    a = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(az[sl]))), np.nanmean(np.cos(np.radians(az[sl])))))
    return float(e), float(a)


def _register_tile(imgs, preds, V15, xg, yg, vel_dxdy, P, log, S=None):
    """
    Register the advected images of one tile.
    imgs: list of dicts with rho, gt, el (scalar), dt (years)
    preds: list of predicted mu15 for each image
    Returns the shift fields (meters, x and y) for each image, and diagnostics.
    """
    n = len(imgs)
    hp_px = P['hp_sigma'] / L15
    shape = xg.shape
    Sx = [np.zeros(shape, np.float32) for _ in range(n)]
    Sy = [np.zeros(shape, np.float32) for _ in range(n)]
    diag = [dict() for _ in range(n)]

    def sample(k):
        x = xg + vel_dxdy[0] * imgs[k]['dt'] + Sx[k]
        y = yg + vel_dxdy[1] * imgs[k]['dt'] + Sy[k]
        r = stack.sample_image(imgs[k]['rho'], imgs[k]['gt'], x, y)
        m = np.isfinite(r)
        if S is not None:
            m &= S[k]
        o, alb = stack.normalize_image(r, imgs[k]['el'], hp_px, m)
        return o, alb, m

    P_hp = []
    for k in range(n):
        p, _ = stack.high_pass(preds[k], hp_px, V15 & np.isfinite(preds[k]))
        P_hp.append(p)
    vmask = V15 & np.isfinite(P_hp[0])

    # 1. global shift of each image against its own DEM prediction
    O = []
    for k in range(n):
        o, alb, m = sample(k)
        dr, dc, pk, ok = stack.global_shift(P_hp[k], o, vmask, m & np.isfinite(o),
                                            max_shift=P['max_global_shift'], decimate=1)
        diag[k].update(g_dem_peak=float(pk) if np.isfinite(pk) else np.nan,
                       g_dem_dx=float(dc * L15) if np.isfinite(dc) else np.nan,
                       g_dem_dy=float(-dr * L15) if np.isfinite(dr) else np.nan,
                       g_dem_ok=bool(ok), valid_frac=float(m[V15].mean()) if V15.any() else 0.)
        O.append((o, m))
    peaks = np.array([d['g_dem_peak'] if d['g_dem_ok'] else -1 for d in diag])
    peaks[~np.isfinite(peaks)] = -1
    dem_reg_ok = peaks.max() >= P.get('min_global_peak', 0.15)
    if dem_reg_ok:
        ref = int(np.argmax(peaks + 0.2 * np.array([d['valid_frac'] for d in diag])))
    else:
        ref = int(np.argmax([d['valid_frac'] - abs(imgs[k]['dt']) for k, d in enumerate(diag)]))
    for k in range(n):
        if diag[k]['g_dem_ok'] and peaks[k] >= P.get('min_global_peak', 0.15):
            Sx[k] += diag[k]['g_dem_dx']
            Sy[k] += diag[k]['g_dem_dy']
            diag[k]['global_source'] = 'dem'
        else:
            diag[k]['global_source'] = 'none'
    # 2. images that did not register to the DEM: global shift against the reference image
    o_ref, a_ref, m_ref = sample(ref)
    for k in range(n):
        if k == ref or diag[k]['global_source'] == 'dem':
            continue
        o, alb, m = sample(k)
        dr, dc, pk, ok = stack.global_shift(o_ref, o, m_ref & np.isfinite(o_ref), m & np.isfinite(o),
                                            max_shift=P['max_global_shift'])
        diag[k]['g_img_peak'] = float(pk) if np.isfinite(pk) else np.nan
        if ok and pk >= 0.2:
            Sx[k] += dc * L15
            Sy[k] -= dr * L15
            diag[k]['global_source'] = 'image'
    # 3. block shifts of each image against the reference image
    o_ref, a_ref, m_ref = sample(ref)
    for k in range(n):
        if k == ref:
            continue
        o, alb, m = sample(k)
        rows, cols, DR, DC, PK = stack.block_shifts(o_ref, o, m_ref & np.isfinite(o_ref), m & np.isfinite(o),
                                                    block=P['block'], max_shift=P['block_max_shift'],
                                                    min_peak=0.3)
        dr, dc, fgood = stack.smooth_shift_field(rows, cols, DR, DC, PK, shape)
        Sx[k] += (dc * L15).astype(np.float32)
        Sy[k] -= (dr * L15).astype(np.float32)
        diag[k]['img_block_frac'] = float(fgood)
        diag[k]['img_block_rms'] = float(np.sqrt(np.nanmean(DR**2 + DC**2)) * L15) if np.isfinite(DR).any() else np.nan
    # 4. the stack (median) against the reference prediction, in blocks
    diag_stack = dict(ref=ref, dem_reg_ok=bool(dem_reg_ok))
    if dem_reg_ok:
        Os = []
        for k in range(n):
            o, alb, m = sample(k)
            Os.append(np.where(m, o, np.nan))
        with np.errstate(all='ignore'):
            C = stack.nanmedian0(np.stack(Os)) if n > 1 else Os[0]
        mc = np.isfinite(C)
        rows, cols, DR, DC, PK = stack.block_shifts(P_hp[ref], C, vmask, mc, block=P['block'],
                                                    max_shift=P['block_max_shift'], min_peak=0.25)
        dr, dc, fgood = stack.smooth_shift_field(rows, cols, DR, DC, PK, shape)
        for k in range(n):
            Sx[k] += (dc * L15).astype(np.float32)
            Sy[k] -= (dr * L15).astype(np.float32)
        diag_stack.update(dem_block_frac=float(fgood),
                          dem_block_med_dx=float(np.nanmedian(DC) * L15) if np.isfinite(DC).any() else np.nan,
                          dem_block_med_dy=float(-np.nanmedian(DR) * L15) if np.isfinite(DR).any() else np.nan,
                          dem_block_rms=float(np.sqrt(np.nanmean(DR**2 + DC**2)) * L15) if np.isfinite(DR).any() else np.nan,
                          dem_block_peak=float(np.nanmedian(PK)) if np.isfinite(PK).any() else np.nan)
    # final samples
    out = []
    for k in range(n):
        o, alb, m = sample(k)
        out.append((o, alb, m))
    return out, P_hp, Sx, Sy, diag, diag_stack


def check_dem(dem_name, out_dir, **kwargs):
    P = dict(DEFAULTS)
    P.update({k: v for k, v in kwargs.items() if v is not None})
    t_start = time.time()
    base = dem.strip_base(dem_name)
    this_out = os.path.join(out_dir, base)
    os.makedirs(this_out, exist_ok=True)
    log_file = open(os.path.join(this_out, base + '_ldcheck.log'), 'a')

    def log(*args):
        msg = ' '.join(str(a) for a in args)
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    summary = dict(dem=base, params={k: v for k, v in P.items()}, status='started',
                   start=datetime.datetime.now().isoformat())
    json_file = os.path.join(this_out, base + '_ldcheck.json')

    def finish(status):
        summary['status'] = status
        summary['run_time_s'] = time.time() - t_start
        with open(json_file, 'w') as fh:
            json.dump(summary, fh, indent=1, default=float)
        log(f'{base}: {status} ({summary["run_time_s"]:.0f} s)')
        log_file.close()
        return summary

    files = dem.find_2m_files(dem_name)
    if files is None:
        log(f'downloading 2-m files for {base}')
        files = dem.download_2m(dem_name)
    summary['files'] = files
    t_dem = dem.strip_time(base)
    ty = landsat.decimal_year(t_dem)
    summary['t_dem'] = t_dem.isoformat()
    bounds, poly = dem.dem_footprint(files['dem'])
    if bounds is None:
        return finish('empty_dem')
    if P.get('roi') is not None:
        roi = [float(v) for v in str(P['roi']).split(',')] if isinstance(P['roi'], str) else list(P['roi'])
        bounds = [max(bounds[0], roi[0]), max(bounds[1], roi[1]), min(bounds[2], roi[2]), min(bounds[3], roi[3])]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return finish('roi_outside_dem')
        poly = poly.intersection(shapely.geometry.box(*bounds))
    xf = pyproj.Transformer.from_crs(3031, 4326, always_xy=True)
    xy = np.array(poly.exterior.coords)
    lon, lat = xf.transform(xy[:, 0], xy[:, 1])
    summary['bounds'] = bounds

    scenes = landsat.search_scenes(list(zip(lon, lat)), t_dem, max_dt_days=P['max_dt_days'],
                                   min_sun_elev=P['min_sun_elev'], max_cloud=P['max_cloud'])
    acqs = landsat.group_acquisitions(scenes)
    log(f'{base}: t={t_dem.isoformat()}, {len(scenes)} scenes in {len(acqs)} acquisitions')
    acqs = select_acquisitions(acqs, P['n_max'])
    if len(acqs) == 0:
        return finish('no_images')

    vel = VelocityField(bounds, source=P['velocity'], stack_file=P['velocity_stack'], t=ty)
    summary['velocity_source'] = vel.source
    # pad the chips to cover the advection and registration
    xs = np.linspace(bounds[0], bounds[2], 40)
    ys = np.linspace(bounds[1], bounds[3], 40)
    XX, YY = np.meshgrid(xs, ys)
    vx, vy = vel(XX, YY)
    vmax = float(np.nanmax(np.hypot(vx, vy)))
    dt_max = max(abs(a['dt_days']) for a in acqs) / 365.25
    pad = vmax * dt_max + P['max_global_shift'] * L15 + P['tile_pad'] + 300.
    chip_bounds = dem.snap_bounds(bounds, pad)
    G = dem.snap_bounds(bounds, 0.)
    nx = int(round((G[2] - G[0]) / L15))
    ny = int(round((G[3] - G[1]) / L15))
    log(f'grid {nx} x {ny} (15 m), vmax={vmax:.0f} m/yr, chip pad={pad:.0f} m')

    imgs = []
    for a in acqs:
        chip_dir = os.path.join(P['chip_root'], base, a['key'])
        try:
            chips = landsat.get_acquisition_chips(a, chip_bounds, chip_dir, footprint=poly.buffer(pad))
        except landsat.ERSLoginBlocked as e:
            log(f"\t{a['key']}: {e}")
            return finish('login_blocked')
        except Exception as e:
            log(f"\t{a['key']}: chip read failed: {e}")
            continue
        if not chips:
            continue
        A = stack.load_acquisition(chips, chip_bounds)
        rho, gt, scloud = A['rho'], A['gt'], A['scloud']
        if np.isfinite(rho).mean() < 0.01:
            continue
        imgs.append(dict(key=a['key'], rho=rho, gt=gt, time=A['time'], scloud=scloud, gt60=A['gt60'],
                         scloud_frac=float(scloud.mean()) if scloud is not None else np.nan,
                         dt=(a['t_year'] - ty), dt_days=a['dt_days'],
                         t_year=a['t_year'], cloud_cover=a['cloud_cover'],
                         scenes=[c['id'] for c in chips]))
        log(f"\t{a['key']}: dt={a['dt_days']:+.1f} d, sun_el={a['sun_elev']:.1f}, cloud={a['cloud_cover']:.0f}%, "
            f"valid={np.isfinite(rho).mean():.2f}, spectral cloud={imgs[-1]['scloud_frac']:.2f}")
    if not imgs:
        return finish('no_image_data')
    n = len(imgs)
    summary['acquisitions'] = [{k: v for k, v in im.items() if k not in ('rho', 'time', 'gt', 'gt60', 'scloud')} for im in imgs]

    # strip-level accumulators.  Pooled sums are kept without the clear-image gain
    # f_clear (a scalar found at the end): with f_k = f_clear * g_k and weights 1/noise2,
    #   A = sum g vop / n2,  B = sum g^2 vpp / n2,  C = sum voo / n2,  D = sum 1 / n2
    #   F = A / (f_clear B),  SNR = f_clear sqrt(B),  r = A / sqrt(B C)
    shp = (ny, nx)
    acc = dict(A=np.zeros(shp, np.float32), B=np.zeros(shp, np.float32), C=np.zeros(shp, np.float32),
               D=np.zeros(shp, np.float32), n_clear=np.zeros(shp, np.int16),
               valid=np.zeros(shp, np.float32), matched=np.full(shp, np.nan, np.float32))
    gain_samples = [dict(vop=[], vpp=[], g=[]) for _ in range(n)]
    clear_count = np.zeros(n)
    if P['save_debug']:
        dbg = dict(o=np.full((n,) + shp, np.nan, np.float32), p=np.full((n,) + shp, np.nan, np.float32),
                   shift_x=np.full((n,) + shp, np.nan, np.float32), shift_y=np.full((n,) + shp, np.nan, np.float32),
                   clear=np.zeros((n,) + shp, bool), noise2=np.full((n,) + shp, np.nan, np.float32))
    hp_px = P['hp_sigma'] / L15
    pad_px = int(P['tile_pad'] / L15)
    floor2 = P['noise_floor'] ** 2
    # tile edges, without slivers
    n_tr = max(1, int(round(ny * L15 / P['tile_size'])))
    n_tc = max(1, int(round(nx * L15 / P['tile_size'])))
    r_edges = np.linspace(0, ny, n_tr + 1).astype(int)
    c_edges = np.linspace(0, nx, n_tc + 1).astype(int)
    tiles_diag = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        for c0, c1 in zip(c_edges[:-1], c_edges[1:]):
            t0 = time.time()
            R0, C0 = max(r0 - pad_px, 0), max(c0 - pad_px, 0)
            R1, C1 = min(r1 + pad_px, ny), min(c1 + pad_px, nx)
            tb = [G[0] + C0 * L15, G[3] - R1 * L15, G[0] + C1 * L15, G[3] - R0 * L15]
            if not poly.intersects(shapely.geometry.box(*tb)):
                continue
            d3 = dem.read_dem_3m(files, tb)
            z3 = d3['z']
            if np.isfinite(z3).mean() < 0.01:
                continue
            valid_frac15 = shading.block_mean(np.where(np.isfinite(z3), 1., np.nan))[1]
            V15 = valid_frac15 >= 0.6
            if V15.sum() < 2000:
                continue
            xc = G[0] + L15 * (np.arange(C0, C1) + 0.5)
            yc = G[3] - L15 * (np.arange(R0, R1) + 0.5)
            xg, yg = np.meshgrid(xc, yc)
            vxg, vyg = vel(xg, yg)
            # images covering this tile
            tile_imgs, preds = [], []
            zup = shading._zup(z3, shading.W3, 2000.)
            for k, im in enumerate(imgs):
                gt = im['gt']
                cc0 = int((tb[0] - gt[0]) / gt[1])
                cc1 = int((tb[2] - gt[0]) / gt[1])
                rr0 = int((tb[3] - gt[3]) / gt[5])
                rr1 = int((tb[1] - gt[3]) / gt[5])
                sl = (slice(max(rr0, 0), rr1), slice(max(cc0, 0), cc1))
                cover = np.isfinite(im['rho'][sl]).mean() if im['rho'][sl].size else 0
                if cover < 0.05:
                    continue
                el, az = stack.sun_at(im['time'], float(xg.mean()), float(yg.mean()))
                el, az = float(el), float(az)
                psi = float(shading.grid_azimuth(az, xg.mean(), yg.mean()))
                mu15, sh15, _ = shading.predicted_image(z3, el, psi, psf_sigma=P['psf_sigma'], zup=zup)
                tile_imgs.append(dict(k=k, rho=im['rho'], gt=gt, el=el, psi=psi, dt=im['dt'], scloud=im['scloud'],
                                      gt60=im['gt60']))
                preds.append(mu15)
            if not tile_imgs:
                continue
            # spectral clouds, advected like the images (nearest neighbour)
            S = []
            for ti in tile_imgs:
                if ti['scloud'] is None:
                    S.append(np.ones(xg.shape, bool))
                    continue
                xs = xg + vxg * ti['dt']
                ys = yg + vyg * ti['dt']
                gtc = ti['gt60']
                cc = np.clip(((xs - gtc[0]) / gtc[1]).astype(int), 0, ti['scloud'].shape[1] - 1)
                rr = np.clip(((ys - gtc[3]) / gtc[5]).astype(int), 0, ti['scloud'].shape[0] - 1)
                S.append(~ti['scloud'][rr, cc])
            # drop images that are (spectrally) cloudy over almost all of the tile
            keep = [j for j in range(len(tile_imgs)) if (S[j] & V15).sum() >= 0.03 * V15.sum()]
            dropped = [imgs[tile_imgs[j]['k']]['key'] for j in range(len(tile_imgs)) if j not in keep]
            tile_imgs = [tile_imgs[j] for j in keep]
            preds = [preds[j] for j in keep]
            S = [S[j] for j in keep]
            if not tile_imgs:
                tiles_diag.append(dict(r0=int(r0), c0=int(c0), images=[], dropped=dropped))
                continue
            res, P_hp, Sx, Sy, rdiag, sdiag = _register_tile(tile_imgs, preds, V15, xg, yg, (vxg, vyg), P, log, S=S)
            O = [r[0] for r in res]
            ALB = [r[1] for r in res]
            M = [r[2] & np.isfinite(r[0]) for r in res]
            nt = len(O)
            cs = stack.cloud_screen(O, ALB, M, S=S, sigma=8., ncc_min=P['cloud_ncc_min'], tex_min=P['tex_min'])
            inner = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
            outer = (slice(r0, r1), slice(c0, c1))
            for j, ti in enumerate(tile_imgs):
                k = ti['k']
                o = O[j]
                clear = cs['clear'][j]
                w = (clear & V15 & np.isfinite(P_hp[j])).astype(np.float32)
                p_hp, _ = stack.high_pass(preds[j], hp_px, w > 0)
                o_hp, _ = stack.high_pass(np.where(w > 0, o, np.nan), hp_px, w > 0)
                mom = compare.image_moments(o_hp, p_hp, w, P['stat_sigma'])
                ok = (mom['W'] > 0.5) & clear & V15
                gk = float(cs['g'][j]) if np.isfinite(cs['g'][j]) else 1.
                n2 = cs['noise2'][j]
                n2 = np.where(np.isfinite(n2), n2, P['single_image_noise'] ** 2 if nt == 1 else np.nan)
                n2 = np.maximum(n2, floor2)
                ok &= np.isfinite(n2)
                inv = np.where(ok, 1. / n2, 0.).astype(np.float32)
                vop = np.where(ok, mom['vop'], 0.)
                vpp = np.where(ok, mom['vpp'], 0.)
                voo = np.where(ok, mom['voo'], 0.)
                acc['A'][outer] += (inv * gk * vop)[inner]
                acc['B'][outer] += (inv * gk * gk * vpp)[inner]
                acc['C'][outer] += (inv * voo)[inner]
                acc['D'][outer] += inv[inner]
                acc['n_clear'][outer] += ok[inner]
                clear_count[k] += ok[inner].sum()
                # subsample for the DEM-based gain estimate
                sub = (slice(None, None, 3), slice(None, None, 3))
                okk = ok[inner][sub]
                gain_samples[k]['vop'].append(mom['vop'][inner][sub][okk])
                gain_samples[k]['vpp'].append(mom['vpp'][inner][sub][okk])
                gain_samples[k]['g'].append(np.full(okk.sum(), gk, np.float32))
                rdiag[j].update(G_tile=float(cs['G'][j]), g_tile=float(cs['g'][j]),
                                clear_frac=float((clear & V15).sum() / max(V15.sum(), 1)))
                if P['save_debug']:
                    dbg['o'][k][outer] = np.where(M[j], o, np.nan)[inner]
                    dbg['p'][k][outer] = P_hp[j][inner]
                    dbg['shift_x'][k][outer] = Sx[j][inner]
                    dbg['shift_y'][k][outer] = Sy[j][inner]
                    dbg['clear'][k][outer] = ok[inner]
                    dbg['noise2'][k][outer] = n2[inner]
            acc['valid'][outer] = valid_frac15[inner]
            if 'matched' in d3:
                acc['matched'][outer] = shading.block_mean(d3['matched'])[0][inner]
            tiles_diag.append(dict(r0=int(r0), c0=int(c0), images=[imgs[t['k']]['key'] for t in tile_imgs], dropped=dropped,
                                   reg=rdiag, stack=sdiag, time=time.time() - t0))
            log(f'\ttile r{r0} c{c0}: {nt} images, ref={imgs[tile_imgs[sdiag["ref"]]["k"]]["key"]}, '
                f'dem_reg_ok={sdiag["dem_reg_ok"]}, peaks=' +
                ','.join(f'{d["g_dem_peak"]:.2f}' for d in rdiag) + ', g=' +
                ','.join(f'{g:.2f}' for g in cs['g']) + f', {time.time()-t0:.0f} s')
    summary['tiles'] = tiles_diag
    if not any(t['images'] for t in tiles_diag):
        return finish('no_overlap' if not tiles_diag else 'all_cloudy')

    # ---- the clear-image gain f_clear: from the DEM (each image's gain over its
    # strongest predicted shading, divided by its relative gain g), limited to
    # [f_clear_min, f_clear_max] so that a bad DEM cannot lower it much.
    V = acc['valid'] >= 0.6
    f_dem = np.full(n, np.nan)
    g_img = np.full(n, np.nan)
    for k in range(n):
        if not gain_samples[k]['vop']:
            continue
        vop = np.concatenate(gain_samples[k]['vop'])
        vpp = np.concatenate(gain_samples[k]['vpp'])
        gg = np.concatenate(gain_samples[k]['g'])
        if vop.size < 100:
            continue
        f_dem[k], _ = compare.estimate_gain(vop, vpp, np.ones_like(vop))
        g_img[k] = float(np.median(gg))
        imgs[k].update(gain_dem=float(f_dem[k]), g_rel=float(g_img[k]))
    for k in range(n):
        imgs[k]['clear_frac'] = float(clear_count[k] / max(V.sum(), 1))
    use = np.isfinite(f_dem) & (g_img > 0.8)
    f_clear_dem = float(np.median(f_dem[use] / g_img[use])) if use.any() else np.nan
    f_clear = float(np.clip(f_clear_dem if np.isfinite(f_clear_dem) else P['f_clear_default'],
                            P['f_clear_min'], P['f_clear_max']))
    summary.update(f_clear_dem=f_clear_dem, f_clear=f_clear)
    summary['acquisitions'] = [{k: v for k, v in im.items() if k not in ('rho', 'time', 'gt', 'gt60', 'scloud')}
                               for im in imgs]
    # ---- pooled statistics
    A, B, C, D = (acc[k].astype(np.float64) for k in 'ABCD')
    n_clear = acc['n_clear']
    with np.errstate(invalid='ignore', divide='ignore'):
        F = (A / (f_clear * B)).astype(np.float32)
        SNR = (f_clear * np.sqrt(B)).astype(np.float32)
        r = (A / np.sqrt(B * C)).astype(np.float32)
        amp = (f_clear * np.sqrt(B / D)).astype(np.float32)      # predicted image std (mu units)
        img_amp = np.sqrt(C / D).astype(np.float32)               # observed image std (mu units)
        img_snr = np.sqrt(C).astype(np.float32)
        U = (amp * np.sqrt(np.clip(1. - F, 0., 1.))).astype(np.float32)
    for q in (F, SNR, r, amp, img_amp, img_snr, U):
        q[n_clear == 0] = np.nan
    cls = compare.classify(F, SNR, img_snr, n_clear, amp=amp, img_amp=img_amp, snr_min=P['snr_min'],
                           amp_min=P['amp_min'], F_good=P['F_good'], F_bad=P['F_bad'])
    cls[~V] = 0
    summary.update(compare.summarize(cls, F, SNR, V))
    summary['frac_with_clear_image'] = float(((n_clear > 0) & V).sum() / max(V.sum(), 1))
    summary['frac_matched'] = float(np.nanmean(acc['matched'][V])) if V.any() else np.nan
    summary['valid_area_km2'] = float(V.sum() * L15 * L15 / 1e6)
    sig = V & (cls >= 2) & (cls <= 4)
    mt = acc['matched']
    for name, sel in (('matched', mt >= 0.5), ('unmatched', mt < 0.5)):
        ns = (sig & sel).sum()
        summary[f'frac_signal_{name}'] = float(ns / max((V & sel).sum(), 1))
        summary[f'frac_inconsistent_of_signal_{name}'] = float((sig & sel & (cls == 4)).sum() / ns) if ns else np.nan
    out_tif = os.path.join(this_out, base + '_ldcheck.tif')
    _write_tif(out_tif, [cls.astype(np.float32), F, SNR, r, amp, img_amp, U, n_clear.astype(np.float32),
                         acc['valid'], acc['matched']],
               G, ['class', 'F', 'SNR', 'r', 'pred_amp', 'image_amp', 'unconfirmed_amp', 'n_clear',
                   'DEM_valid_frac', 'matched_frac'])
    sig = V & (cls >= 2) & (cls <= 4)
    summary['U_p90_signal'] = float(np.nanpercentile(U[sig], 90)) if sig.sum() > 10 else np.nan
    summary['U_area_gt_0.01'] = float(((U > 0.01) & V).sum() * L15 * L15 / 1e6)
    summary['U_frac_gt_0.01'] = float(((U > 0.01) & V).sum() / max(V.sum(), 1))
    summary['U_frac_gt_0.02'] = float(((U > 0.02) & V).sum() / max(V.sum(), 1))
    summary['out_tif'] = out_tif
    if P['save_debug']:
        import h5py
        with h5py.File(os.path.join(this_out, base + '_ldcheck_debug.h5'), 'w') as h:
            for key, val in dbg.items():
                h.create_dataset(key, data=val, compression='gzip')
            for key in 'ABCD':
                h.create_dataset(key, data=acc[key], compression='gzip')
            h.create_dataset('matched', data=acc['matched'], compression='gzip')
            h.attrs['bounds'] = G
            h.attrs['keys'] = [im['key'] for im in imgs]
    try:
        from .quicklook import quicklook
        quicklook(out_tif, os.path.join(this_out, base + '_ldcheck.png'), summary)
    except Exception as e:
        log(f'quicklook failed: {e}')
    return finish('done')


def main(argv=None):
    parser = argparse.ArgumentParser(description='check a REMA strip DEM against Landsat pan images',
                                     fromfile_prefix_chars='@')
    parser.add_argument('dem', help='strip DEM name or file (any resolution)')
    parser.add_argument('--out_dir', default='/Volumes/ice1/ben/REMA/landsat_DEM_check/qa')
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument('--' + k, action='store_true', default=None)
        elif v is None:
            parser.add_argument('--' + k, type=str, default=None)
        else:
            parser.add_argument('--' + k, type=type(v), default=None)
    args = parser.parse_args(argv)
    kw = {k: getattr(args, k) for k in DEFAULTS}
    try:
        check_dem(args.dem, args.out_dir, **kw)
    except Exception as e:
        import traceback
        base = dem.strip_base(args.dem)
        os.makedirs(os.path.join(args.out_dir, base), exist_ok=True)
        with open(os.path.join(args.out_dir, base, base + '_ldcheck.json'), 'w') as fh:
            json.dump(dict(dem=base, status='error', error=repr(e), traceback=traceback.format_exc()), fh, indent=1)
        raise


if __name__ == '__main__':
    main()
