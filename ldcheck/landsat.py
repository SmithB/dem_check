"""
Landsat 8/9 Collection-2 Level-1 access.

Scenes are found with the USGS STAC server (landsatlook), and read with GDAL
windowed reads of the cloud-optimized GeoTIFFs.  The landsatlook data URLs need
a USGS ERS login; the credentials come from ~/.netrc (machine ers.cr.usgs.gov),
as in landsatutilities/download_Landsat_urls.py.  The login cookies are handed to
GDAL with GDAL_HTTP_COOKIE.

Over Antarctica the Level-1 products are delivered on a polar stereographic grid
(standard parallel -71, central meridian 0, i.e. EPSG:3031) with 15-m pan pixel
centers on multiples of 15 m.
"""
import os
import re
import json
import time
import netrc
import datetime
import threading
import numpy as np
import requests
from osgeo import gdal, osr

gdal.UseExceptions()

STAC_URL = "https://landsatlook.usgs.gov/stac-server"
ERS_LOGIN = "https://ers.cr.usgs.gov/login"

_session_lock = threading.Lock()
_session = {'cookie': None, 'time': 0., 'requests': None}
COOKIE_MAX_AGE = 2400.        # seconds before a cookie is considered stale
LOGIN_FAIL_BACKOFF = 1800.    # after a failed login, no process tries again for this long
LOGIN_MIN_INTERVAL = 120.     # no two logins (any process) closer than this
CACHE_DIR = os.path.expanduser('~/.cache/ldcheck')
COOKIE_FILE = os.path.join(CACHE_DIR, 'ers_cookie.json')
LOCK_FILE = os.path.join(CACHE_DIR, 'ers_login.lock')


class ERSLoginBlocked(RuntimeError):
    pass


def _read_cookie_file():
    try:
        with open(COOKIE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_cookie_file(d):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = COOKIE_FILE + f'.{os.getpid()}'
    with open(tmp, 'w') as fh:
        json.dump(d, fh)
    os.chmod(tmp, 0o600)
    os.replace(tmp, COOKIE_FILE)


def _session_from_cookie(cookie):
    s = requests.Session()
    for part in cookie.split('; '):
        if '=' in part:
            k, v = part.split('=', 1)
            s.cookies.set(k, v, domain='.usgs.gov')
    return s


def ers_login(force=False, stale_cookie=None):
    """
    Return a requests.Session and the cookie string for GDAL.

    The login cookie is shared by all processes through a cache file
    (~/.cache/ldcheck/ers_cookie.json, mode 600) and a lock file, so that a batch
    of workers logs in once, not once per process.  A login happens only when
    the cached cookie is missing or older than COOKIE_MAX_AGE, or when force=True
    and the cached cookie is still the one that failed (stale_cookie).  Logins are
    at least LOGIN_MIN_INTERVAL apart, and after a failed login nobody tries again
    for LOGIN_FAIL_BACKOFF seconds (ERSLoginBlocked is raised instead): repeated
    failed logins can lock the USGS account.
    """
    import fcntl
    from bs4 import BeautifulSoup
    os.makedirs(CACHE_DIR, exist_ok=True)
    with _session_lock, open(LOCK_FILE, 'a') as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            d = _read_cookie_file()
            now = time.time()
            fresh = d.get('cookie') and now - d.get('time', 0) < COOKIE_MAX_AGE
            if fresh and not (force and d['cookie'] == stale_cookie):
                if _session['cookie'] != d['cookie']:
                    _session.update(cookie=d['cookie'], time=d['time'], requests=_session_from_cookie(d['cookie']))
                return _session['requests'], _session['cookie']
            if now - d.get('fail_time', 0) < LOGIN_FAIL_BACKOFF:
                raise ERSLoginBlocked(f"ERS login failed at {time.ctime(d['fail_time'])}; not retrying before "
                                      f"{time.ctime(d['fail_time'] + LOGIN_FAIL_BACKOFF)}")
            if now - d.get('login_time', 0) < LOGIN_MIN_INTERVAL:
                time.sleep(LOGIN_MIN_INTERVAL - (now - d.get('login_time', 0)))
            user, _, password = netrc.netrc().authenticators('ers.cr.usgs.gov')
            s = requests.Session()
            d['login_time'] = time.time()
            try:
                r = s.get(ERS_LOGIN, timeout=60)
                csrf = BeautifulSoup(r.content, 'html.parser').find('input', attrs={'name': 'csrf'})['value']
                r = s.post(ERS_LOGIN, data={'username': user, 'password': password, 'csrf': csrf}, timeout=60)
                cookie = '; '.join(f'{c.name}={c.value}' for c in s.cookies if c.domain.endswith('usgs.gov'))
                if 'EROS_SSO' not in cookie:
                    raise RuntimeError(f'USGS ERS login failed (HTTP {r.status_code}, no EROS_SSO cookie)')
            except Exception:
                d['fail_time'] = time.time()
                _write_cookie_file(d)
                raise
            d.update(cookie=cookie, time=time.time(), fail_time=0)
            _write_cookie_file(d)
            _session.update(cookie=cookie, time=d['time'], requests=s)
            return s, cookie
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _gdal_http_config(cookie):
    return {'GDAL_HTTP_COOKIE': cookie,
            'CPL_VSIL_CURL_USE_HEAD': 'NO',
            'GDAL_NUM_THREADS': '8',          # parallel range requests for COG tiles (4-5x faster)
            'GDAL_INGESTED_BYTES_AT_OPEN': '65536',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.TIF,.tif',
            'GDAL_HTTP_MAX_RETRY': '4',
            'GDAL_HTTP_RETRY_DELAY': '5',
            'VSI_CACHE': 'TRUE',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'}


def decimal_year(dt):
    """datetime -> decimal year"""
    y0 = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    y1 = datetime.datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
    return dt.year + (dt - y0).total_seconds() / (y1 - y0).total_seconds()


def search_scenes(lonlat_polygon, t_center, max_dt_days=45, min_sun_elev=8.,
                  max_cloud=95., platforms=('LANDSAT_8', 'LANDSAT_9'), max_items=500):
    """
    Search the USGS STAC server for Landsat 8/9 L1 scenes.

    Parameters
    ----------
    lonlat_polygon : list of (lon, lat) tuples (closed or not)
    t_center : datetime.datetime (timezone-aware, UTC)
    max_dt_days : half width of the search window
    min_sun_elev : minimum scene-center sun elevation, degrees
    max_cloud : maximum scene cloud cover, percent (USGS's estimate, which is poor over ice)

    Returns
    -------
    list of dicts, one per scene
    """
    from pystac_client import Client
    coords = [list(map(float, p)) for p in lonlat_polygon]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    t0 = t_center - datetime.timedelta(days=max_dt_days)
    t1 = t_center + datetime.timedelta(days=max_dt_days)
    for attempt in range(4):
        try:
            api = Client.open(STAC_URL)
            search = api.search(collections=['landsat-c2l1'],
                                intersects={'type': 'Polygon', 'coordinates': [coords]},
                                datetime=f'{t0.strftime("%Y-%m-%dT%H:%M:%SZ")}/{t1.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                                query={'platform': {'in': list(platforms)}},
                                max_items=max_items)
            items = list(search.items())
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    out = []
    for it in items:
        p = it.properties
        if p.get('view:sun_elevation', 0) < min_sun_elev:
            continue
        if p.get('eo:cloud_cover', 0) > max_cloud:
            continue
        if 'pan' not in it.assets:
            continue
        dt = it.datetime
        out.append(dict(
            id=it.id,
            platform=p.get('platform'),
            datetime=dt.isoformat(),
            t_year=decimal_year(dt),
            dt_days=(dt - t_center).total_seconds() / 86400.,
            path=int(p['landsat:wrs_path']), row=int(p['landsat:wrs_row']),
            sun_elev=float(p['view:sun_elevation']), sun_az=float(p['view:sun_azimuth']),
            cloud_cover=float(p.get('eo:cloud_cover', np.nan)),
            correction=p.get('landsat:correction'),
            geometry=it.geometry,
            pan=it.assets['pan'].href,
            SAA=it.assets['SAA'].href if 'SAA' in it.assets else None,
            SZA=it.assets['SZA'].href if 'SZA' in it.assets else None,
            qa_pixel=it.assets['qa_pixel'].href if 'qa_pixel' in it.assets else None,
            green=it.assets['green'].href if 'green' in it.assets else None,
            swir16=it.assets['swir16'].href if 'swir16' in it.assets else None,
            cirrus=it.assets['cirrus'].href if 'cirrus' in it.assets else None,
            mtl=it.assets['MTL.json'].href if 'MTL.json' in it.assets else None))
    return out


def group_acquisitions(scenes):
    """Group scenes by (platform, path, date): adjacent rows of one pass form one acquisition."""
    groups = {}
    for s in scenes:
        key = (s['platform'], s['path'], s['datetime'][:10])
        groups.setdefault(key, []).append(s)
    out = []
    for key, ss in groups.items():
        ss = sorted(ss, key=lambda s: s['row'])
        out.append(dict(key=f"{key[0].replace('ANDSAT_', '')}_{key[2]}_p{key[1]:03d}",
                        platform=key[0], path=key[1], date=key[2], scenes=ss,
                        t_year=float(np.mean([s['t_year'] for s in ss])),
                        dt_days=float(np.mean([s['dt_days'] for s in ss])),
                        sun_elev=float(np.mean([s['sun_elev'] for s in ss])),
                        cloud_cover=float(np.mean([s['cloud_cover'] for s in ss]))))
    return out


def read_mtl(href, retries=3):
    """Read the MTL.json for a scene; returns the parts needed for calibration."""
    stale = None
    d = None
    for attempt in range(retries):
        s, cookie = ers_login(force=stale is not None, stale_cookie=stale)
        try:
            r = s.get(href, timeout=120)
        except Exception:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200 and r.headers.get('content-type', '').startswith('application/json'):
            d = r.json()['LANDSAT_METADATA_FILE']
            break
        if 'ers.cr.usgs.gov' in r.url:     # redirected to the login page: the cookie expired
            stale = cookie
        else:
            time.sleep(10 * (attempt + 1))
    if d is None:
        raise RuntimeError(f'could not read {href}')
    rescale = d['LEVEL1_RADIOMETRIC_RESCALING']
    img = d['IMAGE_ATTRIBUTES']
    bands = {}
    for b in ('3', '6', '8', '9'):
        if f'REFLECTANCE_MULT_BAND_{b}' in rescale:
            bands[b] = [float(rescale[f'REFLECTANCE_MULT_BAND_{b}']), float(rescale[f'REFLECTANCE_ADD_BAND_{b}'])]
    return dict(refl_mult=float(rescale['REFLECTANCE_MULT_BAND_8']),
                refl_add=float(rescale['REFLECTANCE_ADD_BAND_8']),
                band_refl=bands,
                sun_elev=float(img['SUN_ELEVATION']), sun_az=float(img['SUN_AZIMUTH']),
                date=img['DATE_ACQUIRED'], time=img['SCENE_CENTER_TIME'],
                saturation_b8=img.get('SATURATION_BAND_8', 'N'),
                geometric_rmse=img.get('GEOMETRIC_RMSE_MODEL'),
                product_id=d['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID'])


def _cookie_ok(s, href):
    """True if a 1-byte range request with this session returns data (not the login page)."""
    try:
        r = s.get(href, headers={'Range': 'bytes=0-0'}, timeout=60, allow_redirects=True)
        return r.status_code in (200, 206) and 'ers.cr.usgs.gov' not in r.url
    except Exception:
        return True     # network trouble, not an auth problem


def _block_runs(bounds, res, footprint, block=512):
    """Runs of output blocks (rows r0:r1, cols c0:c1) that intersect the footprint polygon."""
    import shapely.geometry
    nx = int(round((bounds[2] - bounds[0]) / res))
    ny = int(round((bounds[3] - bounds[1]) / res))
    runs = []
    for r0 in range(0, ny, block):
        r1 = min(r0 + block, ny)
        cur = None
        for c0 in range(0, nx, block):
            c1 = min(c0 + block, nx)
            bb = shapely.geometry.box(bounds[0] + c0 * res, bounds[3] - r1 * res, bounds[0] + c1 * res, bounds[3] - r0 * res)
            if footprint is None or footprint.intersects(bb):
                if cur is None:
                    cur = [r0, r1, c0, c1]
                else:
                    cur[3] = c1
            elif cur is not None:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
    return nx, ny, runs


def read_window(href, bounds, out_file, res, resample='nearest', retries=3, footprint=None):
    """
    Windowed read of a Landsat COG onto a grid aligned with the 15-m pan grid.

    bounds : [xmin, ymin, xmax, ymax] of the output pixel EDGES, in EPSG:3031
    res : output resolution
    footprint : optional shapely polygon (EPSG:3031); only the 512-pixel blocks of the
        output that intersect it are read (the rest is zero = no data).  For strips that
        run diagonally across their bounding box this cuts the download severalfold.
    """
    tmp = out_file + '.part'
    err = None
    stale = None
    nx, ny, runs = _block_runs(bounds, res, footprint)
    for attempt in range(retries):
        s, cookie = ers_login(force=stale is not None, stale_cookie=stale)
        try:
            with gdal.config_options(_gdal_http_config(cookie)):
                src = gdal.Open('/vsicurl/' + href)
                dtype = src.GetRasterBand(1).DataType
                out = None
                for r0, r1, c0, c1 in runs:
                    sub = [bounds[0] + c0 * res, bounds[3] - r1 * res, bounds[0] + c1 * res, bounds[3] - r0 * res]
                    ds = gdal.Warp('', src, options=gdal.WarpOptions(
                        format='MEM', outputBounds=sub, xRes=res, yRes=res, dstSRS='EPSG:3031',
                        resampleAlg=resample, srcNodata=0, dstNodata=0))
                    a = ds.GetRasterBand(1).ReadAsArray()
                    if out is None:
                        out = np.zeros((ny, nx), dtype=a.dtype)
                    out[r0:r1, c0:c1] = a
                    ds = None
                src = None
            if out is None:
                out = np.zeros((ny, nx), dtype=np.uint16)
            drv = gdal.GetDriverByName('GTiff')
            dst = drv.Create(tmp, nx, ny, 1, dtype, options=['COMPRESS=DEFLATE', 'TILED=YES', 'PREDICTOR=2'])
            dst.SetGeoTransform((bounds[0], res, 0., bounds[3], 0., -res))
            dst.SetProjection(osr.SpatialReference(wkt=_EPSG3031_WKT()).ExportToWkt())
            dst.GetRasterBand(1).WriteArray(out)
            dst.GetRasterBand(1).SetNoDataValue(0)
            dst = None
            os.rename(tmp, out_file)
            return out_file
        except Exception as e:
            err = e
            if os.path.isfile(tmp):
                os.remove(tmp)
            if not _cookie_ok(s, href):
                stale = cookie
            else:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'could not read {href}: {err}')


def _EPSG3031_WKT():
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3031)
    return sr.ExportToWkt()


def scene_overlaps(scene_geom, bounds):
    """True if a STAC footprint (lon/lat) overlaps the EPSG:3031 box."""
    import shapely.geometry
    import pyproj
    xf = pyproj.Transformer.from_crs(4326, 3031, always_xy=True)
    poly = shapely.geometry.shape(scene_geom)
    xy = np.array(poly.exterior.coords)
    x, y = xf.transform(xy[:, 0], xy[:, 1])
    p3031 = shapely.geometry.Polygon(np.c_[x, y])
    box = shapely.geometry.box(*bounds)
    return p3031.intersection(box).area / box.area


def _covers(fname, bounds, tol=1.):
    """True if the GeoTIFF fname covers bounds (pixel edges)."""
    try:
        ds = gdal.Open(fname)
        gt = ds.GetGeoTransform()
        x0, x1 = gt[0], gt[0] + gt[1] * ds.RasterXSize
        y1, y0 = gt[3], gt[3] + gt[5] * ds.RasterYSize
        return x0 <= bounds[0] + tol and y0 <= bounds[1] + tol and x1 >= bounds[2] - tol and y1 >= bounds[3] - tol
    except Exception:
        return False


class download_slot:
    """
    A cross-process semaphore (MAX_DOWNLOADS lock files) that limits the number of
    simultaneous downloads from landsatlook by all the workers of a batch.
    """
    MAX_DOWNLOADS = 5

    def __enter__(self):
        import fcntl
        os.makedirs(CACHE_DIR, exist_ok=True)
        while True:
            for i in range(self.MAX_DOWNLOADS):
                fh = open(os.path.join(CACHE_DIR, f'download_slot_{i}.lock'), 'a')
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fh = fh
                    return self
                except OSError:
                    fh.close()
            time.sleep(1.)

    def __exit__(self, *args):
        import fcntl
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def get_acquisition_chips(acq, bounds, chip_dir, verbose=False, footprint=None):
    """
    Download (windowed) the pan band and the solar angle bands for every scene of
    an acquisition, over bounds (pixel edges, EPSG:3031, multiples of 15 m +- 7.5).

    Returns a list of dicts with the local file names and calibration for each scene.
    """
    os.makedirs(chip_dir, exist_ok=True)
    out = []
    for s in acq['scenes']:
        if scene_overlaps(s['geometry'], bounds) < 0.002:
            continue
        base = os.path.join(chip_dir, s['id'])
        files = {}
        meta_file = base + '_meta.json'
        meta = None
        if os.path.isfile(meta_file):
            with open(meta_file) as fh:
                meta = json.load(fh)
            if 'band_refl' not in meta:
                meta = None
        if meta is None:
            with download_slot():
                meta = read_mtl(s['mtl'])
            meta['scene'] = {k: v for k, v in s.items() if k != 'geometry'}
            with open(meta_file, 'w') as fh:
                json.dump(meta, fh, indent=1)
        # pan at 15 m, the cloud-test bands at 60 m (read from the COG overviews)
        # (the sun angles are computed from the scene time; see sun.py)
        # (the visible band of the snow index is the pan band itself)
        for band, res, resample in [('pan', 15, 'nearest'), ('swir16', 60, 'average'), ('cirrus', 60, 'average')]:
            if s.get(band) is None:
                continue
            if res == 15:
                f, b = base + f'_{band}.tif', bounds
            else:
                f = base + f'_{band}{res}.tif'
                b = [np.floor(bounds[0] / res) * res, np.floor(bounds[1] / res) * res,
                     np.ceil(bounds[2] / res) * res, np.ceil(bounds[3] / res) * res]
            if os.path.isfile(f) and not _covers(f, bounds):
                os.remove(f)
            if not os.path.isfile(f):
                t0 = time.time()
                with download_slot():
                    read_window(s[band], b, f, res, resample=resample, footprint=footprint)
                if verbose:
                    print(f'\t{os.path.basename(f)}: {time.time()-t0:.1f} s', flush=True)
            files[band] = f
        out.append(dict(id=s['id'], files=files, meta=meta))
    return out
