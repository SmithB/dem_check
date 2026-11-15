"""
Find and read REMA 2-m strip DEMs.

The working grid has 3-m pixels whose edges coincide with the edges of the
Landsat 15-m pan pixels (pan pixel centers are on multiples of 15 m), so that a
5x5 block mean of a 3-m field is exactly the 15-m pixel average.
"""
import os
import re
import glob
import json
import datetime
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

DEFAULT_2M_DIRS = sorted(glob.glob('/Volumes/ice1/ben/REMA/fullres/*/2m')) + \
    ['/Volumes/ice1/ben/REMA/landsat_DEM_check/dems_2m']
DEFAULT_META_DIRS = sorted(glob.glob('/Volumes/ice1/ben/REMA/fullres/*/meta'))

PGC_RE = re.compile(r'(SETSM_s2s\d+_(\w\w\d\d)_(\d{8})_([0-9A-F]+)_([0-9A-F]+)_(\d+)m_(lsf_)?seg(\d+))')
URL_BASE = 'https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/rema/strips'
L15 = 15.
W3 = 3.


def strip_base(name):
    """'SETSM_..._16m_lsf_seg3_dem_filt.tif' -> 'SETSM_..._2m_lsf_seg3' """
    m = PGC_RE.search(os.path.basename(name))
    if m is None:
        raise ValueError(f'not a PGC strip name: {name}')
    return re.sub(r'_\d+m_(lsf_)?seg', lambda mm: '_2m_' + (mm.group(1) or '') + 'seg', m.group(1))


def _resolve(path):
    """Return a readable path for path, fixing symlinks made relative to the wrong directory."""
    if os.path.isfile(path):
        return path
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.isabs(target):
            for base in [os.path.dirname(os.path.dirname(path)), os.path.dirname(path)]:
                cand = os.path.normpath(os.path.join(base, target))
                if os.path.isfile(cand):
                    return cand
    return None


def find_2m_files(name, dirs=None):
    """Find the 2-m dem, matchtag and bitmask files for a strip (any resolution name)."""
    base = strip_base(name)
    if dirs is None:
        dirs = DEFAULT_2M_DIRS
    if os.path.isfile(name):
        dirs = [os.path.dirname(os.path.abspath(name))] + list(dirs)
    for d in dirs:
        out = {}
        for ext in ['dem', 'matchtag', 'bitmask']:
            f = _resolve(os.path.join(d, f'{base}_{ext}.tif'))
            if f is not None:
                out[ext] = f
        if 'dem' in out:
            return out
    return None


def strip_url(name, geocell=None):
    """The PGC S3 json url for a strip, from all_REMA_urls.txt."""
    base = strip_base(name)
    url_file = '/Volumes/ice1/ben/REMA/index/all_REMA_urls.txt'
    with open(url_file) as fh:
        for line in fh:
            line = line.strip()
            if line.endswith(base):
                return line + '.json' if not line.endswith('.json') else line
    return None


def download_2m(name, dst_dir='/Volumes/ice1/ben/REMA/landsat_DEM_check/dems_2m'):
    """Download the 2-m dem, matchtag and bitmask for a strip from PGC (public S3)."""
    import requests
    base = strip_base(name)
    url = strip_url(name)
    if url is None:
        raise IndexError(f'no url for {base}')
    url = url.replace('.json', '')
    os.makedirs(dst_dir, exist_ok=True)
    out = {}
    for ext in ['dem', 'matchtag', 'bitmask']:
        f = os.path.join(dst_dir, f'{base}_{ext}.tif')
        if not os.path.isfile(f):
            with requests.get(f'{url}_{ext}.tif', stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(f + '.part', 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        fh.write(chunk)
            os.rename(f + '.part', f)
        out[ext] = f
    return out


def strip_time(name, meta_dirs=None):
    """Acquisition time (UTC datetime) of a strip: the PGC json 'datetime' if found, else noon of the file date."""
    base = strip_base(name)
    if meta_dirs is None:
        meta_dirs = DEFAULT_META_DIRS
    for d in meta_dirs:
        f = os.path.join(d, base + '.json')
        if os.path.isfile(f):
            try:
                with open(f) as fh:
                    meta = json.load(fh)
                if meta.get('id', '') == base or meta.get('id', '').startswith(base.rsplit('_seg', 1)[0]):
                    t = meta['properties']['datetime']
                    return datetime.datetime.fromisoformat(t.replace('Z', '+00:00'))
            except Exception:
                pass
    date = PGC_RE.search(base).group(3)
    return datetime.datetime.strptime(date + '12', '%Y%m%d%H').replace(tzinfo=datetime.timezone.utc)


def dem_footprint(dem_file, res=64.):
    """
    Bounds (EPSG:3031) of the valid part of a DEM, and a simplified footprint
    polygon (shapely, EPSG:3031) from a decimated read.
    """
    import shapely.geometry
    import shapely.ops
    ds = gdal.Open(dem_file)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    dec = max(1, int(res / gt[1]))
    bx, by = max(1, nx // dec), max(1, ny // dec)
    band = ds.GetRasterBand(1)
    z = band.ReadAsArray(buf_xsize=bx, buf_ysize=by, resample_alg=gdal.GRIORA_NearestNeighbour)
    nd = band.GetNoDataValue()
    valid = np.isfinite(z)
    if nd is not None:
        valid &= z != nd
    valid &= z > -500
    dx = gt[1] * nx / bx
    dy = gt[5] * ny / by
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return None, None
    x = gt[0] + (cols + 0.5) * dx
    y = gt[3] + (rows + 0.5) * dy
    bounds = [x.min() - abs(dx), y.min() - abs(dy), x.max() + abs(dx), y.max() + abs(dy)]
    pts = shapely.geometry.MultiPoint(np.c_[x, y])
    poly = pts.convex_hull.buffer(2 * abs(dx)).simplify(abs(dx))
    return bounds, poly


def snap_bounds(bounds, pad=0.):
    """Bounds snapped outward to 15-m pan pixel EDGES (multiples of 15 m +/- 7.5 m)."""
    x0 = np.floor((bounds[0] - pad - 7.5) / L15) * L15 + 7.5
    y0 = np.floor((bounds[1] - pad - 7.5) / L15) * L15 + 7.5
    x1 = np.ceil((bounds[2] + pad - 7.5) / L15) * L15 + 7.5
    y1 = np.ceil((bounds[3] + pad - 7.5) / L15) * L15 + 7.5
    return [float(x0), float(y0), float(x1), float(y1)]


def read_dem_3m(files, bounds, out_dir=None):
    """
    Read the DEM onto the 3-m working grid over bounds (15-m pixel edges).

    Returns dict with z (float32, NaN for no data), matched (fraction of matched
    2-m pixels), bad_bitmask (fraction of 2-m pixels with bitmask edge/cloud bits),
    and the grid (x, y pixel centers, increasing x and decreasing y).
    """
    nx = int(round((bounds[2] - bounds[0]) / W3))
    ny = int(round((bounds[3] - bounds[1]) / W3))
    out = {}
    opts = dict(format='MEM', outputBounds=bounds, width=nx, height=ny, dstSRS='EPSG:3031',
                multithread=True, warpOptions=['NUM_THREADS=4'])
    ds = gdal.Warp('', files['dem'], options=gdal.WarpOptions(resampleAlg='average', srcNodata=-9999,
                                                               dstNodata=np.nan, outputType=gdal.GDT_Float32, **opts))
    z = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    z[z < -500] = np.nan
    out['z'] = z
    ds = None
    if 'matchtag' in files:
        ds = gdal.Warp('', files['matchtag'], options=gdal.WarpOptions(resampleAlg='average', srcNodata=255,
                                                                        dstNodata=-1, outputType=gdal.GDT_Float32, **opts))
        m = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        m[m < 0] = 0
        out['matched'] = m
        ds = None
    if 'bitmask' in files:
        # bitmask: 0 = good, bit 1 = edge, bit 2 = water, bit 4 = cloud; dem_filter keeps 0 and 2
        src = gdal.Open(files['bitmask'])
        gt = src.GetGeoTransform()
        # read only the window under the tile (with a 2-pixel margin)
        c0 = max(int(np.floor((bounds[0] - gt[0]) / gt[1])) - 2, 0)
        c1 = min(int(np.ceil((bounds[2] - gt[0]) / gt[1])) + 2, src.RasterXSize)
        r0 = max(int(np.floor((bounds[3] - gt[3]) / gt[5])) - 2, 0)
        r1 = min(int(np.ceil((bounds[1] - gt[3]) / gt[5])) + 2, src.RasterYSize)
        if c1 > c0 and r1 > r0:
            b = src.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
            bad = ((b != 0) & (b != 2)).astype(np.uint8)
            drv = gdal.GetDriverByName('MEM')
            mem = drv.Create('', c1 - c0, r1 - r0, 1, gdal.GDT_Byte)
            mem.SetGeoTransform((gt[0] + c0 * gt[1], gt[1], 0., gt[3] + r0 * gt[5], 0., gt[5]))
            mem.SetProjection(src.GetProjection())
            mem.GetRasterBand(1).WriteArray(bad)
            ds = gdal.Warp('', mem, options=gdal.WarpOptions(resampleAlg='average', outputType=gdal.GDT_Float32, **opts))
            out['bad_bitmask'] = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            ds = None
            mem = None
        src = None
    out['x'] = bounds[0] + W3 * (np.arange(nx) + 0.5)
    out['y'] = bounds[3] - W3 * (np.arange(ny) + 0.5)
    out['bounds'] = bounds
    return out
