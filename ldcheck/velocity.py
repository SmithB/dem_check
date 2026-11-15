"""
Velocity fields for advecting Landsat images to the DEM time.

Two sources:
  'itslive' : ITS_LIVE v2 composite vx/vy GeoTIFFs (m/yr), EPSG:3031 (default:
              the 120-m Antarctic composite on /Volumes/ice1)
  'stack'   : an h5 file with x, y, time, U, V (m/yr), e.g. the SPK quarterly
              velocity_stacks; the time slice nearest the DEM time is used, and
              gaps are filled from the ITS_LIVE composite.
"""
import numpy as np
import h5py
from osgeo import gdal
import scipy.ndimage as snd
from scipy.interpolate import RegularGridInterpolator

gdal.UseExceptions()

ITSLIVE_VX = '/Volumes/ice1/ben/ITS_LIVE/AA_v02.1/ITS_LIVE_velocity_120m_RGI19A_0000_V02.1_vx.tif'
ITSLIVE_VY = '/Volumes/ice1/ben/ITS_LIVE/AA_v02.1/ITS_LIVE_velocity_120m_RGI19A_0000_V02.1_vy.tif'


def _read_tif_window(fname, bounds, res=120.):
    ds = gdal.Warp('', fname, options=gdal.WarpOptions(
        format='MEM', outputBounds=bounds, xRes=res, yRes=res, dstSRS='EPSG:3031',
        resampleAlg='bilinear', dstNodata=np.nan, outputType=gdal.GDT_Float32))
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    x = gt[0] + gt[1] * (np.arange(ds.RasterXSize) + 0.5)
    y = gt[3] + gt[5] * (np.arange(ds.RasterYSize) + 0.5)
    a[np.abs(a) > 3.e4] = np.nan
    return x, y, a


def _fill_nan(a, max_iter=50):
    """Fill NaNs by repeated 3x3 averaging of the valid neighbours (then zero)."""
    a = a.copy()
    for _ in range(max_iter):
        bad = ~np.isfinite(a)
        if not bad.any():
            break
        w = snd.uniform_filter((~bad).astype(float), 3, mode='nearest')
        s = snd.uniform_filter(np.where(bad, 0, a), 3, mode='nearest')
        fill = bad & (w > 0)
        a[fill] = s[fill] / w[fill]
    a[~np.isfinite(a)] = 0.
    return a


class VelocityField:
    def __init__(self, bounds, source='itslive', stack_file=None, t=None,
                 vx_file=ITSLIVE_VX, vy_file=ITSLIVE_VY, pad=5000.):
        b = [bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad]
        x, y, vx = _read_tif_window(vx_file, b)
        _, _, vy = _read_tif_window(vy_file, b)
        self.source = 'itslive'
        if source == 'stack' and stack_file is not None:
            with h5py.File(stack_file, 'r') as h:
                xs, ys, ts = h['x'][:], h['y'][:], h['time'][:]
                it = int(np.argmin(np.abs(ts - t)))
                ix = np.flatnonzero((xs >= b[0] - 500) & (xs <= b[2] + 500))
                iy = np.flatnonzero((ys >= b[1] - 500) & (ys <= b[3] + 500))
                if ix.size > 2 and iy.size > 2:
                    U = h['U'][iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1, it]
                    V = h['V'][iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1, it]
                    self.stack_time = float(ts[it])
                    xx, yy = np.meshgrid(x, y)
                    for comp, A in (('vx', U), ('vy', V)):
                        f = RegularGridInterpolator((ys[iy], xs[ix]), A, bounds_error=False, fill_value=np.nan)
                        vs = f(np.c_[yy.ravel(), xx.ravel()]).reshape(xx.shape)
                        base = vx if comp == 'vx' else vy
                        good = np.isfinite(vs)
                        base[good] = vs[good]
                    self.source = 'stack'
        vx = _fill_nan(vx)
        vy = _fill_nan(vy)
        # y must increase for RegularGridInterpolator
        order = np.argsort(y)
        self._fx = RegularGridInterpolator((y[order], x), vx[order], bounds_error=False, fill_value=0.)
        self._fy = RegularGridInterpolator((y[order], x), vy[order], bounds_error=False, fill_value=0.)

    def __call__(self, x, y):
        pts = np.c_[np.ravel(y), np.ravel(x)]
        return self._fx(pts).reshape(np.shape(x)), self._fy(pts).reshape(np.shape(x))
