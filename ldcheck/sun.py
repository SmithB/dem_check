"""Solar position (NOAA spreadsheet algorithm, ~0.01 deg; no refraction)."""
import numpy as np
import datetime


def sun_position(t, lat, lon):
    """
    t : timezone-aware datetime (UTC); lat, lon : degrees (arrays allowed)
    Returns elevation and azimuth (clockwise from north), degrees.
    """
    t = t.astimezone(datetime.timezone.utc)
    jd = (t - datetime.datetime(2000, 1, 1, 12, tzinfo=datetime.timezone.utc)).total_seconds() / 86400. + 2451545.0
    jc = (jd - 2451545.0) / 36525.
    L = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    r = np.radians
    C = np.sin(r(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(r(2 * M)) * (0.019993 - 0.000101 * jc) \
        + np.sin(r(3 * M)) * 0.000289
    app = L + C - 0.00569 - 0.00478 * np.sin(r(125.04 - 1934.136 * jc))
    obl = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obl = obl + 0.00256 * np.cos(r(125.04 - 1934.136 * jc))
    decl = np.degrees(np.arcsin(np.sin(r(obl)) * np.sin(r(app))))
    vy = np.tan(r(obl / 2)) ** 2
    eqt = 4 * np.degrees(vy * np.sin(2 * r(L)) - 2 * e * np.sin(r(M)) + 4 * e * vy * np.sin(r(M)) * np.cos(2 * r(L))
                         - 0.5 * vy * vy * np.sin(4 * r(L)) - 1.25 * e * e * np.sin(2 * r(M)))
    minutes = t.hour * 60 + t.minute + (t.second + t.microsecond / 1e6) / 60.
    tst = np.mod(minutes + eqt + 4 * np.asarray(lon, float), 1440)
    ha = np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180)
    lat = np.asarray(lat, float)
    cz = np.sin(r(lat)) * np.sin(r(decl)) + np.cos(r(lat)) * np.cos(r(decl)) * np.cos(r(ha))
    zen = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    ca = (np.sin(r(lat)) * np.cos(r(zen)) - np.sin(r(decl))) / (np.cos(r(lat)) * np.sin(r(zen)))
    a = np.degrees(np.arccos(np.clip(ca, -1, 1)))
    az = np.where(ha > 0, np.mod(a + 180, 360), np.mod(540 - a, 360))
    return 90. - zen, az
