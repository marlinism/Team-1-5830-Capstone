import pandas as pd

from parking_processing.common.constants import TZ_LOCAL

def parse_point_wkt(series: pd.Series):
    """
    Parsing exported location: 'POINT (-122.33 47.61)'

    Returns (lat, lon) float series.
    """
    s = series.astype(str)
    m = s.str.extract(r"POINT\s*\(\s*([-0-9.]+)\s+([-0-9.]+)\s*\)")
    lon = pd.to_numeric(m[0], errors="coerce")
    lat = pd.to_numeric(m[1], errors="coerce")
    return lat, lon

def to_utc(ts: pd.Series, tz_local: str = TZ_LOCAL) -> pd.Series:
    """
    Convert timestamp to UTC
    """
    t = pd.to_datetime(ts, errors="coerce")

    if getattr(t.dt, "tz", None) is not None:
        return t.dt.tz_convert("UTC")

    return t.dt.tz_localize(tz_local, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

def floor_bin(ts_utc: pd.Series, freq: str) -> pd.Series:
    return ts_utc.dt.floor(freq)

def week_start_local(ts_utc: pd.Series, tz_local: str = TZ_LOCAL) -> pd.Series:
    """
    Convert Monday week start to form of 'YYYY-MM-DD'
    """
    tloc = ts_utc.dt.tz_convert(tz_local)

    wk = (tloc - pd.to_timedelta(tloc.dt.weekday, unit="D")).dt.normalize()
    return wk.dt.strftime("%Y-%m-%d")