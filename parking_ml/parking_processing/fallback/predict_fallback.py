from __future__ import annotations

import io
import os
import gzip
import tempfile
from dataclasses import dataclass
from typing import Literal, List, Set, Tuple

import boto3
import numpy as np
import pandas as pd

from parking_processing.utils.s3 import parse_s3_uri

Target = Literal["y_15", "y_30"]


@dataclass
class FallbackCfg:
    s3_root: str
    year: int
    target: Target

    # s3://.../profile.csv.gz returned by run_build_sat_profile
    sat_profile_s3: str

    # relative to prefix (NOT full s3://) unless you pass an s3://...
    free_days_key: str = "meta/free_days.csv"

    tz_local: str = "America/Los_Angeles"
    model_name: str = "fallback"
    uncertainty_low: float = 0.05
    uncertainty_high: float = 0.10
    overwrite: bool = False
    tmp_dir: str = "/tmp/fallback_preds"


# ---------- S3 helpers ----------

def _parse_s3_object_uri(uri: str) -> Tuple[str, str]:
    """
    Parse an S3 OBJECT uri without forcing a trailing slash.
    Example:
      s3://bucket/a/b/file.csv.gz  -> (bucket, "a/b/file.csv.gz")
      s3://bucket/a/b/file.csv.gz/ -> (bucket, "a/b/file.csv.gz")  (strip trailing '/')
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"Expected s3://... uri, got: {uri!r}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) == 2 else ""
    key = key.lstrip("/").rstrip("/")  # IMPORTANT
    return bucket, key


def _read_csv_gz_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    key = key.rstrip("/")  # defensive: never request ".../file.csv.gz/"
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as f:
        return pd.read_csv(f)


def _read_csv_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    key = key.rstrip("/")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    return pd.read_csv(io.BytesIO(raw))


def _preds_dir(prefix: str, year: int, target: str, model_name: str) -> str:
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}preds/{model_name}/year={year}/target={target}/"


def _pred_key(prefix: str, year: int, week: str, target: str, model_name: str) -> str:
    return f"{_preds_dir(prefix, year, target, model_name)}week={week}/pred.csv.gz"


def _success_key(prefix: str, year: int, week: str, target: str, model_name: str) -> str:
    return f"{_preds_dir(prefix, year, target, model_name)}week={week}/_SUCCESS"


def _exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ---------- domain logic ----------

def _load_free_days(s3, bucket: str, prefix: str, free_days_key: str) -> Set[pd.Timestamp]:
    """
    free_days_key can be:
      - "meta/free_days.csv" (relative to prefix)
      - "s3://bucket/path/to/free_days.csv"

    Accepts columns:
      - date_local (preferred)
      - observed_date / date / day (fallback)

    If 'is_free' exists, we keep only rows where is_free is truthy.
    Returns a set of naive day timestamps (date-only).
    """
    if free_days_key.startswith("s3://"):
        b2, k2 = _parse_s3_object_uri(free_days_key)
        df = _read_csv_s3(s3, b2, k2)
    else:
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        df = _read_csv_s3(s3, bucket, f"{prefix}{free_days_key}")

    # normalize colnames
    df.columns = [c.lower() for c in df.columns]

    # filter only free days if column exists
    if "is_free" in df.columns:
        # accept True/False, 1/0, "true"/"false", "yes"/"no"
        s = df["is_free"]
        if s.dtype == bool:
            mask = s
        else:
            mask = s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        df = df[mask].copy()

    # pick date column
    if "date_local" in df.columns:
        col = "date_local"
    elif "observed_date" in df.columns:
        col = "observed_date"
    elif "date" in df.columns:
        col = "date"
    elif "day" in df.columns:
        col = "day"
    else:
        raise KeyError(f"free_days.csv must include a date column. Have: {list(df.columns)}")

    s = pd.to_datetime(df[col], errors="coerce")
    s = s.dropna()

    # store as date-only timestamps
    return set(pd.Timestamp(d.date()) for d in s)


def _week_grid_utc(week_start: str, tz_local: str) -> pd.DatetimeIndex:
    """
    Full 7-day grid of 15-min timestamps aligned to LOCAL midnight, returned in UTC.
    7 days * 96 = 672
    """
    start_local = pd.Timestamp(f"{week_start} 00:00:00", tz=tz_local)
    start_utc = start_local.tz_convert("UTC")
    return pd.date_range(start=start_utc, periods=7 * 96, freq="15min", tz="UTC")


def _load_sat_profile(cfg: FallbackCfg) -> pd.DataFrame:
    s3 = boto3.client("s3")

    # IMPORTANT: sat_profile_s3 is an OBJECT uri, do NOT use parse_s3_uri()
    b, k = _parse_s3_object_uri(cfg.sat_profile_s3)

    df = _read_csv_gz_s3(s3, b, k)

    need = {"sourceelementkey", "slot", "p_sat_mean"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"sat profile missing columns {missing}. Have: {list(df.columns)}")

    df["sourceelementkey"] = df["sourceelementkey"].astype(int)
    df["slot"] = df["slot"].astype(int)
    df["p_sat_mean"] = pd.to_numeric(df["p_sat_mean"], errors="coerce").fillna(0.5)
    return df


def _make_fallback_df_for_week(
    *,
    week: str,
    tz_local: str,
    free_days: Set[pd.Timestamp],
    sat_profile: pd.DataFrame,
    target: str,
    uncertainty_low: float,
    uncertainty_high: float,
) -> pd.DataFrame:
    ts_utc = _week_grid_utc(week, tz_local)
    ts_local = ts_utc.tz_convert(tz_local)
    local_dates = ts_local.date

    free_set = set(d.date() for d in free_days)

    # Need fallback on Sundays OR observed free days
    need_fb = np.array([(d.weekday() == 6) or (d in free_set) for d in local_dates], dtype=bool)

    out_col = "p15" if target == "y_15" else "p30"
    if not need_fb.any():
        return pd.DataFrame(columns=["ts15_utc", "sourceelementkey", out_col])

    ts_fb = ts_utc[need_fb]
    ts_fb_local = ts_fb.tz_convert(tz_local)
    slot = (ts_fb_local.hour * 4 + (ts_fb_local.minute // 15)).astype(int)

    base = pd.DataFrame({"ts15_utc": ts_fb, "slot": slot})
    df = base.merge(sat_profile, on="slot", how="left")

    # add uncertainty
    rng = np.random.default_rng(abs(hash((week, target))) % (2**32))
    u_hi = rng.uniform(1.0 - uncertainty_high, 1.0 + uncertainty_high, size=len(df))
    if uncertainty_low != uncertainty_high:
        u_lo = rng.uniform(1.0 - uncertainty_low, 1.0 + uncertainty_low, size=len(df))
        mix = rng.uniform(0, 1, size=len(df))
        u = mix * u_hi + (1 - mix) * u_lo
    else:
        u = u_hi

    p = (df["p_sat_mean"].to_numpy(dtype=float) * u).clip(0.0, 1.0)

    out = df[["ts15_utc", "sourceelementkey"]].copy()
    out[out_col] = p
    out["ts15_utc"] = pd.to_datetime(out["ts15_utc"], utc=True)
    out["sourceelementkey"] = out["sourceelementkey"].astype(int)

    out = out.sort_values(["ts15_utc", "sourceelementkey"]).reset_index(drop=True)
    return out


def run_fallback_for_weeks(cfg: FallbackCfg, weeks: List[str]) -> None:
    """
    Writes fallback predictions ONLY for Sundays + observed free parking dates.
    Output path:
      s3://<bucket>/<prefix>/preds/<model_name>/year=YYYY/target=y_15/week=YYYY-MM-DD/pred.csv.gz
    """
    s3 = boto3.client("s3")

    bucket, prefix = parse_s3_uri(cfg.s3_root)  # OK: cfg.s3_root is a prefix/dir

    sat_profile = _load_sat_profile(cfg)
    free_days = _load_free_days(s3, bucket, prefix, cfg.free_days_key)

    os.makedirs(cfg.tmp_dir, exist_ok=True)

    for week in weeks:
        suc = _success_key(prefix, cfg.year, week, cfg.target, cfg.model_name)

        if (not cfg.overwrite) and _exists(s3, bucket, suc):
            print(f"[skip] week={week} already has fallback _SUCCESS")
            continue

        df = _make_fallback_df_for_week(
            week=week,
            tz_local=cfg.tz_local,
            free_days=free_days,
            sat_profile=sat_profile,
            target=cfg.target,
            uncertainty_low=cfg.uncertainty_low,
            uncertainty_high=cfg.uncertainty_high,
        )

        if df.empty:
            s3.put_object(Bucket=bucket, Key=suc, Body=f"ok empty week={week}\n".encode("utf-8"))
            print(f"[ok] week={week} (no fallback needed)")
            continue

        local_path = os.path.join(cfg.tmp_dir, f"fallback_{cfg.year}_{week}_{cfg.target}.csv.gz")
        with gzip.open(local_path, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)

        key = _pred_key(prefix, cfg.year, week, cfg.target, cfg.model_name)
        s3.upload_file(local_path, bucket, key)
        s3.put_object(Bucket=bucket, Key=suc, Body=f"ok target={cfg.target}\n".encode("utf-8"))

        print(f"[ok] week={week} -> s3://{bucket}/{key}")

        try:
            os.remove(local_path)
        except Exception:
            pass