from __future__ import annotations

import io
import gzip
from dataclasses import dataclass
from typing import Literal, List, Tuple

import boto3
import numpy as np
import pandas as pd

from parking_processing.utils.s3 import parse_s3_uri

Target = Literal["y_15", "y_30"]


@dataclass
class SatProfileCfg:
    s3_root: str                 # e.g. "s3://smart-park-seattle/parking/"
    year_ref: int                # e.g. 2022
    target: Target               # "y_15" or "y_30"
    tz_local: str = "America/Los_Angeles"
    # where to read the base predictions from:
    # io_pred writes: preds/mvstgcn/year=YYYY/target=y_15/week=.../pred.csv.gz
    base_model_name: str = "mvstgcn"
    # where to store the profile
    out_key: str = "meta/sat_profile"  # under prefix


def _s3_read_csv_gz(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as f:
        return pd.read_csv(f)


def _list_week_pred_keys(
    s3, bucket: str, prefix: str, year: int, target: str, base_model_name: str
) -> List[str]:
    # s3 key prefix
    # parking/preds/mvstgcn/year=2022/target=y_15/week=YYYY-MM-DD/pred.csv.gz
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    pfx = f"{prefix}preds/{base_model_name}/year={year}/target={target}/"
    keys: List[str] = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=pfx):
        for it in page.get("Contents", []):
            k = it["Key"]
            if k.endswith("/pred.csv.gz"):
                keys.append(k)

    return sorted(keys)


def run_build_sat_profile(cfg: SatProfileCfg) -> str:
    """
    Build a Saturday profile from existing predictions:
      - filter rows where local day is Saturday
      - compute slot = hour*4 + minute/15  (0..95)
      - average probability per (sourceelementkey, slot)
    Save as CSV.GZ to S3 and return its s3:// uri.
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(cfg.s3_root)

    prob_col = "p15" if cfg.target == "y_15" else "p30"

    pred_keys = _list_week_pred_keys(
        s3, bucket, prefix, cfg.year_ref, cfg.target, cfg.base_model_name
    )
    if not pred_keys:
        raise FileNotFoundError(
            f"No prediction files found under preds/{cfg.base_model_name}/year={cfg.year_ref}/target={cfg.target}/"
        )

    chunks = []
    for k in pred_keys:
        df = _s3_read_csv_gz(s3, bucket, k)
        if df.empty:
            continue
        if "ts15_utc" not in df.columns or "sourceelementkey" not in df.columns:
            continue
        if prob_col not in df.columns:
            # if the file is "both" it may contain p15/p30, else only one
            continue

        ts_local = pd.to_datetime(df["ts15_utc"], utc=True).dt.tz_convert(cfg.tz_local)
        dow = ts_local.dt.dayofweek  # Mon=0 ... Sat=5 ... Sun=6
        df = df.loc[dow == 5, ["ts15_utc", "sourceelementkey", prob_col]].copy()
        if df.empty:
            continue

        ts_local = pd.to_datetime(df["ts15_utc"], utc=True).dt.tz_convert(cfg.tz_local)
        slot = (ts_local.dt.hour * 4 + (ts_local.dt.minute // 15)).astype(int)
        df["slot"] = slot
        df.rename(columns={prob_col: "p"}, inplace=True)
        chunks.append(df[["sourceelementkey", "slot", "p"]])

    if not chunks:
        raise RuntimeError("No Saturday rows found to build profile (check your preds coverage).")

    all_sat = pd.concat(chunks, ignore_index=True)
    prof = (
        all_sat.groupby(["sourceelementkey", "slot"], as_index=False)["p"]
        .mean()
        .rename(columns={"p": "p_sat_mean"})
    )

    # write to S3 as csv.gz
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    out_key = f"{prefix}{cfg.out_key}/year_ref={cfg.year_ref}/target={cfg.target}/profile.csv.gz"

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        prof.to_csv(io.TextIOWrapper(gz, encoding="utf-8", newline=""), index=False)
    buf.seek(0)

    s3.put_object(Bucket=bucket, Key=out_key, Body=buf.getvalue())
    return f"s3://{bucket}/{out_key}"