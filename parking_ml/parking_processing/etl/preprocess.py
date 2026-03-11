import numpy as np
import pandas as pd

from parking_processing.utils.geo import parse_point_wkt, to_utc, floor_bin
from parking_processing.common.constants import TZ_LOCAL, BIN_FREQ

K_EMPTY = 3  # based on pos_rate_calculation.ipynb

REQUIRED_COLUMNS = {"ts", "paid_occupancy", "space_count", "sourceelementkey", "location"}

DF15_COLUMNS = [
    "sourceelementkey", "ts15_utc",
    "paid_occupancy", "space_count", "rate", "lat", "lon",
    "flag_occ_lt0", "flag_occ_gt_spaces",
    "empty_spots", "occ_rate", "is_full", "has_space",
    "area", "subarea", "parking_cat", "blockfacename", "sideofstreet", "time_limit_cat",
]

def empty_df15() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in DF15_COLUMNS})

def validate_required_cols(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required canonical columns: {missing}. Have: {list(df.columns)}")

def parse_and_cast(df: pd.DataFrame, tz_local: str = TZ_LOCAL) -> pd.DataFrame:
    ts_utc = to_utc(df["ts"], tz_local=tz_local)
    lat, lon = parse_point_wkt(df["location"])

    paid = pd.to_numeric(df["paid_occupancy"], errors="coerce")
    space = pd.to_numeric(df["space_count"], errors="coerce")
    rate = pd.to_numeric(df["rate"], errors="coerce") if "rate" in df.columns else np.nan

    out = pd.DataFrame({
        "sourceelementkey": df["sourceelementkey"],
        "ts_utc": ts_utc,
        "paid_occupancy": paid,
        "space_count": space,
        "lat": lat,
        "lon": lon,
        "rate": rate,
        "area": df["area"] if "area" in df.columns else np.nan,
        "subarea": df["subarea"] if "subarea" in df.columns else np.nan,
        "parking_cat": df["parking_cat"] if "parking_cat" in df.columns else np.nan,
        "blockfacename": df["blockfacename"] if "blockfacename" in df.columns else np.nan,
        "sideofstreet": df["sideofstreet"] if "sideofstreet" in df.columns else np.nan,
        "time_limit_cat": df["time_limit_cat"] if "time_limit_cat" in df.columns else np.nan,
    })
    return out

def filter_invalid_rows(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["sourceelementkey", "ts_utc", "lat", "lon", "paid_occupancy", "space_count"]).copy()
    d = d[d["space_count"] > 0].copy()
    d = d[d["lat"].between(-90, 90) & d["lon"].between(-180, 180)].copy()
    return d

def add_flags_and_features(d: pd.DataFrame) -> pd.DataFrame:
    # flags about raw paid_occupancy quality
    d["flag_occ_lt0"] = (d["paid_occupancy"] < 0).astype(np.int8)
    d["flag_occ_gt_spaces"] = (d["paid_occupancy"] > d["space_count"]).astype(np.int8)

    # clamp paid_occupancy into [0, space_count] and KEEP it as a column
    d["paid_occupancy_clamped"] = np.minimum(
        d["paid_occupancy"].clip(lower=0),
        d["space_count"],
    ).astype(np.float32)

    # derived proxy state (based on clamped)
    d["empty_spots"] = (d["space_count"] - d["paid_occupancy_clamped"]).astype(np.float32)
    d["occ_rate"] = (d["paid_occupancy_clamped"] / d["space_count"]).astype(np.float32)

    # binary state
    d["is_full"] = (d["empty_spots"] <= 0).astype(np.int8)
    d["has_space"] = (d["empty_spots"] > K_EMPTY).astype(np.int8)

    # 15-min bin (UTC, stored as naive datetime to match pipeline)
    d["ts15_utc"] = floor_bin(d["ts_utc"], freq=BIN_FREQ).dt.tz_localize(None)

    return d

def aggregate_df15(d: pd.DataFrame) -> pd.DataFrame:

    d = d.sort_values(["sourceelementkey", "ts15_utc", "ts_utc"])
    last = d.groupby(["sourceelementkey", "ts15_utc"], as_index=False).tail(1)

    df15 = d.groupby(["sourceelementkey", "ts15_utc"], as_index=False).agg({
        "space_count": "max",
        "rate": "mean",
        "lat": "median",
        "lon": "median",
        "flag_occ_lt0": "max",
        "flag_occ_gt_spaces": "max",
        "area": "first",
        "subarea": "first",
        "parking_cat": "first",
        "blockfacename": "first",
        "sideofstreet": "first",
        "time_limit_cat": "first",
    })

    df15 = df15.merge(
        last[[
            "sourceelementkey", "ts15_utc",
            "paid_occupancy_clamped",  
            "empty_spots", "occ_rate",
            "is_full", "has_space"
        ]],
        on=["sourceelementkey", "ts15_utc"],
        how="left",
    )

    df15 = df15.rename(columns={"paid_occupancy_clamped": "paid_occupancy"})

    return df15

def preprocess_chunk_to_df15(chunk: pd.DataFrame, tz_local: str = TZ_LOCAL) -> pd.DataFrame:
    """
    Preprocess one CSV *chunk* into 15-minute “street snapshot” rows.

    Output (df15): one row per (sourceelementkey, ts15_utc) with:
    - state: paid_occupancy(**last**), space_count(max), rate(mean), lat/lon(median)
    - derived: empty_spots/occ_rate(**last**), is_full/has_space(**last**)
    - flags: flag_occ_lt0/max, flag_occ_gt_spaces/max
    - metadata: area/subarea/parking_cat/blockfacename/sideofstreet/time_limit_cat (first)
    """
    validate_required_cols(chunk)
    d = parse_and_cast(chunk, tz_local)
    d = filter_invalid_rows(d)
    if d.empty:
        return empty_df15()
    d = add_flags_and_features(d)
    return aggregate_df15(d)