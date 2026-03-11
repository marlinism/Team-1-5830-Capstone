from __future__ import annotations

import duckdb
import os, io, time, json
from dataclasses import dataclass
from urllib.parse import urlparse
from sklearn.neighbors import BallTree

import numpy as np
import pandas as pd
import requests
import boto3
from tqdm import tqdm

from parking_processing.common.constants import TZ_LOCAL
from parking_processing.utils.io import ensure_dir, write_gz_csv
from parking_processing.utils.geo import week_start_local
from parking_processing.etl.preprocess import preprocess_chunk_to_df15


# -----------------------------
# S3 helpers
# -----------------------------
def parse_s3_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 uri must start with s3://")
    u = urlparse(s3_uri)
    bucket = u.netloc
    prefix = u.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def s3_put_text(s3, bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))

# -----------------------------
# Socrata request (CSV) with retries + Retry-After
# -----------------------------
_last_request_ts = 0.0

def request_csv_df_with_retries(session: requests.Session, url: str, params: dict,
                                min_interval_s: float = 0.5, max_retries: int = 20, timeout: int = 120) -> pd.DataFrame:
    global _last_request_ts
    backoff = 2.0

    for _ in range(max_retries):
        # throttle
        now = time.time()
        wait = (_last_request_ts + min_interval_s) - now
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()

        r = session.get(url, params=params, timeout=timeout)

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            sleep_s = float(ra) if (ra and ra.isdigit()) else min(backoff, 120)
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 120)
            continue

        if r.status_code in (500, 502, 503, 504):
            time.sleep(min(backoff, 120))
            backoff = min(backoff * 2, 120)
            continue

        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")

        head = r.content[:120].lstrip()
        if head.startswith(b"{") or head.startswith(b"<"):
            # sometimes a non-CSV error body
            time.sleep(min(backoff, 120))
            backoff = min(backoff * 2, 120)
            continue

        return pd.read_csv(io.BytesIO(r.content), low_memory=False)

    raise RuntimeError("Too many retries requesting Socrata")

def detect_datetime_mode(session: requests.Session, base_csv_url: str, date_col: str, min_interval_s: float):
    """
    Returns:
      ('iso', None) if values look like 2022-01-01T...
      ('mdy', '%m/%d/%Y %I:%M:%S %p') for 04/28/2022 12:49:00 PM
    """
    df = request_csv_df_with_retries(session, base_csv_url, {"$select": date_col, "$limit": 1}, min_interval_s=min_interval_s)
    if df.empty:
        raise RuntimeError("Could not fetch sample row to detect datetime format")
    v = str(df.iloc[0][date_col])
    if "T" in v and len(v) >= 10 and v[:4].isdigit():
        return "iso", None
    return "mdy", "%m/%d/%Y %I:%M:%S %p"

def week_starts_for_year(year: int, tz_local: str):
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz_local)
    start = start - pd.Timedelta(days=start.dayofweek)  # Monday
    end = pd.Timestamp(f"{year+1}-01-01 00:00:00", tz=tz_local)
    cur = start
    weeks = []
    while cur < end:
        weeks.append(cur)
        cur += pd.Timedelta(days=7)
    return weeks

def build_where(date_mode: str, date_col: str, fmt: str | None, w0: pd.Timestamp, w1: pd.Timestamp) -> str:
    s = w0.strftime("%Y-%m-%dT%H:%M:%S.000")
    e = w1.strftime("%Y-%m-%dT%H:%M:%S.000")
    if date_mode == "iso":
        return f"{date_col} >= '{s}' AND {date_col} < '{e}'"
    # text timestamps -> cast in query
    return (
        f"to_floating_timestamp({date_col}, '{fmt}') >= '{s}' AND "
        f"to_floating_timestamp({date_col}, '{fmt}') < '{e}'"
    )

def normalize_and_rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "occupancydatetime": "ts",
        "paidoccupancy": "paid_occupancy",
        "parkingspacecount": "space_count",
        "paidparkingrate": "rate",
        "paidparkingarea": "area",
        "paidparkingsubarea": "subarea",
        "parkingcategory": "parking_cat",
        "parkingtimelimitcategory": "time_limit_cat",
    }
    return df.rename(columns=rename)

# -----------------------------
# Incremental node accumulator
# -----------------------------
class NodeAccumulator:
    def __init__(self):
        self.n = {}
        self.lat_sum = {}
        self.lon_sum = {}
        self.space_max = {}
        self.rate_sum = {}
        self.rate_n = {}
        self.meta_first = {}

    def update(self, df15: pd.DataFrame):
        if df15.empty:
            return
        cols_meta = ["area","subarea","parking_cat","blockfacename","sideofstreet","time_limit_cat"]
        agg = {"lat":"mean","lon":"mean","space_count":"max","rate":"mean"}
        agg.update({c:"first" for c in cols_meta if c in df15.columns})
        g = df15.groupby("sourceelementkey", as_index=False).agg(agg)

        for _, r in g.iterrows():
            k = int(r["sourceelementkey"])
            self.n[k] = self.n.get(k, 0) + 1
            self.lat_sum[k] = self.lat_sum.get(k, 0.0) + float(r["lat"])
            self.lon_sum[k] = self.lon_sum.get(k, 0.0) + float(r["lon"])
            self.space_max[k] = max(self.space_max.get(k, 0), int(r["space_count"]))

            if not pd.isna(r.get("rate", np.nan)):
                self.rate_sum[k] = self.rate_sum.get(k, 0.0) + float(r["rate"])
                self.rate_n[k] = self.rate_n.get(k, 0) + 1

            if k not in self.meta_first:
                self.meta_first[k] = {c: r.get(c, np.nan) for c in cols_meta}

    def finalize(self) -> pd.DataFrame:
        keys = sorted(self.n.keys())
        rows = []
        for k in keys:
            cnt = self.n[k]
            lat = self.lat_sum[k] / cnt
            lon = self.lon_sum[k] / cnt
            rate = (self.rate_sum.get(k, 0.0) / self.rate_n[k]) if self.rate_n.get(k, 0) else np.nan
            meta = self.meta_first.get(k, {})
            rows.append({
                "sourceelementkey": k,
                "lat": lat,
                "lon": lon,
                "space_count_max": self.space_max.get(k, 0),
                "rate_mean": rate,
                **meta
            })
        return pd.DataFrame(rows)

# -----------------------------
# Topology edges (kNN)
# -----------------------------
def build_topology_edges_knn(nodes: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    R = 6371000.0

    X = np.deg2rad(nodes[["lat","lon"]].to_numpy(dtype=float))
    tree = BallTree(X, metric="haversine")
    dist, ind = tree.query(X, k=k+1)

    keys = nodes["sourceelementkey"].to_numpy()
    edges = []
    for i in range(len(nodes)):
        for jpos in range(1, k+1):
            j = ind[i, jpos]
            edges.append((int(keys[i]), int(keys[j]), float(dist[i, jpos] * R)))

    e = pd.DataFrame(edges, columns=["src","dst","distance_m"])
    a = np.minimum(e["src"].values, e["dst"].values)
    b = np.maximum(e["src"].values, e["dst"].values)
    e["a"] = a; e["b"] = b
    e = e.groupby(["a","b"], as_index=False)["distance_m"].min()
    return e.rename(columns={"a":"src","b":"dst"})

# -----------------------------
# Split train/validate/test (DuckDB)
# -----------------------------
def label_week_parts_to_gz(in_glob: str, out_csv_gz: str):
    os.makedirs(os.path.dirname(out_csv_gz), exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
    COPY (
      SELECT *
      FROM (
        SELECT
          *,
          CAST(LEAD(has_space, 1) OVER (PARTITION BY sourceelementkey ORDER BY ts15_utc) AS TINYINT) AS y_15,
          CAST(LEAD(has_space, 2) OVER (PARTITION BY sourceelementkey ORDER BY ts15_utc) AS TINYINT) AS y_30
        FROM read_csv_auto('{in_glob}', union_by_name=true)
      )
      WHERE y_15 IS NOT NULL AND y_30 IS NOT NULL
    )
    TO '{out_csv_gz}' (FORMAT CSV, HEADER TRUE, COMPRESSION GZIP);
    """)
    con.close()

def build_splits_from_weeks(weeks: list[str], val_weeks=4, test_weeks=8):
    weeks = sorted(weeks)
    if len(weeks) < (val_weeks + test_weeks + 2):
        raise ValueError(f"Not enough weeks ({len(weeks)}) for splits.")
    return {
        "train_weeks": weeks[:-(val_weeks+test_weeks)],
        "val_weeks": weeks[-(val_weeks+test_weeks):-test_weeks],
        "test_weeks": weeks[-test_weeks:],
    }

# -----------------------------
# Config + pipeline
# -----------------------------
@dataclass
class Config:
    domain: str
    dataset_id: str
    year: int
    out_s3: str

    date_col: str = "occupancydatetime"
    limit: int = 50_000
    tz_local: str = TZ_LOCAL
    min_interval_s: float = 0.5

    local_tmp: str = "/tmp/smartpark_pipeline"
    knn_k: int = 8
    val_weeks: int = 4
    test_weeks: int = 8

def run_pipeline(cfg: Config):
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(cfg.out_s3)

    base_csv_url = f"https://{cfg.domain}/resource/{cfg.dataset_id}.csv"

    session = requests.Session()
    tok = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    if tok:
        session.headers.update({"X-App-Token": tok})  # App Token only

    date_mode, fmt = detect_datetime_mode(session, base_csv_url, cfg.date_col, cfg.min_interval_s)

    node_acc = NodeAccumulator()
    ensure_dir(cfg.local_tmp)

    processed_weeks = set()

    select_cols = [
        "occupancydatetime","paidoccupancy","blockfacename","sideofstreet","sourceelementkey",
        "parkingtimelimitcategory","parkingspacecount","paidparkingarea","paidparkingsubarea",
        "paidparkingrate","parkingcategory","location",
    ]

    for w0 in week_starts_for_year(cfg.year, cfg.tz_local):
        w1 = w0 + pd.Timedelta(days=7)
        week_str = w0.strftime("%Y-%m-%d")

        clean_week_prefix = f"{prefix}clean/year={cfg.year}/week={week_str}/"
        labeled_week_prefix = f"{prefix}labeled/year={cfg.year}/week={week_str}/"

        clean_success = clean_week_prefix + "_SUCCESS"
        labeled_success = labeled_week_prefix + "_SUCCESS"

        # if labeled exists, week is fully done
        if s3_exists(s3, bucket, labeled_success):
            processed_weeks.add(week_str)
            continue

        # local week dirs
        week_dir = os.path.join(cfg.local_tmp, f"year={cfg.year}", f"week={week_str}")
        parts_dir = os.path.join(week_dir, "parts")
        ensure_dir(parts_dir)

        where = build_where(date_mode, cfg.date_col, fmt, w0, w1)

        offset = 0
        parts_written = 0
        pbar = tqdm(desc=f"week {week_str}", unit="rows", leave=False)

        while True:
            part_key = clean_week_prefix + f"parts/offset={offset:09d}.csv.gz"
            local_part = os.path.join(parts_dir, f"offset={offset:09d}.csv.gz")

            if s3_exists(s3, bucket, part_key):
                offset += cfg.limit
                continue

            params = {
                "$select": ",".join(select_cols),
                "$where": where,
                "$order": f"{cfg.date_col} ASC",
                "$limit": cfg.limit,
                "$offset": offset,
            }

            raw = request_csv_df_with_retries(session, base_csv_url, params, min_interval_s=cfg.min_interval_s)
            if raw.empty:
                break

            pbar.update(len(raw))

            chunk = normalize_and_rename_to_canonical(raw)
            df15 = preprocess_chunk_to_df15(chunk, tz_local=cfg.tz_local)

            node_acc.update(df15)

            if not df15.empty:
                # guard: keep only this local-week partition
                ts15 = pd.to_datetime(df15["ts15_utc"], errors="coerce").dt.tz_localize("UTC")
                df15["_week"] = week_start_local(ts15, tz_local=cfg.tz_local)
                df15 = df15[df15["_week"] == week_str].drop(columns=["_week"])

                if not df15.empty:
                    write_gz_csv(df15, local_part)
                    s3.upload_file(local_part, bucket, part_key)
                    parts_written += 1

            if len(raw) < cfg.limit:
                break

            offset += cfg.limit

        pbar.close()
        s3_put_text(s3, bucket, clean_success, f"parts={parts_written}\n")

        # ---- label this week from local parts ----
        labeled_local = os.path.join(week_dir, "labeled.csv.gz")
        in_glob = os.path.join(parts_dir, "*.csv.gz")
        if parts_written > 0:
            label_week_parts_to_gz(in_glob, labeled_local)
            s3.upload_file(labeled_local, bucket, labeled_week_prefix + "data.csv.gz")
            os.remove(labeled_local)

        s3_put_text(s3, bucket, labeled_success, "ok\n")

        # cleanup local weekly parts to save disk
        for fn in os.listdir(parts_dir):
            if fn.endswith(".csv.gz"):
                os.remove(os.path.join(parts_dir, fn))

        processed_weeks.add(week_str)
        print(f"week={week_str} parts={parts_written} labeled={'yes' if parts_written>0 else 'no'}")

    # ---- finalize artifacts ----
    nodes = node_acc.finalize()
    graphs_prefix = f"{prefix}graphs/year={cfg.year}/"
    splits_prefix = f"{prefix}splits/year={cfg.year}/"

    buf = io.StringIO(); nodes.to_csv(buf, index=False)
    s3_put_text(s3, bucket, graphs_prefix + "nodes.csv", buf.getvalue())

    topo = build_topology_edges_knn(nodes, k=cfg.knn_k)
    buf = io.StringIO(); topo.to_csv(buf, index=False)
    s3_put_text(s3, bucket, graphs_prefix + "topology_edges.csv", buf.getvalue())

    splits = build_splits_from_weeks(sorted(processed_weeks), val_weeks=cfg.val_weeks, test_weeks=cfg.test_weeks)
    s3_put_text(s3, bucket, splits_prefix + "splits.json", json.dumps(splits, indent=2))

    print(f"clean:   s3://{bucket}/{prefix}clean/year={cfg.year}/week=*/parts/*.csv.gz")
    print(f"labeled: s3://{bucket}/{prefix}labeled/year={cfg.year}/week=*/data.csv.gz")
    print(f"nodes:   s3://{bucket}/{graphs_prefix}nodes.csv")
    print(f"topo:    s3://{bucket}/{graphs_prefix}topology_edges.csv")
    print(f"splits:  s3://{bucket}/{splits_prefix}splits.json")

if __name__ == "__main__":
    cfg = Config(
        domain="cos-data.seattle.gov",
        dataset_id=os.environ.get("PARKING_DATASET_ID", "bwk6-iycu"),
        year=int(os.environ.get("PARKING_YEAR", "2022")),
        out_s3=os.environ["PARKING_OUT_S3"],
        min_interval_s=float(os.environ.get("PARKING_MIN_INTERVAL", "0.5")),
        knn_k=int(os.environ.get("PARKING_KNN_K", "8")),
    )
    run_pipeline(cfg)