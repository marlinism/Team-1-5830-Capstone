import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from parking_processing.common.constants import TZ_LOCAL
from parking_processing.utils.io import ensure_dir, write_gz_csv
from parking_processing.utils.geo import week_start_local
from parking_processing.utils.schema import build_rename_map, canonical_usecols
from parking_processing.etl.preprocess import preprocess_chunk_to_df15


def run_etl(raw_csv: str, out_dir: str, year: int, chunksize: int = 2_000_000) -> None:
    """
    Stream a yearly CSV and write weekly 15-min snapshot parts to:
      out_dir/year=YYYY/week=YYYY-MM-DD/part_000000.csv.gz
    """
    out_year = os.path.join(out_dir, f"year={year}")
    ensure_dir(out_year)
    print("ETL starts. Output:", out_year)

    header_cols = pd.read_csv(raw_csv, nrows=0).columns
    rename_map = build_rename_map(header_cols)
    usecols = canonical_usecols(header_cols)

    week_counts = defaultdict(int)

    total_in = 0
    total_after_clean = 0
    total_snapshots = 0

    reader = pd.read_csv(raw_csv, chunksize=chunksize, usecols=usecols, low_memory=False)

    for chunk_i, chunk in enumerate(tqdm(reader, desc=f"ETL {year}", unit="chunk")):
        total_in += len(chunk)

        # Rename to canonical schema
        chunk = chunk.rename(columns=rename_map)

        # Preprocess -> 15-min snapshots
        df15 = preprocess_chunk_to_df15(chunk, tz_local=TZ_LOCAL)
        total_after_clean += len(chunk)  # still raw rows; we’ll log snapshots only
        total_snapshots += len(df15)

        if df15.empty:
            continue

        # Assign week partition
        # from .dt.tz_localize("UTC") to tuc=True
        ts15 = pd.to_datetime(df15["ts15_utc"], errors="coerce", utc=True)
        df15["_week"] = week_start_local(ts15, tz_local=TZ_LOCAL)

        # Write per-week parts
        for week_str, g in df15.groupby("_week"):
            week_dir = os.path.join(out_year, f"week={week_str}")
            ensure_dir(week_dir)

            idx = week_counts[week_str]
            week_counts[week_str] += 1
            out_path = os.path.join(week_dir, f"part_{idx:06d}.csv.gz")

            write_gz_csv(g.drop(columns=["_week"]), out_path)

        if (chunk_i + 1) % 10 == 0:
            print(f"[progress] chunks={chunk_i+1} raw_rows_seen={total_in:,} snapshot_rows_written={total_snapshots:,}")

    print("ETL completes:", out_year)