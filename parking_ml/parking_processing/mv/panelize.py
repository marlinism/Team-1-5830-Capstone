from __future__ import annotations
import numpy as np
import pandas as pd

from parking_processing.common.constants import TZ_LOCAL

def panelize_week(
    df: pd.DataFrame,
    node_keys: np.ndarray,
    week_str: str,
    tz_local: str = TZ_LOCAL,
):
    """
    Convert row-wise week df into tensors for MV-STGCN.

    Returns:
      X:    [T, N, F] float32
      y15:  [T, N] int8 (-1 means missing)
      y30:  [T, N] int8 (-1 means missing)
      mask: [T, N] uint8 (1 means observation exists at t,n)
      ts_utc: [T] datetime64[ns] UTC grid
    """
    key_to_idx = {int(k): i for i, k in enumerate(node_keys)}
    N = len(node_keys)

    # Week grid: Monday 00:00 local → 7 days × 15-min bins
    start_local = pd.Timestamp(week_str + " 00:00:00", tz=tz_local)
    ts_local = pd.date_range(start_local, periods=7 * 24 * 4, freq="15min")
    ts_utc = ts_local.tz_convert("UTC")
    T = len(ts_utc)
    t0 = ts_utc[0]

    F_cols = ["occ_rate", "empty_spots"]
    if "rate" in df.columns:
        F_cols.append("rate")
    F = len(F_cols)

    X = np.zeros((T, N, F), dtype=np.float32)
    mask = np.zeros((T, N), dtype=np.uint8)
    y15 = np.full((T, N), -1, dtype=np.int8)
    y30 = np.full((T, N), -1, dtype=np.int8)

    ts = pd.to_datetime(df["ts15_utc"], errors="coerce", utc=True)
    ok = ts.notna() & df["sourceelementkey"].notna()
    df = df.loc[ok].copy()
    ts = ts.loc[ok]

    n_idx = df["sourceelementkey"].astype(int).map(key_to_idx)
    ok2 = n_idx.notna()
    df = df.loc[ok2].copy()
    ts = ts.loc[ok2]
    n_idx = n_idx.loc[ok2].astype(int).to_numpy(np.int64)

    dt = (ts - t0).dt.total_seconds().to_numpy(np.int64)
    t_idx = (dt // (15 * 60)).astype(np.int64)

    in_range = (t_idx >= 0) & (t_idx < T)
    df = df.loc[in_range].copy()
    n_idx = n_idx[in_range]
    t_idx = t_idx[in_range]

    for fi, c in enumerate(F_cols):
        X[t_idx, n_idx, fi] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float32)

    mask[t_idx, n_idx] = 1

    if "y_15" in df.columns:
        y15[t_idx, n_idx] = pd.to_numeric(df["y_15"], errors="coerce").fillna(-1).to_numpy(np.int8)
    if "y_30" in df.columns:
        y30[t_idx, n_idx] = pd.to_numeric(df["y_30"], errors="coerce").fillna(-1).to_numpy(np.int8)

    return X, y15, y30, mask, ts_utc.to_numpy(dtype="datetime64[ns]")