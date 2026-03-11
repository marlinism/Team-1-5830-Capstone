from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from parking_processing.common.constants import TZ_LOCAL
from parking_processing.utils.s3 import s3_read_csv_gz


def build_signature_from_train_weeks(
    s3, bucket: str, labeled_prefix: str, train_weeks: list[str],
    node_keys: np.ndarray,
    tz_local: str = TZ_LOCAL,
    mode: str = "dow_hour",  # "dow_hour" (168) or "dow_hour_q" (672)
) -> np.ndarray:
    """
    Create a per-node usage signature used to build the feature graph.
    Signature dims:
      - dow_hour: 7*24 = 168
      - dow_hour_q: 7*24*4 = 672
    """
    key_to_idx = {int(k): i for i, k in enumerate(node_keys)}
    N = len(node_keys)
    B = 168 if mode == "dow_hour" else 672

    S_sum = np.zeros((N, B), dtype=np.float64)
    S_cnt = np.zeros((N, B), dtype=np.int64)

    usecols = ["sourceelementkey", "ts15_utc", "occ_rate"]
    for wk in tqdm(train_weeks, desc="Signature (train weeks)"):
        key = f"{labeled_prefix}week={wk}/data.csv.gz"
        df = s3_read_csv_gz(s3, bucket, key, usecols=usecols)
        if df.empty:
            continue

        ts = pd.to_datetime(df["ts15_utc"], errors="coerce", utc=True)
        ok = ts.notna() & df["sourceelementkey"].notna() & df["occ_rate"].notna()
        df = df.loc[ok].copy()
        ts = ts.loc[ok]

        tloc = ts.dt.tz_convert(tz_local)
        dow = tloc.dt.dayofweek.to_numpy(np.int64)
        hour = tloc.dt.hour.to_numpy(np.int64)

        if mode == "dow_hour":
            b = dow * 24 + hour
        else:
            q = (tloc.dt.minute.to_numpy(np.int64) // 15)
            b = (dow * 24 + hour) * 4 + q

        n = df["sourceelementkey"].astype(int).map(key_to_idx)
        ok2 = n.notna()
        n = n.loc[ok2].astype(int).to_numpy(np.int64)
        b = b[ok2.to_numpy()]
        v = df.loc[ok2, "occ_rate"].to_numpy(np.float64)

        np.add.at(S_sum, (n, b), v)
        np.add.at(S_cnt, (n, b), 1)

    bucket_mean = S_sum.sum(axis=0) / np.maximum(S_cnt.sum(axis=0), 1)
    S = S_sum / np.maximum(S_cnt, 1)
    miss = (S_cnt == 0)
    S[miss] = bucket_mean[np.where(miss)[1]]
    return S.astype(np.float32)


def build_feature_edges_topk_cosine(
    S: np.ndarray,
    node_keys: np.ndarray,
    k: int = 16,
    block: int = 1024,
) -> pd.DataFrame:
    """
    Cosine top-k feature graph edges.
    Output: src, dst, cosine_sim (undirected unique)
    """
    N = S.shape[0]
    norm = np.linalg.norm(S, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    U = S / norm

    edges = []
    for i0 in tqdm(range(0, N, block), desc=f"Feature graph top{k}"):
        i1 = min(N, i0 + block)
        sims = U[i0:i1] @ U.T  # [block,N]

        for bi in range(i1 - i0):
            sims[bi, i0 + bi] = -np.inf  # remove self

        top_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        top_sim = np.take_along_axis(sims, top_idx, axis=1)

        order = np.argsort(-top_sim, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)

        for bi in range(i1 - i0):
            src_key = int(node_keys[i0 + bi])
            for j, s in zip(top_idx[bi], top_sim[bi]):
                if not np.isfinite(s):
                    continue
                dst_key = int(node_keys[j])
                edges.append((src_key, dst_key, float(s)))

    e = pd.DataFrame(edges, columns=["src", "dst", "cosine_sim"])
    a = np.minimum(e["src"].values, e["dst"].values)
    b = np.maximum(e["src"].values, e["dst"].values)
    e["a"] = a; e["b"] = b
    e = e.groupby(["a", "b"], as_index=False)["cosine_sim"].max()
    return e.rename(columns={"a": "src", "b": "dst"})