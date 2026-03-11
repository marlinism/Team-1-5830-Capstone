from __future__ import annotations

import os
import io
import json
import tempfile
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch

from parking_processing.utils.s3 import (
    s3_client,
    parse_s3_uri,
    s3_get_text,
    s3_download_file,
)
from parking_processing.utils.graph import (
    load_nodes,
    make_node_index,
    edges_to_indexed,
    build_norm_adj_sparse,
)
from parking_processing.utils.io_pred import (
    write_pred_csv_gz,
    upload_week_predictions,
    week_predictions_exist,
)
from parking_processing.model.mvstgcn import MVSTGCN


Target = Literal["y_15", "y_30", "both"]


@dataclass
class PredictCfg:
    # Data roots
    s3_root: str              # e.g. "s3://smart-park-seattle/parking/"
    year: int

    # Model artifact directory (LOCAL folder OR s3:// prefix)
    # Must contain: mvstgcn.pt and nodes_order.csv
    model_dir: str

    # MV-STGCN parameters
    tin: int = 12
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Graph toggles
    feat_topk: int = 16
    use_topology: int = 1
    use_feature_graph: int = 1

    # Output
    tmp_dir: str = "/tmp/mvstgcn_predict"
    overwrite: bool = False

    # Which horizon(s) to output
    target: Target = "both"


def _identity_adj_sparse(n: int, device: str) -> torch.Tensor:
    """Sparse identity adjacency (used when graph is disabled)."""
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def _load_adjs_from_s3(
    cfg: PredictCfg,
    bucket: str,
    prefix: str,
    node_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load / build normalized sparse adjacency matrices for:
      - topology graph (distance-based)
      - feature graph (cosine similarity)
    """
    s3 = s3_client()

    graphs_prefix = f"{prefix}graphs/year={cfg.year}/"
    mv_prefix = f"{prefix}mvstgcn/year={cfg.year}/"
    feat_key = f"{mv_prefix}feature_edges_top{cfg.feat_topk}.csv"

    # N from graph nodes (safe)
    nodes = load_nodes(s3_get_text(s3, bucket, graphs_prefix + "nodes.csv"))
    N = len(nodes)

    # ---- topology ----
    if cfg.use_topology:
        topo = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, graphs_prefix + "topology_edges.csv")))
        topo["w"] = 1.0 / (pd.to_numeric(topo["distance_m"], errors="coerce").fillna(1e9) + 1.0)
        topo_idx, topo_w = edges_to_indexed(topo, node_to_idx, w_col="w")
        A_topo = build_norm_adj_sparse(N, topo_idx, topo_w).to(cfg.device).coalesce()
    else:
        A_topo = _identity_adj_sparse(N, cfg.device)

    # ---- feature graph ----
    if cfg.use_feature_graph:
        feat = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, feat_key)))

        # be forgiving on column naming
        if "cosine_sim" in feat.columns:
            cos_col = "cosine_sim"
        elif "cosine_sin" in feat.columns:
            cos_col = "cosine_sin"
        else:
            raise KeyError(f"Feature edge file missing cosine column. Have: {list(feat.columns)}")

        feat["w"] = pd.to_numeric(feat[cos_col], errors="coerce").fillna(0).clip(lower=0)
        feat_idx, feat_w = edges_to_indexed(feat, node_to_idx, w_col="w")
        A_feat = build_norm_adj_sparse(N, feat_idx, feat_w).to(cfg.device).coalesce()
    else:
        A_feat = _identity_adj_sparse(N, cfg.device)

    return A_topo, A_feat


def _materialize_model_dir(model_dir: str) -> str:
    """
    If model_dir is s3://... download mvstgcn.pt + nodes_order.csv to a temp local folder
    and return that local folder path. If it's already local, return as-is.
    """
    if isinstance(model_dir, str) and model_dir.startswith("s3://"):
        s3 = s3_client()
        bucket, prefix = parse_s3_uri(model_dir)  # ensures trailing '/'

        local_dir = tempfile.mkdtemp(prefix="mvstgcn_model_")
        pt_local = os.path.join(local_dir, "mvstgcn.pt")
        nodes_local = os.path.join(local_dir, "nodes_order.csv")

        s3_download_file(s3, bucket, prefix + "mvstgcn.pt", pt_local)
        s3_download_file(s3, bucket, prefix + "nodes_order.csv", nodes_local)

        return local_dir

    return model_dir


def _load_model(cfg: PredictCfg):
    """
    Load mvstgcn.pt + nodes_order.csv from cfg.model_dir.
    Returns: (model, nodes_order_df, ckpt_dict, node_to_idx)
    """
    local_model_dir = _materialize_model_dir(cfg.model_dir)

    pt_path = os.path.join(local_model_dir, "mvstgcn.pt")
    nodes_path = os.path.join(local_model_dir, "nodes_order.csv")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing {pt_path}")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Missing {nodes_path}")

    ckpt = torch.load(pt_path, map_location="cpu")
    n_nodes = int(ckpt["n_nodes"])
    n_feats = int(ckpt["n_feats"])
    hidden = int(ckpt["hidden"])

    model = MVSTGCN(n_nodes=n_nodes, n_feats=n_feats, hidden=hidden).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    nodes_order = pd.read_csv(nodes_path)  # column: sourceelementkey
    if "sourceelementkey" not in nodes_order.columns:
        raise KeyError(f"nodes_order.csv must contain 'sourceelementkey'. Have: {list(nodes_order.columns)}")
    if len(nodes_order) != n_nodes:
        raise ValueError(f"nodes_order length {len(nodes_order)} != ckpt n_nodes {n_nodes}")

    node_to_idx = {int(k): i for i, k in enumerate(nodes_order["sourceelementkey"].tolist())}

    # small sanity check: feature dim match at inference time
    if "n_feats" not in ckpt or int(ckpt["n_feats"]) != n_feats:
        raise ValueError("Checkpoint missing/invalid n_feats")

    return model, nodes_order, ckpt, node_to_idx


def _panel_s3_key(prefix: str, year: int, week: str) -> str:
    # matches build_inputs output:
    # mvstgcn/year=YYYY/panels/week=YYYY-MM-DD/panel.npz
    return f"{prefix}mvstgcn/year={year}/panels/week={week}/panel.npz"


def _read_panel_npz(path: str) -> dict:
    """Load panel.npz and return dict-like arrays."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    if "X" not in out:
        raise KeyError(f"panel.npz missing 'X'. Keys={list(out.keys())}")
    return out


def _normalize_ts(ts_arr) -> pd.DatetimeIndex:
    """
    Accept ts stored as:
      - ISO strings
      - numpy datetime64
      - int64 epoch seconds/ms (best-effort)
    """
    try:
        return pd.to_datetime(ts_arr, utc=True)
    except Exception:
        ts = np.asarray(ts_arr)
        if np.issubdtype(ts.dtype, np.integer):
            unit = "ms" if ts.max() > 10_000_000_000 else "s"
            return pd.to_datetime(ts, unit=unit, utc=True)
        return pd.to_datetime(ts.astype(str), utc=True)


@torch.no_grad()
def predict_week_from_panel(
    *,
    model: MVSTGCN,
    A_topo: torch.Tensor,
    A_feat: torch.Tensor,
    nodes_order: pd.DataFrame,
    panel: dict,
    tin: int,
    batch_size: int,
    device: str,
    target: Target,
) -> pd.DataFrame:
    """
    Sliding-window inference on one week's panel.

    panel["X"] shape: [T, N, F]
    timestamps expected in panel["ts_utc"] (preferred),
    or panel["ts15_utc"], or panel["ts"] (length T)

    Returns long dataframe:
      ts15_utc, sourceelementkey, p15?, p30?
    """
    X_all = panel["X"]
    if X_all.ndim != 3:
        raise ValueError(f"panel['X'] must be [T,N,F]. Got shape={X_all.shape}")
    T, N, F = X_all.shape

    if "ts_utc" in panel:
        ts = _normalize_ts(panel["ts_utc"])
    elif "ts15_utc" in panel:
        ts = _normalize_ts(panel["ts15_utc"])
    elif "ts" in panel:
        ts = _normalize_ts(panel["ts"])
    else:
        raise KeyError(f"panel.npz missing timestamps. Keys={list(panel.keys())}")

    if len(ts) != T:
        raise ValueError(f"Timestamps length {len(ts)} != T {T}")
    if tin > T:
        raise ValueError(f"tin={tin} > T={T}. Panel too short.")

    # windows end at t = tin-1 ... T-1
    ends = np.arange(tin - 1, T, dtype=int)
    pred_ts = ts[ends]  # DatetimeIndex length Btot
    Btot = len(ends)

    preds_p15 = []
    preds_p30 = []
    preds_logit15 = []
    preds_logit30 = []

    X_all_t = torch.from_numpy(X_all).float()  # [T,N,F] CPU

    for b0 in range(0, Btot, batch_size):
        b_ends = ends[b0 : b0 + batch_size]

        # [B,Tin,N,F]
        Xb = torch.stack(
            [X_all_t[(t - tin + 1) : (t + 1)] for t in b_ends],
            dim=0,
        ).to(device)

        log15, log30 = model(Xb, A_topo, A_feat)  # each [B,N]

        # store logits + probs (for calibration/debug)
        if target in ("y_15", "both"):
            l15 = log15.detach().cpu().numpy()
            preds_logit15.append(l15)
            preds_p15.append(1.0 / (1.0 + np.exp(-l15)))  # sigmoid in numpy
        if target in ("y_30", "both"):
            l30 = log30.detach().cpu().numpy()
            preds_logit30.append(l30)
            preds_p30.append(1.0 / (1.0 + np.exp(-l30)))

    P15 = np.concatenate(preds_p15, axis=0) if preds_p15 else None  # [Btot,N]
    P30 = np.concatenate(preds_p30, axis=0) if preds_p30 else None
    L15 = np.concatenate(preds_logit15, axis=0) if preds_logit15 else None
    L30 = np.concatenate(preds_logit30, axis=0) if preds_logit30 else None

    keys = nodes_order["sourceelementkey"].to_numpy(dtype=int)

    # Build long dataframe efficiently (no per-timestamp DataFrame concatenation)
    ts_rep = np.repeat(pred_ts.values.astype("datetime64[ns]"), N)
    keys_tile = np.tile(keys, Btot)

    out = pd.DataFrame(
        {
            "ts15_utc": pd.to_datetime(ts_rep, utc=True),
            "sourceelementkey": keys_tile,
        }
    )
    if P15 is not None:
        out["p15"] = P15.reshape(-1)
    if P30 is not None:
        out["p30"] = P30.reshape(-1)
    if L15 is not None:
        out["logit15"] = L15.reshape(-1)
    if L30 is not None:
        out["logit30"] = L30.reshape(-1)

    # already ordered by construction, but keep it explicit:
    out = out.sort_values(["ts15_utc", "sourceelementkey"]).reset_index(drop=True)
    return out


def _week_predictions_exist_safe(cfg: PredictCfg, week: str) -> bool:
    """
    Calls week_predictions_exist(...) with target if the function supports it.
    (Keeps this file compatible if your io_pred.py is slightly behind.)
    """
    try:
        return week_predictions_exist(
            s3_root=cfg.s3_root,
            year=cfg.year,
            week=week,
            target=cfg.target,
        )
    except TypeError:
        # older signature without target
        return week_predictions_exist(
            s3_root=cfg.s3_root,
            year=cfg.year,
            week=week,
        )


def _upload_week_predictions_safe(cfg: PredictCfg, week: str, local_csv_gz: str) -> str:
    """
    Calls upload_week_predictions(...) with target if supported.
    """
    try:
        return upload_week_predictions(
            s3_root=cfg.s3_root,
            year=cfg.year,
            week=week,
            target=cfg.target,
            local_csv_gz=local_csv_gz,
            success_text=f"ok target={cfg.target}\n",
        )
    except TypeError:
        # older signature without target
        return upload_week_predictions(
            s3_root=cfg.s3_root,
            year=cfg.year,
            week=week,
            local_csv_gz=local_csv_gz,
            success_text=f"ok target={cfg.target}\n",
        )


def run_predict_mvstgcn(cfg: PredictCfg, weeks: list[str]) -> None:
    """
    For each week:
      - download panel.npz from S3
      - run inference
      - write + upload pred.csv.gz to S3
    """
    s3 = s3_client()
    bucket, prefix = parse_s3_uri(cfg.s3_root)

    model, nodes_order, ckpt, node_to_idx = _load_model(cfg)
    A_topo, A_feat = _load_adjs_from_s3(cfg, bucket, prefix, node_to_idx)

    os.makedirs(cfg.tmp_dir, exist_ok=True)

    for week in weeks:
        if (not cfg.overwrite) and _week_predictions_exist_safe(cfg, week):
            print(f"[skip] week={week} already has _SUCCESS (target={cfg.target})")
            continue
            
        panel_key = _panel_s3_key(prefix, cfg.year, week)
        local_panel = os.path.join(cfg.tmp_dir, f"panel_{cfg.year}_{week}.npz")
        local_pred = os.path.join(cfg.tmp_dir, f"pred_{cfg.year}_{week}_{cfg.target}.csv.gz")

        print(f"[dl]  s3://{bucket}/{panel_key}")
        s3_download_file(s3, bucket, panel_key, local_panel)

        panel = _read_panel_npz(local_panel)

        df_pred = predict_week_from_panel(
            model=model,
            A_topo=A_topo,
            A_feat=A_feat,
            nodes_order=nodes_order,
            panel=panel,
            tin=cfg.tin,
            batch_size=cfg.batch_size,
            device=cfg.device,
            target=cfg.target,
        )

        write_pred_csv_gz(df_pred, local_pred)
        out_uri = _upload_week_predictions_safe(cfg, week, local_pred)
        print(f"[ok]  week={week} -> {out_uri}")

        # cleanup local temp files (keep tmp_dir itself)
        try:
            os.remove(local_panel)
            os.remove(local_pred)
        except Exception:
            pass