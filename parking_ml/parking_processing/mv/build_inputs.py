from __future__ import annotations
import os, io, json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from tqdm import tqdm

from parking_processing.common.constants import TZ_LOCAL
from parking_processing.utils.s3 import (
    s3_client, parse_s3_uri, s3_get_text, s3_put_text,
    s3_read_csv_gz, s3_upload_file,
)
from parking_processing.utils.graph import load_nodes
from parking_processing.mv.feature_graph import build_signature_from_train_weeks, build_feature_edges_topk_cosine
from parking_processing.mv.panelize import panelize_week


@dataclass
class BuildInputsCfg:
    s3_root: str   # s3://smart-park-seattle/parking/
    year: int
    feat_topk: int = 16
    sig_mode: str = "dow_hour"  # "dow_hour" or "dow_hour_q"
    tmp_dir: str = "/tmp/smartpark_mv_inputs"


def run_build_inputs(cfg: BuildInputsCfg):
    s3 = s3_client()
    bucket, prefix = parse_s3_uri(cfg.s3_root)

    labeled_prefix = f"{prefix}labeled/year={cfg.year}/"
    graphs_prefix  = f"{prefix}graphs/year={cfg.year}/"
    splits_key     = f"{prefix}splits/year={cfg.year}/splits.json"

    out_prefix     = f"{prefix}mvstgcn/year={cfg.year}/"
    panels_prefix  = f"{out_prefix}panels/"
    feat_key       = f"{out_prefix}feature_edges_top{cfg.feat_topk}.csv"
    node_map_key   = f"{out_prefix}node_id_map.json"
    success_key    = f"{out_prefix}_SUCCESS"

    os.makedirs(cfg.tmp_dir, exist_ok=True)

    # nodes define N and ordering
    nodes_text = s3_get_text(s3, bucket, graphs_prefix + "nodes.csv")
    nodes = load_nodes(nodes_text)
    node_keys = nodes["sourceelementkey"].astype(int).to_numpy()

    node_id_map = {str(int(k)): int(i) for i, k in enumerate(node_keys)}
    s3_put_text(s3, bucket, node_map_key, json.dumps(node_id_map))

    splits = json.loads(s3_get_text(s3, bucket, splits_key))
    train_weeks = splits["train_weeks"]
    all_weeks = sorted(set(train_weeks + splits["val_weeks"] + splits["test_weeks"]))

    # feature graph (train weeks only)
    S = build_signature_from_train_weeks(
        s3, bucket, labeled_prefix, train_weeks,
        node_keys=node_keys, tz_local=TZ_LOCAL, mode=cfg.sig_mode
    )
    feat_edges = build_feature_edges_topk_cosine(S, node_keys, k=cfg.feat_topk)

    buf = io.StringIO()
    feat_edges.to_csv(buf, index=False)
    s3_put_text(s3, bucket, feat_key, buf.getvalue())

    # panels per week
    for wk in tqdm(all_weeks, desc="Panelize weeks"):
        in_key = f"{labeled_prefix}week={wk}/data.csv.gz"
        df = s3_read_csv_gz(s3, bucket, in_key)
        if df.empty:
            continue

        X, y15, y30, mask, ts_utc = panelize_week(df, node_keys=node_keys, week_str=wk, tz_local=TZ_LOCAL)

        local_npz = os.path.join(cfg.tmp_dir, f"panel_{cfg.year}_{wk}.npz")
        np.savez_compressed(local_npz, X=X, y15=y15, y30=y30, mask=mask, ts_utc=ts_utc, node_keys=node_keys)

        out_key = f"{panels_prefix}week={wk}/panel.npz"
        s3_upload_file(s3, local_npz, bucket, out_key)
        os.remove(local_npz)

    s3_put_text(s3, bucket, success_key, "ok\n")

    print("MV inputs written:")
    print(f"  node map:      s3://{bucket}/{node_map_key}")
    print(f"  feature edges: s3://{bucket}/{feat_key}")
    print(f"  panels:        s3://{bucket}/{panels_prefix}week=*/panel.npz")
    print(f"  success:       s3://{bucket}/{success_key}")