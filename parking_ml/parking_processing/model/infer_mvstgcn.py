from __future__ import annotations
import os, io, json, argparse, gzip
import numpy as np
import pandas as pd
import torch

from parking_processing.utils.s3 import s3_client, parse_s3_uri, s3_get_text, s3_download_file, s3_upload_file
from parking_processing.utils.graph import load_nodes, make_node_index, edges_to_indexed, build_norm_adj_sparse
from parking_processing.model.mvstgcn import MVSTGCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3_root", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--feat_topk", type=int, default=16)
    ap.add_argument("--tin", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_path", default="/opt/ml/model/mvstgcn.pt")
    args = ap.parse_args()

    s3 = s3_client()
    bucket, prefix = parse_s3_uri(args.s3_root)

    graphs_prefix = f"{prefix}graphs/year={args.year}/"
    splits_key    = f"{prefix}splits/year={args.year}/splits.json"
    mv_prefix     = f"{prefix}mvstgcn/year={args.year}/"
    panels_prefix = f"{mv_prefix}panels/"
    feat_key      = f"{mv_prefix}feature_edges_top{args.feat_topk}.csv"

    preds_prefix  = f"{prefix}preds/parking/year={args.year}/"  # output

    # load graph
    nodes = load_nodes(s3_get_text(s3, bucket, graphs_prefix + "nodes.csv"))
    node_to_idx = make_node_index(nodes)
    keys = nodes["sourceelementkey"].astype(int).to_numpy()
    N = len(nodes)

    topo = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, graphs_prefix + "topology_edges.csv")))
    topo["w"] = 1.0 / (pd.to_numeric(topo["distance_m"], errors="coerce").fillna(1e9) + 1.0)
    topo_idx, topo_w = edges_to_indexed(topo, node_to_idx, w_col="w")

    feat = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, feat_key)))
    feat["w"] = pd.to_numeric(feat["cosine_sim"], errors="coerce").fillna(0).clip(lower=0)
    feat_idx, feat_w = edges_to_indexed(feat, node_to_idx, w_col="w")

    A_topo = build_norm_adj_sparse(N, topo_idx, topo_w).to(args.device)
    A_feat = build_norm_adj_sparse(N, feat_idx, feat_w).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device)
    n_feats = int(ckpt["n_feats"])
    hidden = int(ckpt["hidden"])

    model = MVSTGCN(n_nodes=N, n_feats=n_feats, hidden=hidden).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    splits = json.loads(s3_get_text(s3, bucket, splits_key))
    test_weeks = splits["test_weeks"]

    tmp_dir = "/tmp/mvstgcn_infer"
    os.makedirs(tmp_dir, exist_ok=True)

    for wk in test_weeks:
        panel_key = f"{panels_prefix}week={wk}/panel.npz"
        local_panel = os.path.join(tmp_dir, f"panel_{wk}.npz")
        s3_download_file(s3, bucket, panel_key, local_panel)
        z = np.load(local_panel, allow_pickle=False)

        X = z["X"].astype(np.float32)   # [T,N,F]
        ts = z["ts_utc"]                # [T]
        T = X.shape[0]

        rows = []
        with torch.no_grad():
            for t in range(args.tin, T):
                Xin = torch.tensor(X[t-args.tin:t][None, ...], dtype=torch.float32, device=args.device)  # [1,Tin,N,F]
                log15, log30 = model(Xin, A_topo, A_feat)
                p15 = torch.sigmoid(log15)[0].detach().cpu().numpy()
                p30 = torch.sigmoid(log30)[0].detach().cpu().numpy()

                ts_t = pd.to_datetime(ts[t]).strftime("%Y-%m-%d %H:%M:%S")
                for i, k in enumerate(keys):
                    rows.append((ts_t, int(k), float(p15[i]), float(p30[i])))

        out_df = pd.DataFrame(rows, columns=["ts15_utc", "sourceelementkey", "p_has_space_15", "p_has_space_30"])
        out_local = os.path.join(tmp_dir, f"preds_{wk}.csv.gz")
        out_df.to_csv(out_local, index=False, compression="gzip")

        out_key = f"{preds_prefix}week={wk}/preds.csv.gz"
        s3_upload_file(s3, out_local, bucket, out_key)

        os.remove(local_panel)
        os.remove(out_local)

        print("Wrote:", f"s3://{bucket}/{out_key}")


if __name__ == "__main__":
    main()