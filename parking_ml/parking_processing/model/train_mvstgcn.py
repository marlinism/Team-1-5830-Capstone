from __future__ import annotations
import os, io, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from parking_processing.utils.s3 import s3_client, parse_s3_uri, s3_get_text, s3_list_keys, s3_download_file
from parking_processing.utils.graph import load_nodes, make_node_index, edges_to_indexed, build_norm_adj_sparse
from parking_processing.model.mvstgcn import MVSTGCN
from parking_processing.model.dataset import TimeWindowDataset


def _download_panels(s3, bucket: str, panels_prefix: str, weeks: list[str], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for wk in weeks:
        key = f"{panels_prefix}week={wk}/panel.npz"
        local = os.path.join(out_dir, f"panel_{wk}.npz")
        s3_download_file(s3, bucket, key, local)
        paths.append(local)
    return paths

def identity_adj_sparse(n: int, device: str):
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3_root", required=True, help="s3://smart-park-seattle/parking/")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--tin", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--feat_topk", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target", default="both", choices=["y_15", "y_30", "both"])
    ap.add_argument("--use_feature_graph", type=int, default=1, choices=[0,1])
    ap.add_argument("--use_topology", type=int, default=1, choices=[0,1])
    ap.add_argument("--early_stop", type=int, default=1, choices=[0, 1])
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    args = ap.parse_args()

    s3 = s3_client()
    bucket, prefix = parse_s3_uri(args.s3_root)

    graphs_prefix = f"{prefix}graphs/year={args.year}/"
    splits_key    = f"{prefix}splits/year={args.year}/splits.json"
    mv_prefix     = f"{prefix}mvstgcn/year={args.year}/"
    panels_prefix = f"{mv_prefix}panels/"
    feat_key      = f"{mv_prefix}feature_edges_top{args.feat_topk}.csv"

    # ---- load nodes + edges ----
    nodes = load_nodes(s3_get_text(s3, bucket, graphs_prefix + "nodes.csv"))
    node_to_idx = make_node_index(nodes)
    N = len(nodes)

    use_topo = bool(args.use_topology)
    use_feat = bool(args.use_feature_graph)

    if use_topo:
        topo = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, graphs_prefix + "topology_edges.csv")))
        topo["w"] = 1.0 / (pd.to_numeric(topo["distance_m"], errors="coerce").fillna(1e9) + 1.0)
        topo_idx, topo_w = edges_to_indexed(topo, node_to_idx, w_col="w")
        A_topo = build_norm_adj_sparse(N, topo_idx, topo_w).to(args.device).coalesce()
    else:
        A_topo = identity_adj_sparse(N, args.device)

    if use_feat:
        feat = pd.read_csv(io.StringIO(s3_get_text(s3, bucket, feat_key)))
        cos_col = "cosine_sim" if "cosine_sim" in feat.columns else "cosine_sin"
        feat["w"] = pd.to_numeric(feat[cos_col], errors="coerce").fillna(0).clip(lower=0)
        feat_idx, feat_w = edges_to_indexed(feat, node_to_idx, w_col="w")
        A_feat = build_norm_adj_sparse(N, feat_idx, feat_w).to(args.device).coalesce()
    else:
        A_feat = identity_adj_sparse(N, args.device)

    splits = json.loads(s3_get_text(s3, bucket, splits_key))
    train_weeks = splits["train_weeks"]
    val_weeks   = splits["val_weeks"]

    # ---- download panels locally (simple + reliable) ----
    local_root = "/tmp/mvstgcn_panels"
    train_paths = _download_panels(s3, bucket, panels_prefix, train_weeks, os.path.join(local_root, "train"))
    val_paths   = _download_panels(s3, bucket, panels_prefix, val_weeks,   os.path.join(local_root, "val"))

    # infer n_feats from one panel
    sample = np.load(train_paths[0])
    n_feats = sample["X"].shape[-1]

    train_ds = TimeWindowDataset(train_paths, tin=args.tin)
    val_ds   = TimeWindowDataset(val_paths, tin=args.tin)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MVSTGCN(n_nodes=N, n_feats=n_feats, hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    def _clip_prob(p, eps=1e-6):
        return np.clip(p, eps, 1 - eps)

    def _safe_auc(y, p):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    
    def _safe_ap(y, p):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    
    def _pred_stats(p):
        p = np.asarray(p).astype(float)
        return {
            "p_min": float(p.min()),
            "p_med": float(np.median(p)),
            "p_max": float(p.max()),
            "p_gt_99": float((p >= 0.99).mean()),
        }
        
    def run_eval():
        model.eval()
        tot_loss = 0.0
        tot_cnt = 0.0
        y_all = []
        p_all = []

        with torch.no_grad():
            for X, y15, y30, m15, m30 in val_dl:
                X = X.to(args.device)
                y15 = y15.to(args.device)
                y30 = y30.to(args.device)
                m15 = m15.to(args.device)
                m30 = m30.to(args.device)

                log15, log30 = model(X, A_topo, A_feat)

                if args.target == "y_15":
                    loss_num = (bce(log15, y15) * m15).sum()
                    cnt = m15.sum().clamp(min=1.0)

                    prob = torch.sigmoid(log15).detach().cpu().numpy().reshape(-1)
                    yy = y15.detach().cpu().numpy().reshape(-1)
                    mm = m15.detach().cpu().numpy().reshape(-1)

                elif args.target == "y_30":
                    loss_num = (bce(log30, y30) * m30).sum()
                    cnt = m30.sum().clamp(min=1.0)

                    prob = torch.sigmoid(log30).detach().cpu().numpy().reshape(-1)
                    yy = y30.detach().cpu().numpy().reshape(-1)
                    mm = m30.detach().cpu().numpy().reshape(-1)

                else:
                    loss_num = (bce(log15, y15) * m15).sum() + (bce(log30, y30) * m30).sum()
                    cnt = (m15.sum() + m30.sum()).clamp(min=1.0)
                    prob15 = torch.sigmoid(log15).detach().cpu().numpy().reshape(-1)
                    yy15 = y15.detach().cpu().numpy().reshape(-1)
                    mm15 = m15.detach().cpu().numpy().reshape(-1)
                    prob30 = torch.sigmoid(log30).detach().cpu().numpy().reshape(-1)
                    yy30 = y30.detach().cpu().numpy().reshape(-1)
                    mm30 = m30.detach().cpu().numpy().reshape(-1)
                    # concat 15+30 for metrics
                    prob = np.concatenate([prob15, prob30])
                    yy = np.concatenate([yy15, yy30])
                    mm = np.concatenate([mm15, mm30])
                tot_loss += float(loss_num.detach().cpu())
                tot_cnt  += float(cnt.detach().cpu())
                keep = (mm > 0.5)
                if keep.any():
                    y_all.append(yy[keep])
                    p_all.append(prob[keep])

        val_loss = tot_loss / max(tot_cnt, 1.0)
    
        if len(y_all):
            y = np.concatenate(y_all).astype(int)
            p = _clip_prob(np.concatenate(p_all).astype(float))
            metrics = {
                "pos_rate": float(y.mean()),
                "auc": _safe_auc(y, p),
                "ap": _safe_ap(y, p),
                "logloss": float(log_loss(y, p, labels=[0,1])),
                **_pred_stats(p),
            }
        else:
            metrics = {}

        return val_loss, metrics

    # for early stop
    best_val = float("inf")
    best_ep = 0
    bad_epochs = 0
    best_state = None
    
    # ---- train ----
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_cnt_sum = 0.0

        for X, y15, y30, m15, m30 in train_dl:
            X = X.to(args.device)
            y15 = y15.to(args.device)
            y30 = y30.to(args.device)
            m15 = m15.to(args.device)
            m30 = m30.to(args.device)

            opt.zero_grad()
            log15, log30 = model(X, A_topo, A_feat)

            if args.target == "y_15":
                loss_num = (bce(log15, y15) * m15).sum()
                denom = m15.sum().clamp(min=1.0)
            elif args.target == "y_30":
                loss_num = (bce(log30, y30) * m30).sum()
                denom = m30.sum().clamp(min=1.0)
            else:
                loss_num = (bce(log15, y15) * m15).sum() + (bce(log30, y30) * m30).sum()
                denom = (m15.sum() + m30.sum()).clamp(min=1.0)

            loss = loss_num / denom
            loss.backward()
            opt.step()

            # accumulate train loss in the SAME normalization as val (sum / sum(mask))
            tr_loss_sum += float(loss_num.detach().cpu())
            tr_cnt_sum  += float(denom.detach().cpu())

        train_loss = tr_loss_sum / max(tr_cnt_sum, 1.0)
        val_loss, vm = run_eval()

        msg = f"epoch {ep}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        if vm:
            msg += " " + " ".join([f"{k}={v:.4f}" for k, v in vm.items()])
        print(msg)

        # ---- early stopping + best checkpoint ----
        improved = (best_val - val_loss) > args.min_delta
        
        if improved:
            best_val = val_loss
            best_ep = ep
            bad_epochs = 0
            # keep best weights in memory
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
            # also save a best checkpoint for debugging / recovery
            model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_nodes": N,
                    "n_feats": n_feats,
                    "hidden": args.hidden,
                    "tin": args.tin,
                    "year": args.year,
                    "target": args.target,
                    "use_topology": int(args.use_topology),
                    "use_feature_graph": int(args.use_feature_graph),
                    "feat_topk": args.feat_topk,
                    "best_epoch": best_ep,
                    "best_val_loss": float(best_val),
                },
                os.path.join(model_dir, "mvstgcn_best.pt"),
            )
        else:
            bad_epochs += 1

        if args.early_stop and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {ep}: best_ep={best_ep} best_val={best_val:.6f}")
            break


    # If we have a best_state, restore it before final save
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # ---- save for SageMaker ----
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_nodes": N,
            "n_feats": n_feats,
            "hidden": args.hidden,
            "tin": args.tin,
            "year": args.year,
            "target": args.target,
            "use_topology": int(args.use_topology),
            "use_feature_graph": int(args.use_feature_graph),
            "feat_topk": args.feat_topk,
            "best_epoch": best_ep,
            "best_val_loss": float(best_val),
        },
        os.path.join(model_dir, "mvstgcn.pt"),
    )

    # save node order so inference can map back
    nodes_out = nodes[["sourceelementkey"]].copy()
    nodes_out.to_csv(os.path.join(model_dir, "nodes_order.csv"), index=False)

    print("Saved model to:", model_dir)


if __name__ == "__main__":
    main()