from __future__ import annotations
import io
import numpy as np
import pandas as pd
import torch


def load_nodes(nodes_csv_text: str) -> pd.DataFrame:
    nodes = pd.read_csv(io.StringIO(nodes_csv_text))
    nodes = nodes.sort_values("sourceelementkey").reset_index(drop=True)
    return nodes


def make_node_index(nodes: pd.DataFrame) -> dict[int, int]:
    keys = nodes["sourceelementkey"].astype(int).to_list()
    return {k: i for i, k in enumerate(keys)}


def edges_to_indexed(
    edges_df: pd.DataFrame,
    node_to_idx: dict[int, int],
    src_col="src",
    dst_col="dst",
    w_col=None,
) -> tuple[np.ndarray, np.ndarray]:
    src = edges_df[src_col].astype(int).map(node_to_idx)
    dst = edges_df[dst_col].astype(int).map(node_to_idx)
    ok = src.notna() & dst.notna()
    src = src[ok].astype(int).to_numpy()
    dst = dst[ok].astype(int).to_numpy()

    if w_col is None:
        w = np.ones(len(src), dtype=np.float32)
    else:
        w = pd.to_numeric(edges_df.loc[ok, w_col], errors="coerce").fillna(0).to_numpy(np.float32)
    return np.stack([src, dst], axis=0), w


def build_norm_adj_sparse(
    n_nodes: int,
    edge_index_undirected: np.ndarray,  # [2,E] (i,j) already includes both directions OR will be symmetrized here
    edge_weight: np.ndarray,            # [E]
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Build normalized sparse adjacency: D^{-1/2} A D^{-1/2} (GCN norm).
    """
    i = edge_index_undirected[0].astype(np.int64)
    j = edge_index_undirected[1].astype(np.int64)
    w = edge_weight.astype(np.float32)

    # symmetrize (add reverse)
    ii = np.concatenate([i, j])
    jj = np.concatenate([j, i])
    ww = np.concatenate([w, w])

    if add_self_loops:
        self_i = np.arange(n_nodes, dtype=np.int64)
        ii = np.concatenate([ii, self_i])
        jj = np.concatenate([jj, self_i])
        ww = np.concatenate([ww, np.ones(n_nodes, dtype=np.float32)])

    # degree
    deg = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(deg, ii, ww)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))

    # normalized weights
    ww_norm = ww * inv_sqrt[ii] * inv_sqrt[jj]

    idx = torch.tensor(np.stack([ii, jj], axis=0), dtype=torch.long)
    val = torch.tensor(ww_norm, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()
    return A