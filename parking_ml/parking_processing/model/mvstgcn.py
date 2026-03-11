from __future__ import annotations
import torch
import torch.nn as nn


def sparse_batch_mm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    A: sparse [N,N]
    X: dense  [B,N,C]
    returns:  [B,N,C]
    """
    B, N, C = X.shape
    XT = X.transpose(0, 1).reshape(N, B * C)        # [N, B*C]
    YT = torch.sparse.mm(A, XT)                     # [N, B*C]
    Y = YT.reshape(N, B, C).transpose(0, 1)         # [B,N,C]
    return Y


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # message passing then linear
        Z = sparse_batch_mm(A, H)
        Z = self.lin(Z)
        Z = self.act(Z)
        return self.dropout(Z)


class MVSTGCN(nn.Module):
    """
    Minimal MV-STGCN-like block:
      - temporal Conv1d over time for each node
      - graph conv on topology + feature graphs (fused)
      - two prediction heads for y15 / y30
    """
    def __init__(
        self,
        n_nodes: int,
        n_feats: int,
        hidden: int = 64,
        temporal_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Temporal: apply Conv1d over time for each node independently
        # Input: [B,T,N,F] -> reshape to [B*N, F, T] for Conv1d
        self.temp = nn.Sequential(
            nn.Conv1d(n_feats, temporal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # MV spatial fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable fusion weight
        self.gcn_topo = GraphConv(hidden, hidden, dropout=dropout)
        self.gcn_feat = GraphConv(hidden, hidden, dropout=dropout)

        # Heads: logits for has_space at +15 and +30
        self.head15 = nn.Linear(hidden, 1)
        self.head30 = nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor, A_topo: torch.Tensor, A_feat: torch.Tensor):
        """
        X: [B,T,N,F]
        Returns logits15, logits30: [B,N]
        """
        B, T, N, F = X.shape
        assert N == self.n_nodes

        x = X.permute(0, 2, 3, 1).contiguous()  # [B,N,F,T]
        x = x.view(B * N, F, T)                 # [B*N,F,T]
        h = self.temp(x)                        # [B*N,H,T]
        h_last = h[:, :, -1]                    # [B*N,H]
        H = h_last.view(B, N, -1)               # [B,N,H]

        topo_out = self.gcn_topo(A_topo, H)
        feat_out = self.gcn_feat(A_feat, H)

        a = torch.clamp(self.alpha, 0.0, 1.0)
        Z = a * topo_out + (1.0 - a) * feat_out  # [B,N,H]

        logits15 = self.head15(Z).squeeze(-1)    # [B,N]
        logits30 = self.head30(Z).squeeze(-1)    # [B,N]
        return logits15, logits30