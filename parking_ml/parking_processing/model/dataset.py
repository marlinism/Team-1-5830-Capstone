from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PanelStore:
    """Caches loaded panel.npz files in memory to avoid repeated disk IO."""
    def __init__(self):
        self.cache = {}

    def load(self, path: str):
        if path not in self.cache:
            self.cache[path] = np.load(path, allow_pickle=False)
        return self.cache[path]


class TimeWindowDataset(Dataset):
    """
    Each item = one time index t (predict for ALL nodes at that t).
    Inputs: X[t-Tin:t], targets y15[t], y30[t] across nodes (with masks).
    """
    def __init__(self, panel_paths: list[str], tin: int = 12, horizons=(1, 2)):
        self.panel_paths = panel_paths
        self.tin = tin
        self.h1, self.h2 = horizons
        self.store = PanelStore()

        # Build index map: global idx -> (panel_path, t)
        self.index = []
        for p in self.panel_paths:
            z = self.store.load(p)
            T = z["X"].shape[0]
            # need t so that y_15/y_30 exist at t (they were precomputed); safe window:
            t_min = self.tin
            t_max = T - 1
            for t in range(t_min, t_max):
                self.index.append((p, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        p, t = self.index[idx]
        z = self.store.load(p)

        X = z["X"][t - self.tin : t]        # [Tin,N,F]
        y15 = z["y15"][t]                   # [N]
        y30 = z["y30"][t]                   # [N]
        mask = z["mask"][t]                 # [N] observation exists

        # label masks: y == -1 => missing
        m15 = (y15 >= 0) & (mask > 0)
        m30 = (y30 >= 0) & (mask > 0)

        # tensors
        X = torch.tensor(X, dtype=torch.float32)
        y15 = torch.tensor(y15, dtype=torch.float32)
        y30 = torch.tensor(y30, dtype=torch.float32)
        m15 = torch.tensor(m15.astype(np.float32), dtype=torch.float32)
        m30 = torch.tensor(m30.astype(np.float32), dtype=torch.float32)

        return X, y15, y30, m15, m30