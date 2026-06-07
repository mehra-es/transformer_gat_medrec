"""DDI graph construction for GAT and loss."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def build_random_ddi_graph(
    num_med: int,
    density: float,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build synthetic DDI graph for demo.

    Returns:
        edge_index: [2, E]
        edge_weight: [E]
        adj_upper: [num_med, num_med] upper-triangular severity matrix
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((num_med, num_med), dtype=np.float32)
    for i in range(num_med):
        for j in range(i + 1, num_med):
            if rng.random() < density:
                w = float(rng.uniform(0.5, 1.0))
                adj[i, j] = w
                adj[j, i] = w

    rows, cols = np.where(adj > 0)
    if len(rows) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float32)

    adj_upper = torch.tensor(np.triu(adj, k=1), dtype=torch.float32)
    return edge_index, edge_weight, adj_upper


def ddi_adjacency_upper(num_med: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Build upper-triangular DDI adjacency from edges (for loss, no double count)."""
    adj = torch.zeros(num_med, num_med, dtype=torch.float32)
    if edge_index.numel() > 0:
        i, j = edge_index[0], edge_index[1]
        w = edge_weight.float()
        adj[i, j] = w
        adj[j, i] = w
    return torch.triu(adj, diagonal=1)
