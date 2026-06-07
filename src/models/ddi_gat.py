"""DDI-aware Graph Attention Network for drug embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class DDIAwareGAT(nn.Module):
    """
    Multi-layer GAT over drug nodes.

    Input:
        drug_features: [num_med, d_model] initial node features
        edge_index: [2, num_edges]
        edge_weight: optional [num_edges] — used as edge attr when supported

    Output:
        drug_embeddings: [num_med, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        in_ch = d_model
        for i in range(num_layers):
            out_per_head = d_model // heads
            out_ch = out_per_head * heads
            if out_ch != d_model:
                out_ch = d_model
            conv = GATConv(
                in_channels=in_ch,
                out_channels=d_model // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=1 if i == 0 else None,
            )
            self.layers.append(conv)
            in_ch = d_model
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        drug_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = drug_features
        ew = edge_weight
        if edge_index.numel() == 0:
            return self.dropout(self.act(x))

        for i, conv in enumerate(self.layers):
            if i == 0 and ew is not None and ew.numel() > 0:
                edge_attr = ew.view(-1, 1).float()
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        return x


def aggregate_patient_drug_embedding(
    drug_embeddings: torch.Tensor,
    latest_medications: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted average of GAT drug embeddings using latest medication vector.

    Args:
        drug_embeddings: [num_med, d_model]
        latest_medications: [B, num_med]

    Returns:
        h_drug: [B, d_model]
    """
    # [B, num_med] @ [num_med, d_model] -> [B, d_model]
    weights = latest_medications.float()
    sums = weights.sum(dim=1, keepdim=True)
    has_meds = sums > 0
    normalized = weights / sums.clamp(min=1e-8)
    h = normalized @ drug_embeddings
    mean_emb = drug_embeddings.mean(dim=0, keepdim=True).expand(h.size(0), -1)
    return torch.where(has_meds, h, mean_emb)
