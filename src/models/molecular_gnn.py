"""Molecular substructure GNN for drug embeddings (Module 4.2.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class MolecularSubstructureGNN(nn.Module):
    """
    Message-passing encoder for drug molecular graphs.

    When per-drug atom graphs are unavailable, each drug is treated as a
    single-node graph with a learnable Morgan-fingerprint-style initializer.
    Output embeddings are concatenated with DDI-GAT embeddings in the dual-graph module.
    """

    def __init__(
        self,
        num_drugs: int,
        d_out: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_drugs = num_drugs
        self.d_out = d_out
        self.atom_init = nn.Embedding(num_drugs, d_out)
        nn.init.normal_(self.atom_init.weight, std=0.02)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(d_out, d_out))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, drug_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            drug_indices: optional [num_drugs] index tensor; defaults to all drugs.

        Returns:
            molecular_embeddings: [num_drugs, d_out]
        """
        n = self.num_drugs if drug_indices is None else drug_indices.numel()
        if drug_indices is None:
            drug_indices = torch.arange(n, device=self.atom_init.weight.device)

        x = self.atom_init(drug_indices)
        # Self-loop graph: each drug is an isolated node when SMILES graphs are absent.
        edge_index = torch.stack(
            [drug_indices, drug_indices],
            dim=0,
        )

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        return x
