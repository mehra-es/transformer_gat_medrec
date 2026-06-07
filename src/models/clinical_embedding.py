"""Clinical embedding layer for diagnoses, medications, and labs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalEmbedding(nn.Module):
    """
    Embed diagnosis, medication, and lab features per visit.

    Input shapes:
        diagnoses: [B, T, num_diag]
        medications: [B, T, num_med]
        labs: [B, T, num_lab]

    Output:
        visit_embeddings: [B, T, d_model]
    """

    def __init__(
        self,
        num_diag: int,
        num_med: int,
        num_lab: int,
        d_model: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        third = d_model // 3
        rem = d_model - 2 * third
        self.diag_proj = nn.Linear(num_diag, third)
        self.med_proj = nn.Linear(num_med, third)
        self.lab_proj = nn.Linear(num_lab, rem)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        diagnoses: torch.Tensor,
        medications: torch.Tensor,
        labs: torch.Tensor,
    ) -> torch.Tensor:
        d = self.diag_proj(diagnoses)
        m = self.med_proj(medications)
        l = self.lab_proj(labs)
        x = torch.cat([d, m, l], dim=-1)
        x = self.out_proj(x)
        x = self.act(x)
        return self.dropout(x)
