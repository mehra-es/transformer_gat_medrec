"""Learnable gated fusion of patient and drug representations."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    gate = sigmoid(W_g(concat(h_patient, h_drug)))
    h_fused = gate * h_patient + (1 - gate) * h_drug
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_layer = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        h_patient: torch.Tensor,
        h_drug: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_fused: [B, d_model]
            gate: [B, d_model]
        """
        concat = torch.cat([h_patient, h_drug], dim=-1)
        gate = torch.sigmoid(self.gate_layer(concat))
        h_fused = gate * h_patient + (1.0 - gate) * h_drug
        return h_fused, gate
