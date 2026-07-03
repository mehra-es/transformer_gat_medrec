"""
TCN-based longitudinal outcome monitoring (Module 7).

Architectural specification only — not wired into training in the present study.
Outcome losses (L_outcome) evaluate to zero until this module is connected.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OutcomeMonitoringTCN(nn.Module):
    """Predicts 30-day readmission, ADE categories, and treatment effectiveness."""

    def __init__(self, d_model: int, num_ade_categories: int = 5) -> None:
        super().__init__()
        self.readmission_head = nn.Linear(d_model, 1)
        self.ade_head = nn.Linear(d_model, num_ade_categories)
        self.effectiveness_head = nn.Linear(d_model, 1)

    def forward(self, h_fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "readmission": torch.sigmoid(self.readmission_head(h_fused)),
            "ade": torch.sigmoid(self.ade_head(h_fused)),
            "effectiveness": self.effectiveness_head(h_fused),
        }
