"""Medication recommendation head."""

from __future__ import annotations

import torch
import torch.nn as nn


class RecommendationHead(nn.Module):
    """Linear projection to medication probabilities."""

    def __init__(self, d_model: int, num_med: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, num_med)

    def forward(self, h_fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, num_med]
            probs: [B, num_med] after sigmoid
        """
        logits = self.fc(h_fused)
        probs = torch.sigmoid(logits)
        return logits, probs
