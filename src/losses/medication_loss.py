"""Multi-objective medication recommendation loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class MedicationLoss(nn.Module):
    """
    L_total = lambda_rec * L_rec + lambda_ddi * L_DDI

    L_DDI = mean_b sum_i (y_prob_b^T @ A_upper @ y_prob_b)
    using upper-triangular adjacency to avoid double counting.
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_ddi: float = 0.5,
        use_bce_logits: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ddi = lambda_ddi
        self.use_bce_logits = use_bce_logits
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce_prob = nn.BCELoss()
        self.register_buffer("ddi_adj_upper", torch.zeros(1, 1))

    def set_ddi_adjacency(self, adj_upper: torch.Tensor) -> None:
        """Set upper-triangular DDI severity matrix [num_med, num_med]."""
        self.register_buffer("ddi_adj_upper", adj_upper.float())

    def recommendation_loss(
        self,
        logits: torch.Tensor | None,
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_bce_logits and logits is not None:
            return self.bce_logits(logits, target)
        return self.bce_prob(probs, target)

    def ddi_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Severity-weighted DDI penalty.

        Args:
            probs: [B, num_med]
        """
        A = self.ddi_adj_upper  # [M, M]
        if A.numel() <= 1:
            return probs.new_tensor(0.0)
        # [B, M] @ [M, M] -> [B, M], elementwise * probs, sum over M
        quad = probs @ A @ probs.T  # wrong — need per sample
        # Per sample: y^T A y = (y.unsqueeze(0) @ A @ y.unsqueeze(1)).squeeze()
        scores = (probs @ A * probs).sum(dim=1)
        return scores.mean()

    def forward(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_rec = self.recommendation_loss(logits, probs, target)
        l_ddi = self.ddi_loss(probs)
        total = self.lambda_rec * l_rec + self.lambda_ddi * l_ddi
        return total, {"l_rec": l_rec.detach(), "l_ddi": l_ddi.detach(), "l_total": total.detach()}
