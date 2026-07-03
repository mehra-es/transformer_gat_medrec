"""Multi-objective medication recommendation loss (Module 5)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MedicationLoss(nn.Module):
    """
    L_total = λ₁·L_rec + λ₂·L_DDI + λ₃·L_outcome + λ₄·L_xai

    L_DDI = mean_b Σ_{(u,v)∈E} w_uv · ŷ_u · ŷ_v  (severity-weighted co-recommendation penalty)

    L_outcome and L_xai are architectural placeholders until Module 7 and Integrated
    Gradients are wired; they evaluate to zero in the current experimental setup.
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_ddi: float = 0.5,
        lambda_outcome: float = 0.3,
        lambda_xai: float = 0.1,
        use_bce_logits: bool = True,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ddi = lambda_ddi
        self.lambda_outcome = lambda_outcome
        self.lambda_xai = lambda_xai
        self.use_bce_logits = use_bce_logits
        self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_prob = nn.BCELoss()
        self.register_buffer("ddi_adj_upper", torch.zeros(1, 1))
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(1))

    def set_ddi_adjacency(self, adj_upper: torch.Tensor) -> None:
        """Set upper-triangular DDI severity matrix [num_med, num_med]."""
        self.register_buffer("ddi_adj_upper", adj_upper.float())

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """Per-drug positive class weights for imbalanced multi-label BCE."""
        self.register_buffer("pos_weight", pos_weight.float())
        self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

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
        Severity-weighted DDI penalty: Σ w_uv · ŷ_u · ŷ_v per sample.

        Args:
            probs: [B, num_med]
        """
        A = self.ddi_adj_upper
        if A.numel() <= 1:
            return probs.new_tensor(0.0)
        scores = (probs @ A * probs).sum(dim=1)
        return scores.mean()

    def outcome_loss(self, outcome_pred: torch.Tensor | None, outcome_target: torch.Tensor | None) -> torch.Tensor:
        """Module 7 placeholder — returns zero until TCN outcome head is connected."""
        if outcome_pred is None or outcome_target is None:
            return torch.tensor(0.0, device=self.ddi_adj_upper.device)
        return torch.tensor(0.0, device=outcome_pred.device)

    def xai_fidelity_loss(
        self,
        shap_attr: torch.Tensor | None = None,
        ig_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L_xai = ||φ_SHAP − φ_IG||² — zero until Integrated Gradients is implemented."""
        if shap_attr is None or ig_attr is None:
            return torch.tensor(0.0, device=self.ddi_adj_upper.device)
        return torch.tensor(0.0, device=shap_attr.device)

    def forward(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor | None = None,
        outcome_pred: torch.Tensor | None = None,
        outcome_target: torch.Tensor | None = None,
        shap_attr: torch.Tensor | None = None,
        ig_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_rec = self.recommendation_loss(logits, probs, target)
        l_ddi = self.ddi_loss(probs)
        l_outcome = self.outcome_loss(outcome_pred, outcome_target)
        l_xai = self.xai_fidelity_loss(shap_attr, ig_attr)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_ddi * l_ddi
            + self.lambda_outcome * l_outcome
            + self.lambda_xai * l_xai
        )
        return total, {
            "l_rec": l_rec.detach(),
            "l_ddi": l_ddi.detach(),
            "l_outcome": l_outcome.detach(),
            "l_xai": l_xai.detach(),
            "l_total": total.detach(),
        }
