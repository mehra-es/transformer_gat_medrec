"""Transformer temporal encoder with last-valid-visit pooling."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerTemporalEncoder(nn.Module):
    """
    Encode visit sequences with nn.TransformerEncoder.

    Uses src_key_padding_mask from visit_mask (True = pad).
    Returns h_patient from the last *valid* visit, not the final padded token.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        visit_embeddings: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visit_embeddings: [B, T, d_model]
            visit_mask: [B, T] with 1 for valid, 0 for pad

        Returns:
            h_patient: [B, d_model]
        """
        # PyTorch: True marks positions to IGNORE
        pad_mask = visit_mask == 0
        encoded = self.encoder(visit_embeddings, src_key_padding_mask=pad_mask)

        lengths = visit_mask.sum(dim=1).long().clamp(min=1)  # [B]
        last_idx = lengths - 1  # [B]
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        h_patient = encoded[batch_idx, last_idx, :]
        return h_patient
