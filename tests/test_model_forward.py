"""Test model forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.transformer_gat_model import TransformerGATMedRec


def test_forward_shape():
    B, T = 4, 5
    num_diag, num_med, num_lab = 50, 100, 20
    d_model = 256

    model = TransformerGATMedRec(
        num_diag=num_diag,
        num_med=num_med,
        num_lab=num_lab,
        d_model=d_model,
    )
    diagnoses = torch.randn(B, T, num_diag).abs()
    medications = (torch.rand(B, T, num_med) > 0.7).float()
    labs = torch.randn(B, T, num_lab)
    visit_mask = torch.ones(B, T)
    visit_mask[:, -1] = 0  # pad last visit
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.8, 0.9])

    out = model(diagnoses, medications, labs, visit_mask, edge_index, edge_weight)
    assert out["probs"].shape == (B, num_med)
    assert out["logits"].shape == (B, num_med)
    assert out["h_patient"].shape == (B, d_model)
    assert out["h_drug"].shape == (B, d_model)
