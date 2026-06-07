"""Test medication loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.medication_loss import MedicationLoss


def test_loss_scalar():
    B, M = 8, 100
    criterion = MedicationLoss()
    adj = torch.triu(torch.rand(M, M), diagonal=1)
    criterion.set_ddi_adjacency(adj)
    logits = torch.randn(B, M)
    probs = torch.sigmoid(logits)
    target = (torch.rand(B, M) > 0.8).float()
    loss, parts = criterion(probs, target, logits)
    assert loss.dim() == 0
    assert parts["l_rec"].numel() == 1


def test_ddi_loss_increases_with_interactions():
    M = 20
    criterion = MedicationLoss(lambda_rec=0.0, lambda_ddi=1.0, use_bce_logits=False)
    adj = torch.zeros(M, M)
    adj[0, 1] = 1.0
    adj_upper = torch.triu(adj + adj.T, diagonal=1)
    criterion.set_ddi_adjacency(adj_upper)

    probs_safe = torch.zeros(1, M)
    probs_safe[0, 0] = 0.9

    probs_risk = torch.zeros(1, M)
    probs_risk[0, 0] = 0.9
    probs_risk[0, 1] = 0.9

    target = torch.zeros(1, M)
    l_safe, _ = criterion(probs_safe, target)
    l_risk, _ = criterion(probs_risk, target)
    assert l_risk.item() > l_safe.item()
