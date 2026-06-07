"""Test evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.metrics import compute_all_metrics


def test_metrics_keys():
    y_true = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
    y_prob = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]], dtype=torch.float32)
    adj = torch.triu(torch.ones(3, 3), 1) * 0.5
    m = compute_all_metrics(y_true, y_prob, threshold=0.5, adj_upper=adj)
    assert "jaccard" in m
    assert "f1_micro" in m
    assert "f1_macro" in m
    assert "prauc" in m
    assert "ddi_rate" in m
