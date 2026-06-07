"""Multilabel medication recommendation metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
)


def _to_numpy(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()


def jaccard_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Jaccard over samples."""
    return float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))


def f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def prauc_multilabel(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-label AP (macro) — standard for multilabel when labels vary."""
    n_labels = y_true.shape[1]
    aps = []
    for j in range(n_labels):
        if y_true[:, j].sum() == 0:
            continue
        try:
            aps.append(average_precision_score(y_true[:, j], y_prob[:, j]))
        except ValueError:
            continue
    return float(np.mean(aps)) if aps else 0.0


def ddi_rate(
    y_pred: np.ndarray,
    adj_upper: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of predicted drug pairs with DDI edges (upper triangle only).

    Uses binary predictions above threshold.
    """
    binary = (y_pred >= threshold).astype(np.float32)
    batch_rates = []
    for b in range(binary.shape[0]):
        idx = np.where(binary[b] > 0)[0]
        if len(idx) < 2:
            batch_rates.append(0.0)
            continue
        pairs = 0
        hits = 0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, c = idx[i], idx[j]
                if a > c:
                    a, c = c, a
                pairs += 1
                if adj_upper[a, c] > 0 or adj_upper[c, a] > 0:
                    hits += 1
        batch_rates.append(hits / pairs if pairs > 0 else 0.0)
    return float(np.mean(batch_rates))


def compute_all_metrics(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float = 0.5,
    adj_upper: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute Jaccard, F1, PRAUC, and DDI rate."""
    yt, yp_prob = _to_numpy(y_true, y_prob)
    yp_bin = (yp_prob >= threshold).astype(np.int32)
    yt_bin = yt.astype(np.int32)

    metrics = {
        "jaccard": jaccard_multilabel(yt_bin, yp_bin),
        "f1_micro": f1_micro(yt_bin, yp_bin),
        "f1_macro": f1_macro(yt_bin, yp_bin),
        "prauc": prauc_multilabel(yt, yp_prob),
    }
    if adj_upper is not None:
        adj_np = adj_upper.detach().cpu().numpy()
        metrics["ddi_rate"] = ddi_rate(yp_prob, adj_np, threshold)
    else:
        metrics["ddi_rate"] = 0.0
    return metrics
