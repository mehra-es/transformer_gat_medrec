"""Batch collation with padding for variable-length sequences."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_patient_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad patient visit sequences to max length in batch.

    visit_mask: 1 for real visits, 0 for padding.
    Transformer src_key_padding_mask uses True for PAD positions.
    """
    batch_size = len(batch)
    max_t = max(item["diagnoses"].shape[0] for item in batch)
    num_diag = batch[0]["diagnoses"].shape[1]
    num_med = batch[0]["medications"].shape[1]
    num_lab = batch[0]["labs"].shape[1]

    diagnoses = torch.zeros(batch_size, max_t, num_diag)
    medications = torch.zeros(batch_size, max_t, num_med)
    labs = torch.zeros(batch_size, max_t, num_lab)
    visit_mask = torch.zeros(batch_size, max_t)
    targets = torch.stack([item["target"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]

    for i, item in enumerate(batch):
        t = item["diagnoses"].shape[0]
        diagnoses[i, :t] = item["diagnoses"]
        medications[i, :t] = item["medications"]
        labs[i, :t] = item["labs"]
        visit_mask[i, :t] = item["visit_mask"]

    return {
        "diagnoses": diagnoses,
        "medications": medications,
        "labs": labs,
        "target": targets,
        "visit_mask": visit_mask,
        "patient_id": patient_ids,
    }
