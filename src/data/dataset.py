"""Patient visit dataset and synthetic data generator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_synthetic_patients(
    num_patients: int,
    num_diag: int,
    num_med: int,
    num_lab: int,
    min_visits: int,
    max_visits: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic longitudinal EHR patients for demo runs."""
    rng = np.random.default_rng(seed)
    patients: List[Dict[str, Any]] = []

    for p in range(num_patients):
        T = int(rng.integers(min_visits, max_visits + 1))
        diagnoses = np.zeros((T, num_diag), dtype=np.float32)
        medications = np.zeros((T, num_med), dtype=np.float32)
        labs = np.zeros((T, num_lab), dtype=np.float32)

        for t in range(T):
            nd = int(rng.integers(1, min(6, num_diag)))
            nm = int(rng.integers(0, min(5, num_med)))
            diagnoses[t, rng.choice(num_diag, size=nd, replace=False)] = 1.0
            if nm > 0:
                medications[t, rng.choice(num_med, size=nm, replace=False)] = 1.0
            labs[t] = rng.normal(0, 1, size=num_lab).astype(np.float32)

        # Target: medications for next visit pattern (last visit uses random subset)
        target = np.zeros(num_med, dtype=np.float32)
        n_tgt = int(rng.integers(1, min(6, num_med)))
        target[rng.choice(num_med, size=n_tgt, replace=False)] = 1.0

        patients.append(
            {
                "patient_id": f"synth_{p:05d}",
                "diagnoses": diagnoses,
                "medications": medications,
                "labs": labs,
                "target": target,
            }
        )
    return patients


def split_patients_by_id(
    patients: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split by patient_id — no visit leakage across splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ids = [p["patient_id"] for p in patients]
    order = rng.permutation(len(ids))
    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = set(order[:n_train])
    val_idx = set(order[n_train : n_train + n_val])
    test_idx = set(order[n_train + n_val :])

    train_p, val_p, test_p = [], [], []
    for i, p in enumerate(patients):
        if i in train_idx:
            train_p.append(p)
        elif i in val_idx:
            val_p.append(p)
        else:
            test_p.append(p)
    return train_p, val_p, test_p


class MedicationRecommendationDataset(Dataset):
    """Dataset returning per-patient visit sequences."""

    def __init__(self, patients: List[Dict[str, Any]]) -> None:
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.patients[idx]
        T = p["diagnoses"].shape[0]
        return {
            "diagnoses": torch.from_numpy(p["diagnoses"]),
            "medications": torch.from_numpy(p["medications"]),
            "labs": torch.from_numpy(p["labs"]),
            "target": torch.from_numpy(p["target"]),
            "visit_mask": torch.ones(T, dtype=torch.float32),
            "patient_id": p["patient_id"],
        }
