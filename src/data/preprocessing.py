"""
Modular EHR preprocessing hooks.

TODO (MIMIC-III / MIMIC-IV):
  1. Map ICD diagnosis codes -> diagnosis vocabulary indices.
  2. Map NDC/RxNorm medication codes -> medication vocabulary indices.
  3. Normalize lab values per feature (z-score or min-max from train only).
  4. Group events into visits by admission_id / hadm_id.
  5. For each visit t, set target = medications at visit t+1 (or discharge list).
  6. Split patients by patient_id into train/val/test — never split visits of one patient.
  7. Build DDI edge list from external interaction DB aligned to med vocabulary.
  8. Serialize to data/processed/{train,val,test}.pkl and ddi_graph.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def build_vocab_from_codes(codes: List[str]) -> Dict[str, int]:
    """Build code -> index vocabulary (sorted for reproducibility)."""
    unique = sorted(set(codes))
    return {c: i for i, c in enumerate(unique)}


def visits_to_multihot(
    indices: List[int],
    vocab_size: int,
) -> np.ndarray:
    """Convert index list to multi-hot vector."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for idx in indices:
        if 0 <= idx < vocab_size:
            vec[idx] = 1.0
    return vec


def patient_record_template(
    patient_id: str,
    diagnoses: np.ndarray,
    medications: np.ndarray,
    labs: np.ndarray,
    target: np.ndarray,
) -> Dict[str, Any]:
    """Standard patient record dict for dataset consumption."""
    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses.astype(np.float32),
        "medications": medications.astype(np.float32),
        "labs": labs.astype(np.float32),
        "target": target.astype(np.float32),
    }


def save_processed(records: List[Dict[str, Any]], path: Path) -> None:
    """Save processed records (placeholder — extend with pickle/parquet)."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f)


def load_processed(path: Path) -> List[Dict[str, Any]]:
    """Load processed records."""
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)
