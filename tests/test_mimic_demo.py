"""Test MIMIC-III demo preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mimic_demo import build_mimic_demo_patients, download_mimic_demo


def test_mimic_demo_builds_patients():
    raw = ROOT / "data" / "raw" / "mimiciii-demo"
    download_mimic_demo(raw)
    patients, meta = build_mimic_demo_patients(raw, top_diag=50, top_med=80, top_lab_items=10)
    assert len(patients) > 0
    assert meta["source"] == "mimiciii-demo-1.4"
    p = patients[0]
    assert p["diagnoses"].ndim == 2
    assert p["target"].ndim == 1
    assert p["target"].sum() > 0
