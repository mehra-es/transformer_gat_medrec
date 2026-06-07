#!/usr/bin/env python3
"""Download and preprocess MIMIC-III demo from PhysioNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mimic_demo import preprocess_and_save
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-III demo (PhysioNet 1.4)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    data_cfg = config["data"]
    mimic_cfg = data_cfg.get("mimic_demo", {})

    meta = preprocess_and_save(
        raw_dir=ROOT / mimic_cfg.get("raw_dir", "data/raw/mimiciii-demo"),
        processed_dir=ROOT / mimic_cfg.get("processed_dir", "data/processed/mimic_demo"),
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        top_diag=mimic_cfg.get("top_diag", 200),
        top_med=mimic_cfg.get("top_med", 300),
        top_lab_items=mimic_cfg.get("top_lab_items", 20),
        seed=config.get("seed", 42),
    )
    print("MIMIC-III demo preprocessing complete:")
    for k, v in meta.items():
        if k not in ("diag_vocab", "med_vocab"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
