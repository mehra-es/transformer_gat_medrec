"""Unified data loading for synthetic and MIMIC-III demo sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_patient_batch
from src.data.dataset import MedicationRecommendationDataset, generate_synthetic_patients, split_patients_by_id
from src.data.ddi_graph import build_random_ddi_graph, ddi_adjacency_upper
from src.data.mimic_demo import (
    load_mimic_demo_ddi,
    load_mimic_demo_splits,
    preprocess_and_save,
)


def _resolve_data_source(config: Dict[str, Any], use_synthetic: bool | None) -> str:
    if use_synthetic is True:
        return "synthetic"
    if use_synthetic is False:
        return config.get("data", {}).get("source", "mimic_demo")
    return config.get("data", {}).get("source", "mimic_demo")


def ensure_mimic_processed(config: Dict[str, Any], root: Path) -> Path:
    """Run MIMIC-III demo preprocessing if artifacts are missing."""
    data_cfg = config["data"]
    mimic_cfg = data_cfg.get("mimic_demo", {})
    processed_dir = root / mimic_cfg.get("processed_dir", "data/processed/mimic_demo")
    if (processed_dir / "meta.json").exists():
        return processed_dir

    raw_dir = root / mimic_cfg.get("raw_dir", "data/raw/mimiciii-demo")
    preprocess_and_save(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        top_diag=mimic_cfg.get("top_diag", 200),
        top_med=mimic_cfg.get("top_med", 300),
        top_lab_items=mimic_cfg.get("top_lab_items", 20),
        seed=config.get("seed", 42),
    )
    return processed_dir


def apply_data_dims_to_config(config: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Update config data dimensions from processed metadata."""
    config = dict(config)
    data_cfg = dict(config["data"])
    data_cfg["num_diag"] = meta["num_diag"]
    data_cfg["num_med"] = meta["num_med"]
    data_cfg["num_lab"] = meta["num_lab"]
    config["data"] = data_cfg
    return config


def build_dataloaders(
    config: Dict[str, Any],
    root: Path,
    use_synthetic: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build loaders and DDI graph. Returns extra metadata dict."""
    data_cfg = config["data"]
    seed = config.get("seed", 42)
    source = _resolve_data_source(config, use_synthetic)
    meta: Dict[str, Any] = {"source": source}

    if source == "synthetic":
        syn = data_cfg["synthetic"]
        patients = generate_synthetic_patients(
            num_patients=syn["num_patients"],
            num_diag=data_cfg["num_diag"],
            num_med=data_cfg["num_med"],
            num_lab=data_cfg["num_lab"],
            min_visits=syn["min_visits"],
            max_visits=syn["max_visits"],
            seed=seed,
        )
        train_p, val_p, test_p = split_patients_by_id(
            patients,
            data_cfg["train_ratio"],
            data_cfg["val_ratio"],
            data_cfg["test_ratio"],
            seed=seed,
        )
        edge_index, edge_weight, adj_upper = build_random_ddi_graph(
            data_cfg["num_med"],
            data_cfg["synthetic"]["ddi_density"],
            seed=seed,
        )
        adj_upper = ddi_adjacency_upper(data_cfg["num_med"], edge_index, edge_weight)
    elif source == "mimic_demo":
        processed_dir = ensure_mimic_processed(config, root)
        train_p, val_p, test_p, mimic_meta = load_mimic_demo_splits(processed_dir)
        edge_index, edge_weight, adj_upper = load_mimic_demo_ddi(processed_dir)
        meta.update(mimic_meta)
        meta["source"] = "mimic_demo"
    else:
        raise ValueError(f"Unknown data source: {source}")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(
        MedicationRecommendationDataset(train_p),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        MedicationRecommendationDataset(val_p),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        MedicationRecommendationDataset(test_p),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, edge_index, edge_weight, adj_upper, meta
