"""Evaluate a saved checkpoint on the test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import collate_patient_batch
from src.metrics.metrics import compute_all_metrics
from src.models.transformer_gat_model import TransformerGATMedRec
from src.data.load_data import apply_data_dims_to_config
from src.train import build_dataloaders, build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    use_synthetic: bool | None = False,
) -> dict:
    """Load checkpoint and report test metrics."""
    config = load_config(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    logger = get_logger("evaluate")

    _, _, test_loader, edge_index, edge_weight, adj_upper, data_meta = build_dataloaders(config, use_synthetic)
    if data_meta.get("source") == "mimic_demo":
        config = apply_data_dims_to_config(config, data_meta)
    adj_upper = adj_upper.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = build_model(config).to(device)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    load_checkpoint(ckpt_path, model, device=device)

    model.eval()
    all_probs, all_targets = [], []
    threshold = config["training"].get("threshold", 0.5)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate"):
            out = model(
                batch["diagnoses"].to(device),
                batch["medications"].to(device),
                batch["labs"].to(device),
                batch["visit_mask"].to(device),
                edge_index,
                edge_weight,
            )
            all_probs.append(out["probs"])
            all_targets.append(batch["target"].to(device))

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(targets, probs, threshold, adj_upper)

    logger.info("Test metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--use_synthetic", type=str, default="false")
    args = parser.parse_args()
    use_syn = args.use_synthetic.lower() in ("true", "1", "yes")
    evaluate_checkpoint(args.checkpoint, args.config, use_syn)


if __name__ == "__main__":
    main()
