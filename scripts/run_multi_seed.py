"""Run four-seed statistical validation (paper Table 11)."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_checkpoint
from src.metrics.metrics import aggregate_seed_metrics
from src.train import train_model
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training and evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--use_synthetic", type=str, default="false")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    use_syn = args.use_synthetic.lower() in ("true", "1", "yes")
    seeds = args.seeds or config.get("training", {}).get("seeds", [42, 1, 123, 2024])

    per_seed: dict[str, dict] = {}
    metrics_list = []

    for seed in seeds:
        cfg = deepcopy(config)
        cfg["seed"] = seed
        ckpt = ROOT / "checkpoints" / f"seed_{seed}.pt"
        cfg["paths"]["best_checkpoint"] = str(ckpt)
        print(f"\n=== Seed {seed} ===")
        train_model(cfg, use_synthetic=use_syn)
        metrics = evaluate_checkpoint(str(ckpt), args.config, use_syn)
        per_seed[str(seed)] = metrics
        metrics_list.append(metrics)

    summary = aggregate_seed_metrics(metrics_list)
    results = {"per_seed": per_seed, "aggregate": summary}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_seed_validation.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
