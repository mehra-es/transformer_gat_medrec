"""Pareto sweep over DDI penalty weight λ₂ (paper Table 14)."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_checkpoint
from src.metrics.metrics import safety_adjusted_effectiveness
from src.train import train_model
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="λ_ddi safety-accuracy Pareto sweep")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--use_synthetic", type=str, default="false")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
    )
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    use_syn = args.use_synthetic.lower() in ("true", "1", "yes")

    results = []
    for lam in args.lambdas:
        cfg = deepcopy(config)
        cfg["loss"]["lambda_ddi"] = lam
        ckpt = ROOT / "checkpoints" / f"pareto_lambda_{lam:.1f}.pt"
        cfg["paths"]["best_checkpoint"] = str(ckpt)
        print(f"\n=== λ_ddi = {lam} ===")
        train_model(cfg, use_synthetic=use_syn)
        metrics = evaluate_checkpoint(str(ckpt), args.config, use_syn)
        row = {
            "lambda_ddi": lam,
            "jaccard": metrics["jaccard"],
            "ddi_rate": metrics["ddi_rate"],
            "f1_micro": metrics["f1_micro"],
            "prauc": metrics["prauc"],
            "sae": safety_adjusted_effectiveness(metrics["jaccard"], metrics["ddi_rate"]),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pareto_lambda_ddi.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
