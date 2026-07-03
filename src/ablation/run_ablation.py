"""Run ablation study variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_model
from src.utils.config import load_config
from src.evaluate import evaluate_checkpoint


ABLATIONS = {
    "full": {},
    "no_transformer": {"use_transformer": False},
    "no_gat": {"use_gat": False},
    "no_molecular_gnn": {"use_molecular_gnn": False},
    "no_gated_fusion": {"use_gated_fusion": False},
    "no_ddi_loss": {"use_ddi_loss": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--use_synthetic", type=str, default="true")
    parser.add_argument("--variant", type=str, default="all", choices=list(ABLATIONS) + ["all"])
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    use_syn = args.use_synthetic.lower() in ("true", "1", "yes")
    variants = ABLATIONS if args.variant == "all" else {args.variant: ABLATIONS[args.variant]}

    results = {}
    for name, ab in variants.items():
        ckpt = ROOT / "checkpoints" / f"ablation_{name}.pt"
        config["paths"]["best_checkpoint"] = str(ckpt)
        train_model(config, use_synthetic=use_syn, ablation=ab)
        metrics = evaluate_checkpoint(str(ckpt), args.config, use_syn)
        results[name] = metrics

    out = ROOT / "logs" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
