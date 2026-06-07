"""Entry point for training, evaluation, and explanation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_checkpoint
from src.explain import explain_patient
from src.train import train_model
from src.utils.config import load_config


def parse_bool(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer-GAT medication recommendation (clinical decision support only)."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "explain"], default="train")
    parser.add_argument("--use_synthetic", type=str, default="true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    use_syn = parse_bool(args.use_synthetic)

    if args.mode == "train":
        train_model(config, use_synthetic=use_syn)
    elif args.mode == "eval":
        evaluate_checkpoint(args.checkpoint, str(config_path), use_syn)
    elif args.mode == "explain":
        result = explain_patient(config, args.checkpoint, use_syn)
        import json
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
