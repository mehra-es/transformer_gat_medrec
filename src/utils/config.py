"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg: str) -> str:
    """Resolve device string from config."""
    import torch

    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg
