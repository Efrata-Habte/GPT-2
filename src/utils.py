"""Small helpers: seeding and config loading. (Plotting helpers arrive in Phase 4.)"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
