"""Loads model/training YAML configs into a simple attribute-access object."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**raw)
