"""Config loading. Paths in the YAML are resolved relative to the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # <root>/src/classifier/config.py


def load_config(path: str | Path = "configs/classifier.yaml", root: str | Path | None = None) -> dict:
    root = Path(root).resolve() if root else REPO_ROOT
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root)
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
