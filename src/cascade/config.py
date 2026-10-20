"""Load configs/default.yaml, overlaid with configs/pinned.json (shared revisions) and then
configs/local.json (this machine only), when they exist."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
PINNED_PATH = REPO_ROOT / "configs" / "pinned.json"   # shared, committed: exact revisions
LOCAL_PATH = REPO_ROOT / "configs" / "local.json"     # per-machine, git-ignored: e.g. model snapshot path


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path or CONFIG_PATH).read_text())
    for overlay in (PINNED_PATH, LOCAL_PATH):
        if overlay.exists():
            _deep_update(cfg, json.loads(overlay.read_text()))
    return cfg


def repo_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return None
