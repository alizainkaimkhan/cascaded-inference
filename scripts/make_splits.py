#!/usr/bin/env python
"""Deduplicate and freeze all splits. Cheap (CPU, seconds); run on the login node.

Refuses to overwrite existing splits unless --force, because every downstream number
depends on them. Commit data/splits/ after the first run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import load_config, repo_path  # noqa: E402
from cascade.data import build_splits, load_raw, write_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = repo_path(cfg["paths"]["splits"])
    if (out / "manifest.json").exists() and not args.force:
        sys.exit(f"{out}/manifest.json exists; splits are frozen. Use --force only if you mean it.")

    splits, report = build_splits(load_raw(repo_path(cfg["paths"]["raw"])), cfg)
    write_splits(splits, report, out)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("subjects")}, indent=2, default=int))


if __name__ == "__main__":
    main()
