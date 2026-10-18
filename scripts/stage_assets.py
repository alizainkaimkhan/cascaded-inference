#!/usr/bin/env python
"""Run ONCE on the LARCC login node (compute nodes have no internet).

Downloads the datasets and the Qwen3-8B snapshot at a pinned commit, writes the datasets
to data/raw/*.parquet, and records every revision in configs/pinned.json, which
load_config() overlays on configs/default.yaml.

    export HF_HOME=$HOME/hf_cache
    python scripts/stage_assets.py               # datasets + model
    python scripts/stage_assets.py --skip-model  # datasets only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import PINNED_PATH, load_config, repo_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--model-revision", default=None, help="pin an explicit commit instead of main's head")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    cfg = load_config()
    api = HfApi()
    raw = repo_path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(PINNED_PATH.read_text()) if PINNED_PATH.exists() else {}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # --- datasets -----------------------------------------------------------
    mm_id = cfg["datasets"]["medmcqa"]["hf_id"]
    mm_rev = api.dataset_info(mm_id).sha
    for split in ("train", "validation"):
        ds = load_dataset(mm_id, split=split, revision=mm_rev)
        ds.to_parquet(str(raw / f"medmcqa_{split}.parquet"))
        print(f"medmcqa/{split}: {len(ds)} rows")

    mq_id = cfg["datasets"]["medqa"]["hf_id"]
    mq_rev = api.dataset_info(mq_id).sha
    ds = load_dataset(mq_id, split="test", revision=mq_rev)
    ds.to_parquet(str(raw / "medqa_test.parquet"))
    print(f"medqa/test: {len(ds)} rows")

    pinned.setdefault("datasets", {})
    pinned["datasets"]["medmcqa"] = {"hf_id": mm_id, "revision": mm_rev, "staged_at": now}
    pinned["datasets"]["medqa"] = {"hf_id": mq_id, "revision": mq_rev, "staged_at": now}

    # --- model --------------------------------------------------------------
    if not args.skip_model:
        model_id = cfg["small_model"]["hf_id"]
        rev = args.model_revision or api.model_info(model_id).sha
        path = snapshot_download(model_id, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        pinned["small_model"] = {"hf_id": model_id, "revision": rev, "local_path": path, "staged_at": now}
        print(f"{model_id}@{rev} -> {path}")

    PINNED_PATH.write_text(json.dumps(pinned, indent=2) + "\n")
    print(f"wrote {PINNED_PATH}")


if __name__ == "__main__":
    main()
