#!/usr/bin/env python
"""Run ONCE on the LARCC login node (compute nodes have no internet).

Downloads the datasets and the Qwen3-8B snapshot and writes the datasets to
data/raw/*.parquet.

Revisions: if configs/pinned.json already pins a revision (the normal case after the first
run, since pinned.json is committed), that exact revision is downloaded, so every
collaborator gets byte-identical data and weights. Only the very first run, or --repin,
resolves the current head and writes new pins.

The local model path differs per person, so it goes to configs/local.json (git-ignored).

    export HF_HOME=$HOME/hf_cache
    python scripts/stage_assets.py               # datasets + model at the pinned revisions
    python scripts/stage_assets.py --skip-model  # datasets only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import LOCAL_PATH, PINNED_PATH, load_config, repo_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--model-revision", default=None, help="pin an explicit model commit (implies re-pinning the model)")
    ap.add_argument("--repin", action="store_true", help="ignore existing pins and pin the current head revisions")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    cfg = load_config()
    api = HfApi()
    raw = repo_path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(PINNED_PATH.read_text()) if PINNED_PATH.exists() else {}
    local = json.loads(LOCAL_PATH.read_text()) if LOCAL_PATH.exists() else {}
    pinned_before = json.dumps(pinned, sort_keys=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def pinned_rev(section: dict | None, hf_id: str) -> str | None:
        if args.repin or not section or section.get("hf_id") != hf_id:
            return None
        return section.get("revision")

    # --- datasets -----------------------------------------------------------
    mm_id = cfg["datasets"]["medmcqa"]["hf_id"]
    mm_rev = pinned_rev(pinned.get("datasets", {}).get("medmcqa"), mm_id) or api.dataset_info(mm_id).sha
    print(f"medmcqa @ {mm_rev}")
    for split in ("train", "validation"):
        ds = load_dataset(mm_id, split=split, revision=mm_rev)
        ds.to_parquet(str(raw / f"medmcqa_{split}.parquet"))
        print(f"medmcqa/{split}: {len(ds)} rows")

    mq_id = cfg["datasets"]["medqa"]["hf_id"]
    mq_rev = pinned_rev(pinned.get("datasets", {}).get("medqa"), mq_id) or api.dataset_info(mq_id).sha
    print(f"medqa @ {mq_rev}")
    ds = load_dataset(mq_id, split="test", revision=mq_rev)
    ds.to_parquet(str(raw / "medqa_test.parquet"))
    print(f"medqa/test: {len(ds)} rows")

    ds_pins = pinned.setdefault("datasets", {})
    for key, hf_id, rev in (("medmcqa", mm_id, mm_rev), ("medqa", mq_id, mq_rev)):
        if (ds_pins.get(key) or {}).get("revision") != rev:
            ds_pins[key] = {"hf_id": hf_id, "revision": rev, "staged_at": now}

    # --- model --------------------------------------------------------------
    if not args.skip_model:
        model_id = cfg["small_model"]["hf_id"]
        rev = (args.model_revision or pinned_rev(pinned.get("small_model"), model_id)
               or api.model_info(model_id).sha)
        path = snapshot_download(model_id, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        sm = pinned.setdefault("small_model", {})
        if sm.get("revision") != rev or sm.get("hf_id") != model_id:
            pinned["small_model"] = {"hf_id": model_id, "revision": rev, "staged_at": now}
        local.setdefault("small_model", {})["local_path"] = path
        LOCAL_PATH.write_text(json.dumps(local, indent=2) + "\n")
        print(f"{model_id}@{rev} -> {path}  (path recorded in {LOCAL_PATH.name})")

    if json.dumps(pinned, sort_keys=True) != pinned_before:
        PINNED_PATH.write_text(json.dumps(pinned, indent=2) + "\n")
        print(f"updated pins in {PINNED_PATH}; commit it")
    else:
        print(f"pins unchanged ({PINNED_PATH.name})")


if __name__ == "__main__":
    main()
