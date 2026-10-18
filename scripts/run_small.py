#!/usr/bin/env python
"""Score a frozen split with the small model; writes results/<run>/<split>.parquet + .meta.json.

    python scripts/run_small.py --split pilot --run base_bf16
    python scripts/run_small.py --split calibration --run base_nf4 --precision nf4
    python scripts/run_small.py --split test --run ft_s1 --precision nf4 --adapter ckpt/ft_s1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import git_commit, git_dirty, load_config, repo_path  # noqa: E402
from cascade.data import load_split  # noqa: E402
from cascade.small_model import build_requests, check_letter_tokens, load_model, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True, help="run name, e.g. base_bf16, base_nf4, ft_s1")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "nf4"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--presentations", type=int, default=None, help="default: 1 + permutations.k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    sm = cfg["small_model"]
    model_path = sm.get("local_path") or sm["hf_id"]
    if not sm.get("revision"):
        print("WARNING: small_model.revision is not pinned; run stage_assets.py first", file=sys.stderr)

    out_dir = repo_path(cfg["paths"]["results"]) / args.run
    out_path = out_dir / f"{args.split}.parquet"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} exists (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_split(args.split, repo_path(cfg["paths"]["raw"]), repo_path(cfg["paths"]["splits"]))
    if args.limit:
        items = items.head(args.limit)
    n_pres = args.presentations or 1 + cfg["permutations"]["k"]

    t0 = time.time()
    model, tok = load_model(model_path, args.precision, args.adapter)
    reqs = build_requests(items, tok, n_pres, cfg["permutations"]["seed"], cfg["prompt"]["version"],
                          sm["enable_thinking"])
    letters = check_letter_tokens(tok, reqs["prompt"].iloc[0], sm["enable_thinking"])
    print(f"letter token ids: {letters.ids}; {len(items)} items x {n_pres} presentations")
    print("--- first prompt ---\n" + reqs["prompt"].iloc[0] + "<READ HERE>\n--------------------")

    t1 = time.time()
    res = score(reqs, model, tok, letters, args.batch_size)
    t2 = time.time()
    res.to_parquet(out_path, index=False)

    import torch
    import transformers

    p0 = res[res["presentation"] == 0]
    meta = {
        "run": args.run, "split": args.split, "n_items": len(items), "n_presentations": n_pres,
        "model_path": model_path, "hf_id": sm["hf_id"], "revision": sm.get("revision"),
        "precision": args.precision, "adapter": args.adapter,
        "prompt_version": cfg["prompt"]["version"], "perm_seed": cfg["permutations"]["seed"],
        "letter_token_ids": letters.ids,
        "accuracy_original_order": float(p0["correct"].mean()),
        "letter_mass_median": float(res["letter_mass"].median()),
        "letter_mass_frac_below_0.9": float((res["letter_mass"] < 0.9).mean()),
        "frac_top_token_is_letter": float(res["top_is_letter"].mean()),
        "load_seconds": t1 - t0, "score_seconds": t2 - t1,
        "gpu_seconds_per_item": (t2 - t1) / len(items),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "git_commit": git_commit(), "git_dirty": git_dirty(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / f"{args.split}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("accuracy_original_order", "letter_mass_median",
                                            "letter_mass_frac_below_0.9", "frac_top_token_is_letter",
                                            "score_seconds")}, indent=2))


if __name__ == "__main__":
    main()
