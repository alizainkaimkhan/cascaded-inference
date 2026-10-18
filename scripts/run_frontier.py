#!/usr/bin/env python
"""Query the frontier model once per item (original option order), cached, then export
results/frontier/<split>.parquet. Runs anywhere with internet and an API key
(your laptop or the LARCC login node; not a compute node).

    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
    python scripts/run_frontier.py --split pilot --dry-run
    python scripts/run_frontier.py --split pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import load_config, repo_path  # noqa: E402
from cascade.data import load_split  # noqa: E402
from cascade.frontier import BACKENDS, FrontierCache, run_uncached  # noqa: E402
from cascade.prompts import IDENTITY, LETTERS, user_message  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="show the prompt and the uncached count; send nothing")
    args = ap.parse_args()

    cfg = load_config()
    fc = cfg["frontier"]
    if not fc.get("provider") or not fc.get("model"):
        sys.exit("set frontier.provider and frontier.model (configs/pinned.json or default.yaml) first")
    model, version = fc["model"], cfg["prompt"]["version"]

    items = load_split(args.split, repo_path(cfg["paths"]["raw"]), repo_path(cfg["paths"]["splits"]))
    if args.limit:
        items = items.head(args.limit)
    jobs = [(it.item_id, user_message(it.question, it.options, IDENTITY, version))
            for it in items.itertuples(index=False)]

    cache = FrontierCache(repo_path(cfg["paths"]["frontier_cache"]))
    if args.dry_run:
        n_uncached = sum(cache.get(model, i, p) is None for i, p in jobs)
        print(jobs[0][1])
        print(f"\n{model}: {len(jobs)} items, {n_uncached} uncached")
        return

    backend = BACKENDS[fc["provider"]](model, fc["max_output_tokens"], fc.get("temperature"))
    run_uncached(backend, cache, model, version, jobs, args.workers)

    rows = []
    for it, (_, prompt) in zip(items.itertuples(index=False), jobs):
        r = cache.get(model, it.item_id, prompt)
        rows.append({
            "item_id": it.item_id, "subject": it.subject, "gold_idx": int(it.gold_idx),
            "frontier_letter": r["parsed_letter"], "parse_status": r["parse_status"],
            "frontier_correct": r["parsed_letter"] == LETTERS[int(it.gold_idx)],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            "response_text": r["response_text"],
        })
    df = pd.DataFrame(rows)
    out = repo_path(cfg["paths"]["results"]) / "frontier"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{args.split}.parquet", index=False)
    summary = {
        "model": model, "split": args.split, "n": len(df),
        "accuracy": float(df["frontier_correct"].mean()),
        "parse_status": df["parse_status"].value_counts().to_dict(),
        "input_tokens_total": int(df["input_tokens"].sum()),
        "output_tokens_total": int(df["output_tokens"].sum()),
    }
    (out / f"{args.split}.meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["parse_status"].get("fail") or summary["parse_status"].get("loose"):
        print("Inspect non-exact parses:\n", df[df["parse_status"] != "exact"][["item_id", "response_text"]].head(20))


if __name__ == "__main__":
    main()
