#!/usr/bin/env python
"""Week-1 go/no-go readout. CPU only; reads cached tables, makes no model calls.

    python scripts/pilot_report.py --run base_bf16 --split pilot
Writes results/<run>/<split>_pilot_report.md and prints it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.cascade import budget_curve, contingency, oracle_accuracy  # noqa: E402
from cascade.config import load_config, repo_path  # noqa: E402
from cascade.metrics import accuracy, aurc, auroc, bootstrap_ci, ece_equal_mass, eaurc  # noqa: E402
from cascade.signals import item_signals  # noqa: E402


def fmt_ci(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base_bf16")
    ap.add_argument("--split", default="pilot")
    args = ap.parse_args()

    cfg = load_config()
    res = repo_path(cfg["paths"]["results"])
    small = item_signals(pd.read_parquet(res / args.run / f"{args.split}.parquet"))
    front = pd.read_parquet(res / "frontier" / f"{args.split}.parquet")
    df = small.merge(front[["item_id", "frontier_correct", "parse_status"]], on="item_id", how="inner")
    if len(df) != len(small):
        sys.exit(f"only {len(df)}/{len(small)} items have frontier answers")

    s, f = df["correct"].to_numpy(), df["frontier_correct"].to_numpy()
    acc_s = bootstrap_ci(accuracy, s)
    acc_f = bootstrap_ci(accuracy, f)
    gap = bootstrap_ci(lambda a, b: b.mean() - a.mean(), s.astype(float), f.astype(float))
    headroom = bootstrap_ci(lambda a, b: np.mean(~a & b), s, f)
    orc_acc, orc_rate = oracle_accuracy(s, f)
    tab = contingency(s, f)

    lines = [f"# Pilot report: {args.run} vs frontier on `{args.split}` (n = {len(df)})", ""]
    lines += ["## Accuracy (95% bootstrap CI)", "",
              f"- Small model: {fmt_ci(acc_s)}",
              f"- Frontier: {fmt_ci(acc_f)}",
              f"- Gap (frontier − small): {fmt_ci(gap)}",
              f"- Frontier parse status: {df['parse_status'].value_counts().to_dict()}", ""]
    lines += ["## Error overlap", "", tab.to_markdown(), "",
              f"- **Escalation headroom** (small wrong, frontier right): {fmt_ci(headroom)}",
              f"- Small wrong AND frontier wrong (unrecoverable): {np.mean(~s & ~f):.3f}",
              f"- Small right, frontier wrong (escalation can hurt): {np.mean(s & ~f):.3f}",
              f"- Oracle cascade: accuracy {orc_acc:.3f} at call rate {orc_rate:.3f}", ""]

    sig_cols = ["msp"] + (["perm_agreement"] if "perm_agreement" in df else [])
    lines += ["## Confidence signals (small model, original-order answer)", "",
              "| signal | AUROC | AURC | E-AURC | ECE (15 equal-mass) | mean conf |",
              "|---|---|---|---|---|---|"]
    for c in sig_cols:
        lines.append(f"| {c} | {auroc(df[c], s):.3f} | {aurc(df[c], s):.3f} | {eaurc(df[c], s):.3f} | "
                     f"{ece_equal_mass(df[c], s):.3f} | {df[c].mean():.3f} |")
    lines += ["", "ECE for perm_agreement is shown for completeness; it is a ranking signal, not a probability.", ""]

    rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    bc = pd.concat({c: budget_curve(df[c], s, f, rates)[["call_rate", "accuracy"]].set_index("call_rate")["accuracy"]
                    for c in sig_cols}, axis=1)
    lines += ["## Budget regime (expected accuracy, ties broken at random)", "", bc.to_markdown(floatfmt=".3f"), ""]

    lm = df["letter_mass"]
    lines += ["## Prompt sanity", "",
              f"- Probability mass on A–D at the answer position: median {lm.median():.4f}, "
              f"min {lm.min():.4f}, share below 0.9: {(lm < 0.9).mean():.3f}",
              "- If mass is often below ~0.9, the model wants to emit something other than a bare letter; "
              "fix the prompt before trusting any confidence number.", ""]

    lines += ["## Go / no-go guide", "",
              "Proceed if the headroom CI sits clearly above zero (as a rule of thumb, ≥ 0.08–0.10 of items) "
              "and the budget curve at r = 0.2–0.3 recovers a meaningful share of the oracle gain. "
              "Rethink the design if headroom is near zero (errors nested) or the frontier gap is within noise.", ""]

    report = "\n".join(lines)
    (res / args.run / f"{args.split}_pilot_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
