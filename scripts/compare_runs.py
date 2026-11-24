#!/usr/bin/env python
"""Compare small-model runs (base vs fine-tuned) on calibration, routing and the cascade.

    python scripts/compare_runs.py --runs base_nf4 ft_s1 --split test
    python scripts/compare_runs.py --runs base_nf4 ft_s1 ft_s2 ft_s3 --split test --tag nf4_vs_ft
    python scripts/compare_runs.py --runs base_bf16 base_nf4 --split test --tag precision

The first run is the reference; every other run gets paired-bootstrap differences against
it. For each run the temperature is fit on the calibration split and applied unchanged to
the evaluation split. CPU only; reads results/<run>/<split>.parquet and
results/frontier/<split>.parquet.

Writes results/compare/<tag>/<split>/: report.md, metrics.json, curves.csv, and three PNG
figures (reliability diagrams, risk-coverage curves, cascade accuracy vs escalation rate).

Signals
-------
  option probability : the model's probability for its chosen letter. Calibration metrics
                       use the probability itself; ranking metrics (AUROC, AURC, cascade)
                       use its log-odds, which orders items identically but does not round
                       to exactly 1.0, so saturated items keep their order instead of tying.
                       Reported raw and temperature-scaled: with four options, scaling can
                       reorder items slightly, so ranking metrics are computed for both.
  permutation agreement : share of the 10 option orders that give the same answer.
                       A ranking signal only; temperature scaling does not apply to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.cascade import budget_curve, threshold_point  # noqa: E402
from cascade.config import load_config, repo_path  # noqa: E402
from cascade.metrics import (accuracy, aurc, aurc_optimal, auroc, brier_multiclass,  # noqa: E402
                             ece_equal_mass, risk_coverage)
from cascade.recalibrate import fit_temperature, nll  # noqa: E402
from cascade.signals import LOGIT_COLS, item_signals  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1d1d1d", "#6b6b6b", "#e6e6e6"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _p0_logits(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p0 = table[table["presentation"] == 0].sort_values("item_id")
    return p0[LOGIT_COLS].to_numpy(float), p0["gold_idx"].to_numpy(int)


def load_run(results: Path, run: str, split: str, calib_split: str, frontier: pd.DataFrame) -> dict:
    cal = pd.read_parquet(results / run / f"{calib_split}.parquet")
    ev = pd.read_parquet(results / run / f"{split}.parquet")
    T = fit_temperature(*_p0_logits(cal))
    raw = item_signals(ev, 1.0).set_index("item_id").sort_index()
    ts = item_signals(ev, T).set_index("item_id").sort_index()
    logits, gold = _p0_logits(ev)
    f = frontier.set_index("item_id").reindex(raw.index)
    if f["frontier_correct"].isna().any():
        raise SystemExit(f"{run}/{split}: {int(f['frontier_correct'].isna().sum())} items lack frontier answers")
    return {
        "run": run, "T": T, "item_ids": raw.index.to_numpy(),
        "correct": raw["correct"].to_numpy(float),
        "frontier": f["frontier_correct"].to_numpy(float),
        "msp_raw": raw["msp"].to_numpy(float), "msp_ts": ts["msp"].to_numpy(float),
        "rank_p": raw["msp_logodds"].to_numpy(float),
        "rank_p_ts": ts["msp_logodds"].to_numpy(float),  # TS can reorder items slightly (K = 4)
        "perm": raw["perm_agreement"].to_numpy(float) if "perm_agreement" in raw else None,
        "probs_raw": raw[["p_A", "p_B", "p_C", "p_D"]].to_numpy(float),
        "probs_ts": ts[["p_A", "p_B", "p_C", "p_D"]].to_numpy(float),
        "gold": gold, "logits": logits,
        "nll_raw": nll(logits, gold, 1.0), "nll_ts": nll(logits, gold, T),
    }


# ---------------------------------------------------------------------------
# Metrics (one function so the bootstrap computes exactly what the tables report)
# ---------------------------------------------------------------------------

def stats(d: dict, idx: np.ndarray, rates, threshold: float) -> dict:
    c, f = d["correct"][idx], d["frontier"][idx]
    s = {"accuracy": accuracy(c)}
    for v in ("raw", "ts"):
        conf = d[f"msp_{v}"][idx]
        s[f"ece_{v}"] = ece_equal_mass(conf, c)
        s[f"brier_{v}"] = brier_multiclass(d[f"probs_{v}"][idx], d["gold"][idx])
        s[f"mean_conf_{v}"] = float(conf.mean())
        tp = threshold_point(conf, c, f, threshold)
        s[f"thr_call_rate_{v}"] = tp["call_rate"]
        s[f"thr_kept_acc_{v}"] = tp["retained_accuracy"]
        s[f"thr_system_acc_{v}"] = tp["system_accuracy"]
    opt = aurc_optimal(c)
    signals = {"p": d["rank_p"], "p_ts": d["rank_p_ts"]} | ({"perm": d["perm"]} if d["perm"] is not None else {})
    for name, conf in signals.items():
        conf = conf[idx]
        s[f"auroc_{name}"] = auroc(conf, c)
        a = aurc(conf, c)
        s[f"aurc_{name}"], s[f"eaurc_{name}"] = a, a - opt
        bc = budget_curve(conf, c, f, rates)
        for r, acc in zip(rates, bc["accuracy"]):
            s[f"cascade_{name}@{r:g}"] = float(acc)
    h = float(np.mean((1 - c) * f))
    for r in rates:
        s[f"cascade_random@{r:g}"] = float((1 - r) * c.mean() + r * f.mean())
        s[f"cascade_oracle@{r:g}"] = float(c.mean() + min(r, h))
    s["headroom"] = h
    s["frontier_accuracy"] = float(f.mean())
    return s


def paired_bootstrap(ref: dict, other: dict, rates, threshold, keys, n_boot: int, seed: int) -> dict:
    """Percentile CIs of (other - ref), resampling the same items for both runs."""
    if not np.array_equal(ref["item_ids"], other["item_ids"]):
        raise SystemExit("runs cover different items; paired comparison is not possible")
    n = len(ref["correct"])
    full = np.arange(n)
    point_ref, point_oth = stats(ref, full, rates, threshold), stats(other, full, rates, threshold)
    rng = np.random.default_rng(seed)
    deltas = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = stats(ref, idx, rates, threshold), stats(other, idx, rates, threshold)
        for k in keys:
            deltas[k].append(b[k] - a[k])
    out = {}
    for k in keys:
        v = np.asarray(deltas[k], float)
        v = v[np.isfinite(v)]
        lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan)
        out[k] = {"delta": point_oth[k] - point_ref[k], "lo": float(lo), "hi": float(hi)}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a9a")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _reliability_points(conf, correct, n_bins=15):
    order = np.argsort(conf, kind="stable")
    xs, ys = [], []
    for b in np.array_split(order, n_bins):
        if len(b):
            xs.append(conf[b].mean())
            ys.append(correct[b].mean())
    return np.array(xs), np.array(ys)


def figures(runs: list[dict], rates, out: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                         "legend.frameon": False})

    # 1. Reliability diagrams, one panel per run: raw (solid) vs temperature-scaled (dashed)
    k = len(runs)
    fig, axes = plt.subplots(1, k, figsize=(3.1 * k, 3.2), squeeze=False)
    for i, (ax, d) in enumerate(zip(axes[0], runs)):
        col = COLORS[i % len(COLORS)]
        ax.plot([0, 1], [0, 1], color="#b5b5b5", linewidth=1, linestyle=":", label="Perfect calibration")
        for v, ls, mk, lab in (("raw", "-", "o", "Raw"), ("ts", "--", "s", f"Temp.-scaled (T={d['T']:.2f})")):
            x, y = _reliability_points(d[f"msp_{v}"], d["correct"])
            ax.plot(x, y, linestyle=ls, marker=mk, markersize=4.5, linewidth=2, color=col,
                    markerfacecolor=col if v == "raw" else "white", markeredgewidth=1.4, label=lab)
        ax.set_xlim(0.2, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_title(d["run"], color=INK)
        ax.set_xlabel("Mean confidence (bin)")
        if i == 0:
            ax.set_ylabel("Accuracy (bin)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
        _style(ax)
    fig.suptitle(f"Reliability: option probability, {split} split (15 equal-mass bins)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Risk-coverage curves, one panel per signal
    sigs = [("rank_p", "Option probability")] + ([("perm", "Permutation agreement")] if runs[0]["perm"] is not None else [])
    fig, axes = plt.subplots(1, len(sigs), figsize=(4.4 * len(sigs), 3.4), squeeze=False, sharey=True)
    for ax, (key, title) in zip(axes[0], sigs):
        for i, d in enumerate(runs):
            cov, risk = risk_coverage(d[key], d["correct"])
            ax.plot(cov, risk, color=COLORS[i % len(COLORS)], linewidth=2,
                    label=f"{d['run']} (AURC {aurc(d[key], d['correct']):.3f})")
            if key == "rank_p":
                cov, risk = risk_coverage(d["rank_p_ts"], d["correct"])
                ax.plot(cov, risk, color=COLORS[i % len(COLORS)], linewidth=1.2, linestyle="--",
                        label=f"{d['run']}, TS (AURC {aurc(d['rank_p_ts'], d['correct']):.3f})")
        ax.set_title(title, color=INK)
        ax.set_xlabel("Coverage (share of items the small model answers)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        _style(ax)
    axes[0][0].set_ylabel("Selective risk (error rate on answered items)")
    fig.suptitle(f"Risk–coverage, {split} split (lower is better)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Cascade accuracy vs escalation rate, one panel per signal
    fig, axes = plt.subplots(1, len(sigs), figsize=(4.4 * len(sigs), 3.4), squeeze=False, sharey=True)
    fa = runs[0]["frontier"].mean()
    for ax, (key, title) in zip(axes[0], sigs):
        ax.axhline(fa, color="#8a8a8a", linewidth=1, linestyle=":", label=f"Frontier alone ({fa:.3f})")
        for i, d in enumerate(runs):
            col = COLORS[i % len(COLORS)]
            curve = budget_curve(d[key], d["correct"], d["frontier"])
            ax.plot(curve["call_rate"], curve["accuracy"], color=col, linewidth=2, label=d["run"])
            pts = budget_curve(d[key], d["correct"], d["frontier"], rates)
            ax.plot(pts["call_rate"], pts["accuracy"], linestyle="none", marker="o", markersize=5,
                    color=col, markeredgecolor="white", markeredgewidth=1.2)
            ax.plot([0, 1], [d["correct"].mean(), fa], color=col, linewidth=1, linestyle="--", alpha=0.7)
        ax.plot([], [], color="#8a8a8a", linewidth=1, linestyle="--", label="Random escalation (per run)")
        ax.set_title(title, color=INK)
        ax.set_xlabel("Escalation rate (share sent to the frontier)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        _style(ax)
    axes[0][0].set_ylabel("Cascade accuracy")
    fig.suptitle(f"Budget-regime cascade, {split} split (markers at "
                 + ", ".join(f"{int(r * 100)}%" for r in rates) + ")", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "cascade.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def aggregate(names, per_run, keys):
    """mean ± sd [min, max] over the named runs (e.g. fine-tuning seeds)."""
    out = {}
    for k in keys:
        v = np.array([per_run[n][k] for n in names], float)
        out[k] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="first run is the reference")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="calibration")
    ap.add_argument("--tag", default=None, help="output folder name (default: runs joined by _vs_)")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--threshold", type=float, default=0.9, help="face-value confidence threshold")
    ap.add_argument("--aggregate-prefix", default="ft_s", help="runs starting with this are pooled as seeds")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    results = repo_path(cfg["paths"]["results"])
    frontier = pd.read_parquet(results / "frontier" / f"{args.split}.parquet")
    runs = [load_run(results, r, args.split, args.calib_split, frontier) for r in args.runs]
    rates = args.rates
    full = np.arange(len(runs[0]["correct"]))
    per_run = {d["run"]: stats(d, full, rates, args.threshold) for d in runs}
    for d in runs:
        per_run[d["run"]].update({"temperature": d["T"], "nll_raw": d["nll_raw"], "nll_ts": d["nll_ts"]})

    has_perm = runs[0]["perm"] is not None
    delta_keys = (["accuracy", "ece_raw", "ece_ts", "brier_raw", "brier_ts", "auroc_p", "aurc_p", "eaurc_p",
                   "auroc_p_ts", "aurc_p_ts"]
                  + (["auroc_perm", "aurc_perm", "eaurc_perm"] if has_perm else [])
                  + [f"cascade_p@{r:g}" for r in rates]
                  + [f"cascade_p_ts@{r:g}" for r in rates]
                  + ([f"cascade_perm@{r:g}" for r in rates] if has_perm else [])
                  + ["thr_call_rate_raw", "thr_system_acc_raw", "thr_call_rate_ts", "thr_system_acc_ts"])
    ref = runs[0]
    deltas = {d["run"]: paired_bootstrap(ref, d, rates, args.threshold, delta_keys, args.boot, args.seed)
              for d in runs[1:]}

    tag = args.tag or "_vs_".join(args.runs)
    out = results / "compare" / tag / args.split
    out.mkdir(parents=True, exist_ok=True)
    seeds = [r for r in args.runs if r.startswith(args.aggregate_prefix)]

    # ---- tables ----
    names = args.runs
    L = [f"# {' vs '.join(names)} on `{args.split}` (n = {len(full)})", "",
         f"Reference: **{ref['run']}**. Temperatures fit on `{args.calib_split}` only. "
         f"Frontier accuracy {per_run[ref['run']]['frontier_accuracy']:.3f}.", ""]

    L += ["## 1. Accuracy and calibration (option probability)", "",
          "Temperature scaling (TS) cannot change accuracy; it only rescales confidence.", "",
          _table(["run", "accuracy", "T", "ECE raw", "ECE TS", "Brier raw", "Brier TS", "NLL raw", "NLL TS",
                  "mean conf raw", "mean conf TS"],
                 [[n, _f(per_run[n]["accuracy"]), _f(per_run[n]["temperature"], 2), _f(per_run[n]["ece_raw"]),
                   _f(per_run[n]["ece_ts"]), _f(per_run[n]["brier_raw"]), _f(per_run[n]["brier_ts"]),
                   _f(per_run[n]["nll_raw"]), _f(per_run[n]["nll_ts"]), _f(per_run[n]["mean_conf_raw"]),
                   _f(per_run[n]["mean_conf_ts"])] for n in names]), ""]

    rk = [("auroc_p", "AUROC prob"), ("aurc_p", "AURC prob"), ("eaurc_p", "E-AURC prob"),
          ("auroc_p_ts", "AUROC prob TS"), ("aurc_p_ts", "AURC prob TS"), ("eaurc_p_ts", "E-AURC prob TS")]
    if has_perm:
        rk += [("auroc_perm", "AUROC perm"), ("aurc_perm", "AURC perm"), ("eaurc_perm", "E-AURC perm")]
    L += ["## 2. Deciding when to answer: discrimination and risk–coverage", "",
          "Lower AURC / E-AURC is better. TS columns use temperature-scaled confidence; with four "
          "options scaling can reorder items slightly, so they need not equal the raw columns. "
          "E-AURC subtracts the best AURC possible at that run's accuracy, so runs with "
          "different accuracy compare fairly.", "",
          _table(["run"] + [h for _, h in rk], [[n] + [_f(per_run[n][k]) for k, _ in rk] for n in names]), ""]

    rows = []
    for n in names:
        p = per_run[n]
        rows.append([n, "option probability"] + [_f(p[f"cascade_p@{r:g}"]) for r in rates])
        rows.append([n, "option probability, TS"] + [_f(p[f"cascade_p_ts@{r:g}"]) for r in rates])
        if has_perm:
            rows.append([n, "permutation agreement"] + [_f(p[f"cascade_perm@{r:g}"]) for r in rates])
        rows.append([n, "*random escalation*"] + [_f(p[f"cascade_random@{r:g}"]) for r in rates])
        rows.append([n, "*oracle*"] + [_f(p[f"cascade_oracle@{r:g}"]) for r in rates])
    L += ["## 3. Cascade accuracy at fixed escalation rates (budget regime)", "",
          "Expected accuracy when the least-confident share of items goes to the frontier "
          "(ties broken at random).", "",
          _table(["run", "signal"] + [f"{int(r * 100)}%" for r in rates], rows), ""]

    rows = []
    for n in names:
        p = per_run[n]
        for v, lab in (("raw", "raw"), ("ts", "temperature-scaled")):
            rows.append([n, lab, _f(p[f"thr_call_rate_{v}"]), _f(p[f"thr_kept_acc_{v}"]), _f(p[f"thr_system_acc_{v}"])])
    L += [f"## 4. Threshold regime: escalate when confidence < {args.threshold:g}", "",
          "If confidence were honest, the answers kept would be at least "
          f"{args.threshold:.0%} accurate. This is where calibration (and TS) matters.", "",
          _table(["run", "confidence", "escalation rate", "accuracy of kept answers", "cascade accuracy"], rows), ""]

    if deltas:
        L += [f"## 5. Paired differences vs {ref['run']} (95% bootstrap CI, {args.boot} resamples)", "",
              "A CI that excludes 0 is a difference not explained by which items happened to be sampled.", ""]
        for n, dd in deltas.items():
            L += [f"### {n} − {ref['run']}", "",
                  _table(["metric", "difference", "95% CI"],
                         [[k, f"{v['delta']:+.3f}", f"[{v['lo']:+.3f}, {v['hi']:+.3f}]"] for k, v in dd.items()]), ""]

    if len(seeds) >= 2:
        keys = (["accuracy", "ece_raw", "ece_ts", "auroc_p", "aurc_p", "eaurc_p"]
                + [f"cascade_p@{r:g}" for r in rates]
                + ([f"cascade_perm@{r:g}" for r in rates] if has_perm else []))
        ag = aggregate(seeds, per_run, keys)
        L += [f"## 6. Across fine-tuning seeds ({', '.join(seeds)})", "",
              _table(["metric", "mean ± sd", "range"],
                     [[k, f"{m:.3f} ± {s:.3f}", f"[{lo:.3f}, {hi:.3f}]"] for k, (m, s, lo, hi) in ag.items()]), ""]

    L += ["Figures: `reliability.png`, `risk_coverage.png`, `cascade.png` in this folder."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "metrics.json").write_text(json.dumps({"per_run": per_run, "paired_deltas_vs_reference": deltas,
                                                  "reference": ref["run"], "split": args.split,
                                                  "rates": rates, "threshold": args.threshold}, indent=1))
    curves = []
    for d in runs:
        for key, name in (("rank_p", "option_probability"), ("perm", "permutation_agreement")):
            if d.get(key) is None:
                continue
            cov, risk = risk_coverage(d[key], d["correct"])
            bc = budget_curve(d[key], d["correct"], d["frontier"])
            curves.append(pd.DataFrame({"run": d["run"], "signal": name, "coverage": cov, "risk": risk,
                                        # coverage k/n (k = 1..n) <=> n - k items escalated
                                        "call_rate": bc["call_rate"].to_numpy()[:-1][::-1],
                                        "cascade_accuracy": bc["accuracy"].to_numpy()[:-1][::-1]}))
    pd.concat(curves).to_csv(out / "curves.csv", index=False)
    if not args.no_figures:
        figures(runs, rates, out, args.split)
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
