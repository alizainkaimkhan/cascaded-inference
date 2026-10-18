"""Canonical item tables, deduplication, and the frozen splits.

Every dataset is converted to one schema before anything else touches it:

    item_id      str   stable ID (MedMCQA's own `id`; "medqa-test-<row>" for MedQA)
    source       str   "medmcqa" | "medqa"
    origin       str   upstream split the item came from ("train", "validation", "test")
    question     str
    options      list[str] of length 4, in the dataset's original order (A, B, C, D)
    gold_idx     int   0..3, index into `options`
    subject      str   MedMCQA subject_name; "USMLE" for MedQA
    choice_type  str   MedMCQA "single"/"multi"; "single" for MedQA

Splits are stored as sorted item-ID lists (data/splits/<name>.csv) plus a manifest with
counts and hashes, so they can be committed and audited without the raw data.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

LETTERS = ("A", "B", "C", "D")
SPLIT_NAMES = ("ft_train", "ft_dev", "calibration", "test", "medqa_test", "pilot")


# ---------------------------------------------------------------------------
# Loading raw parquet snapshots (written on the login node by stage_assets.py)
# ---------------------------------------------------------------------------

def medmcqa_to_canonical(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "item_id": df["id"].astype(str),
            "source": "medmcqa",
            "origin": origin,
            "question": df["question"].fillna("").astype(str),
            "options": df[["opa", "opb", "opc", "opd"]].fillna("").astype(str).values.tolist(),
            # MedMCQA `cop` is 0-indexed (0 = opa). The hidden test split uses -1.
            "gold_idx": df["cop"].astype(int),
            "subject": df["subject_name"].fillna("Unknown").astype(str),
            "choice_type": df["choice_type"].fillna("single").astype(str),
        }
    )
    return out


def medqa_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    def opts(o):
        if isinstance(o, dict):
            return [str(o[k]) for k in LETTERS]
        # some parquet round-trips turn the dict into a list of {key, value} records
        d = {r["key"]: r["value"] for r in o}
        return [str(d[k]) for k in LETTERS]

    out = pd.DataFrame(
        {
            "item_id": [f"medqa-test-{i:04d}" for i in range(len(df))],
            "source": "medqa",
            "origin": "test",
            "question": df["question"].astype(str).values,
            "options": [opts(o) for o in df["options"]],
            "gold_idx": [LETTERS.index(a) for a in df["answer_idx"]],
            "subject": "USMLE",
            "choice_type": "single",
        }
    )
    return out


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    return {
        "medmcqa_train": medmcqa_to_canonical(pd.read_parquet(raw_dir / "medmcqa_train.parquet"), "train"),
        "medmcqa_validation": medmcqa_to_canonical(
            pd.read_parquet(raw_dir / "medmcqa_validation.parquet"), "validation"
        ),
        "medqa_test": medqa_to_canonical(pd.read_parquet(raw_dir / "medqa_test.parquet")),
    }


# ---------------------------------------------------------------------------
# Validity and deduplication
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return _NON_ALNUM.sub(" ", s).strip()


def dedup_key(question: str, options: list[str], mode: str) -> str:
    q = normalize_text(question)
    if mode == "question":
        return q
    if mode == "question_options":
        return q + " || " + " | ".join(sorted(normalize_text(o) for o in options))
    raise ValueError(f"unknown dedup mode {mode!r}")


def is_valid(df: pd.DataFrame) -> pd.Series:
    has_q = df["question"].str.strip().str.len() > 0
    has_opts = df["options"].apply(lambda o: len(o) == 4 and all(str(x).strip() for x in o))
    gold_ok = df["gold_idx"].between(0, 3)
    distinct_opts = df["options"].apply(lambda o: len({normalize_text(x) for x in o}) == 4)
    return has_q & has_opts & gold_ok & distinct_opts


def _add_keys(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    df["_key"] = [dedup_key(q, o, mode) for q, o in zip(df["question"], df["options"])]
    df["_gold_text"] = [normalize_text(o[g]) for o, g in zip(df["options"], df["gold_idx"])]
    return df


def dedup_within(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicate keys. Groups whose copies disagree on the gold answer text are
    dropped entirely (label noise we cannot adjudicate); otherwise keep the smallest item_id."""
    df = df.sort_values("item_id")
    n_gold = df.groupby("_key")["_gold_text"].transform("nunique")
    conflicted = n_gold > 1
    kept = df[~conflicted].drop_duplicates("_key", keep="first")
    report = {
        "n_in": int(len(df)),
        "n_dup_groups": int((df["_key"].value_counts() > 1).sum()),
        "n_dropped_conflicting_gold": int(conflicted.sum()),
        "n_dropped_redundant_copies": int((~conflicted).sum() - len(kept)),
        "n_out": int(len(kept)),
    }
    return kept, report


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, by: str, rng: np.random.Generator) -> pd.Index:
    """Proportional allocation per stratum with largest-remainder rounding; returns the
    index of the sampled rows. `df` must already be in a deterministic order."""
    counts = df[by].value_counts().sort_index()
    quotas = counts / counts.sum() * n
    alloc = np.floor(quotas).astype(int)
    remainder = (quotas - alloc).sort_values(ascending=False, kind="mergesort")
    for stratum in remainder.index[: n - alloc.sum()]:
        alloc[stratum] += 1
    chosen = []
    for stratum, k in alloc.items():
        idx = df.index[df[by] == stratum].to_numpy()
        chosen.extend(rng.choice(idx, size=int(k), replace=False).tolist())
    assert len(chosen) == n
    return pd.Index(chosen)


def build_splits(raw: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    sc = cfg["splits"]
    mode = sc["dedup_key"]
    rng = np.random.default_rng(sc["seed"])
    report: dict = {"dedup_key": mode, "seed": sc["seed"]}

    tr = _add_keys(raw["medmcqa_train"], mode)
    va = _add_keys(raw["medmcqa_validation"], mode)
    mq = _add_keys(raw["medqa_test"], mode)

    for name, d in (("medmcqa_train", tr), ("medmcqa_validation", va), ("medqa_test", mq)):
        report[f"{name}_invalid"] = int((~is_valid(d)).sum())
    tr, va, mq = tr[is_valid(tr)], va[is_valid(va)], mq[is_valid(mq)]

    # 1. Evaluation pools first; they take priority over training data.
    mq, report["dedup_medqa"] = dedup_within(mq)
    va, report["dedup_validation"] = dedup_within(va)
    cross = va["_key"].isin(set(mq["_key"]))
    report["validation_dropped_overlap_medqa"] = int(cross.sum())
    va = va[~cross]

    # 2. Training pool: drop anything that matches an evaluation item, then dedup.
    eval_keys = set(va["_key"]) | set(mq["_key"])
    leak = tr["_key"].isin(eval_keys)
    report["train_dropped_overlap_eval"] = int(leak.sum())
    tr, report["dedup_train"] = dedup_within(tr[~leak])

    # 3. Calibration / test partition of validation, stratified by subject.
    va = va.sort_values("item_id").reset_index(drop=True)
    cal_idx = stratified_sample(va, sc["calibration_size"], "subject", rng)
    calibration = va.loc[cal_idx]
    test = va.drop(index=cal_idx)

    # 4. Fine-tuning train/dev: one seeded draw, first block -> train, next block -> dev.
    tr = tr.sort_values("item_id").reset_index(drop=True)
    need = sc["ft_train_size"] + sc["ft_dev_size"]
    if need > len(tr):
        raise ValueError(f"need {need} train items, only {len(tr)} after dedup")
    draw = rng.permutation(len(tr))[:need]
    ft_train = tr.iloc[draw[: sc["ft_train_size"]]]
    ft_dev = tr.iloc[draw[sc["ft_train_size"]:]]

    # 5. Pilot: plain seeded sample of the calibration split.
    cal_sorted = calibration.sort_values("item_id").reset_index(drop=True)
    pilot = cal_sorted.iloc[np.sort(rng.choice(len(cal_sorted), sc["pilot_size"], replace=False))]

    splits = {
        "ft_train": ft_train,
        "ft_dev": ft_dev,
        "calibration": calibration,
        "test": test,
        "medqa_test": mq,
        "pilot": pilot,
    }
    splits = {k: v.drop(columns=["_key", "_gold_text"]).sort_values("item_id").reset_index(drop=True)
              for k, v in splits.items()}
    assert_disjoint(splits)
    report["sizes"] = {k: len(v) for k, v in splits.items()}
    report["subjects_calibration"] = splits["calibration"]["subject"].value_counts().sort_index().to_dict()
    report["subjects_test"] = splits["test"]["subject"].value_counts().sort_index().to_dict()
    report["choice_type_eval"] = (
        pd.concat([splits["calibration"], splits["test"]])["choice_type"].value_counts().to_dict()
    )
    return splits, report


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Every split is disjoint from every other, except pilot, which is a subset of calibration."""
    names = [n for n in splits if n != "pilot"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = set(splits[a]["item_id"]) & set(splits[b]["item_id"])
            assert not inter, f"{a} and {b} share {len(inter)} items"
    if "pilot" in splits:
        assert set(splits["pilot"]["item_id"]) <= set(splits["calibration"]["item_id"])


# ---------------------------------------------------------------------------
# Writing and reading frozen splits
# ---------------------------------------------------------------------------

def _ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_splits(splits: dict[str, pd.DataFrame], report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}, "report": report}
    for name, df in splits.items():
        df[["item_id", "subject"]].to_csv(out_dir / f"{name}.csv", index=False)
        manifest["splits"][name] = {"n": len(df), "sha256_ids": _ids_sha256(df["item_id"].tolist())}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int))


def load_split(name: str, raw_dir: Path, splits_dir: Path) -> pd.DataFrame:
    """Canonical items for a frozen split, verified against the manifest hash."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    ids = pd.read_csv(Path(splits_dir) / f"{name}.csv")["item_id"].astype(str).tolist()
    manifest = json.loads((Path(splits_dir) / "manifest.json").read_text())
    if _ids_sha256(ids) != manifest["splits"][name]["sha256_ids"]:
        raise RuntimeError(f"split {name} does not match its manifest hash")
    raw = load_raw(raw_dir)
    pool = pd.concat(raw.values(), ignore_index=True).drop_duplicates("item_id").set_index("item_id")
    missing = [i for i in ids if i not in pool.index]
    if missing:
        raise RuntimeError(f"{len(missing)} IDs in {name} not found in raw data, e.g. {missing[:3]}")
    return pool.loc[ids].reset_index()
