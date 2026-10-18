"""Turn a per-(item, presentation) table from run_small.py into one row per item with both
confidence signals, scored on the same answer (the original-order greedy answer)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import softmax

LOGIT_COLS = ["logit_A", "logit_B", "logit_C", "logit_D"]


def item_signals(table: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    p0 = table[table["presentation"] == 0].set_index("item_id").sort_index()
    if (p0["perm"] != "0123").any():
        raise ValueError("presentation 0 must be the original order")
    probs = softmax(p0[LOGIT_COLS].to_numpy(), temperature)
    out = pd.DataFrame(index=p0.index)
    out["subject"] = p0["subject"]
    out["gold_idx"] = p0["gold_idx"]
    out["pred_idx"] = p0["chosen_orig"]
    out["correct"] = p0["correct"].astype(bool)
    out["msp"] = probs.max(axis=1)
    for j, L in enumerate("ABCD"):
        out[f"p_{L}"] = probs[:, j]
    out["letter_mass"] = p0["letter_mass"]

    n_pres = table.groupby("item_id")["presentation"].nunique()
    if n_pres.max() > 1:
        merged = table.merge(p0[["chosen_orig"]].rename(columns={"chosen_orig": "orig_answer"}),
                             left_on="item_id", right_index=True)
        agree = (merged["chosen_orig"] == merged["orig_answer"]).groupby(merged["item_id"]).mean()
        out["perm_agreement"] = agree.reindex(out.index)
        out["n_presentations"] = n_pres.reindex(out.index)
    return out.reset_index()


def probs_matrix(table: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
    p0 = table[table["presentation"] == 0].sort_values("item_id")
    return softmax(p0[LOGIT_COLS].to_numpy(), temperature)
