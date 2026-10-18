import numpy as np
import pandas as pd
import pytest

from cascade.data import build_splits, dedup_key, normalize_text
from cascade.prompts import (IDENTITY, LETTERS, gold_letter, item_permutations, sft_example,
                             user_message)
from cascade.signals import item_signals
from cascade.small_model import THINK_SUFFIX, check_letter_tokens

SUBJECTS = ["Anatomy", "Pharmacology", "Surgery", "Medicine", "Dental"]


def fake_items(n, origin, prefix, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "item_id": [f"{prefix}{i:06d}" for i in range(n)],
        "source": "medmcqa", "origin": origin,
        "question": [f"{prefix} question number {i}?" for i in range(n)],
        "options": [[f"{prefix}{i} opt {k}" for k in range(4)] for i in range(n)],
        "gold_idx": rng.integers(0, 4, n),
        "subject": rng.choice(SUBJECTS, n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        "choice_type": "single",
    })


@pytest.fixture
def raw():
    tr = fake_items(5000, "train", "t", 0)
    va = fake_items(800, "validation", "v", 1)
    mq = fake_items(100, "test", "m", 2).assign(source="medqa", subject="USMLE")
    # leaks and duplicates to be caught
    tr.loc[0, ["question", "options"]] = va.loc[0, ["question", "options"]].values      # train == val
    tr.at[1, "options"] = list(reversed(va.at[1, "options"]))                          # shuffled copy
    tr.at[1, "question"] = va.at[1, "question"].upper() + "  "
    va.loc[2, ["question", "options", "gold_idx"]] = va.loc[3, ["question", "options", "gold_idx"]].values  # val dup
    va.loc[4, ["question", "options"]] = va.loc[5, ["question", "options"]].values     # val dup...
    va.at[4, "gold_idx"] = (va.at[5, "gold_idx"] + 1) % 4                               # ...conflicting gold
    va.at[6, "options"] = ["x", "x", "y", "z"]                                          # invalid (repeated option)
    return {"medmcqa_train": tr, "medmcqa_validation": va, "medqa_test": mq}


CFG = {"splits": {"seed": 7, "dedup_key": "question_options", "calibration_size": 300,
                  "ft_train_size": 3000, "ft_dev_size": 500, "pilot_size": 50}}


def test_splits_dedup_disjoint_and_sized(raw):
    splits, rep = build_splits(raw, CFG)
    assert rep["sizes"]["calibration"] == 300 and rep["sizes"]["pilot"] == 50
    assert rep["sizes"]["ft_train"] == 3000 and rep["sizes"]["ft_dev"] == 500
    assert rep["train_dropped_overlap_eval"] == 2
    assert rep["dedup_validation"]["n_dropped_conflicting_gold"] == 2
    assert rep["dedup_validation"]["n_dropped_redundant_copies"] == 1
    assert rep["medmcqa_validation_invalid"] == 1
    assert rep["sizes"]["calibration"] + rep["sizes"]["test"] == 800 - 1 - 2 - 1
    ft = set(splits["ft_train"]["item_id"]) | set(splits["ft_dev"]["item_id"])
    assert not ft & {"t000000", "t000001"}


def test_splits_deterministic_and_stratified(raw):
    a, _ = build_splits(raw, CFG)
    b, _ = build_splits(raw, CFG)
    for k in a:
        assert a[k]["item_id"].tolist() == b[k]["item_id"].tolist()
    cal = a["calibration"]["subject"].value_counts(normalize=True)
    pool = pd.concat([a["calibration"], a["test"]])["subject"].value_counts(normalize=True)
    assert (cal - pool).abs().max() < 0.01


def test_normalize_and_dedup_key():
    assert normalize_text("  Which  of the following?? ") == "which of the following"
    k1 = dedup_key("Q?", ["b", "a", "c", "d"], "question_options")
    k2 = dedup_key("q", ["A", "B", "C", "D"], "question_options")
    assert k1 == k2
    assert dedup_key("Q", ["a"] * 4, "question") == "q"


def test_permutations_fixed_distinct_and_mappable():
    p1 = item_permutations("abc", 9, 1)
    assert p1 == item_permutations("abc", 9, 1)
    assert p1 != item_permutations("abd", 9, 1)
    assert p1[0] == IDENTITY and len(set(p1)) == 10
    for perm in p1:
        msg = user_message("Q", ["w", "x", "y", "z"], perm)
        g = gold_letter(2, perm)  # original option "y"
        assert f"\n{g}. y" in msg


class FakeTok:
    """Character-level tokenizer with a Qwen3-like template, enough to exercise the checks."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        s = f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        return s + ("<think>\n\n</think>\n\n" if not enable_thinking else "")

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def test_letter_token_check_and_sft_prefix_identity():
    tok = FakeTok()
    msg = user_message("Q", ["a", "b", "c", "d"])
    from cascade.prompts import chat_prompt
    p = chat_prompt(tok, msg)
    assert p.endswith(THINK_SUFFIX)
    assert check_letter_tokens(tok, p, False).ids == [ord(L) for L in LETTERS]
    ex = sft_example(tok, msg, "C")
    assert ex["prompt"] == p and ex["completion"] == "C<|im_end|>"


def test_item_signals_perm_agreement():
    rows = []
    for item, answers in {"i1": [2, 2, 2, 2], "i2": [1, 3, 1, 0]}.items():
        for pres, ans in enumerate(answers):
            logits = [0.0, 0.0, 0.0, 0.0]
            logits[ans if pres == 0 else 0] = 3.0  # slot picked; only pres 0 logits matter for msp
            rows.append({"item_id": item, "presentation": pres, "perm": "0123" if pres == 0 else "3210",
                         "gold_idx": 2, "subject": "S", "chosen_orig": ans, "correct": ans == 2,
                         "letter_mass": 0.99, **{f"logit_{L}": v for L, v in zip(LETTERS, logits)}})
    sig = item_signals(pd.DataFrame(rows)).set_index("item_id")
    assert sig.loc["i1", "perm_agreement"] == 1.0
    assert sig.loc["i2", "perm_agreement"] == 0.5
    assert sig.loc["i1", "correct"] and not sig.loc["i2", "correct"]
