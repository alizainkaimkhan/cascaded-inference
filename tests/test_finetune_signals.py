import numpy as np
import pandas as pd
import pytest

from cascade.finetune import ListDataset, PadCollator, build_examples
from cascade.frontier import parse_letter
from cascade.prompts import LETTERS, chat_prompt, user_message
from cascade.signals import item_signals
from test_data_prompts import FakeTok


def items(n=3):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)], "question": [f"Q{k}?" for k in range(n)],
                         "options": [["a", "b", "c", "d"]] * n, "gold_idx": [0, 2, 3][:n], "subject": "S"})


def test_sft_examples_supervise_only_the_answer_at_the_inference_position():
    tok = FakeTok()
    letter_ids = [ord(L) for L in LETTERS]
    rows = build_examples(items(), tok, "v1", False, letter_ids)
    for r, gold in zip(rows, [0, 2, 3]):
        prompt = chat_prompt(tok, user_message(f"Q{rows.index(r)}?", ["a", "b", "c", "d"]))
        n = len(tok.encode(prompt))
        assert r["input_ids"][:n] == tok.encode(prompt)          # byte-identical to inference prompt
        assert all(l == -100 for l in r["labels"][:n])           # no loss on the prompt
        assert r["labels"][n] == ord(LETTERS[gold])              # first supervised token = gold letter
        assert r["labels"][n:] == r["input_ids"][n:]


def test_build_examples_rejects_non_letter_completion():
    with pytest.raises(AssertionError):
        build_examples(items(1), FakeTok(), "v1", False, letter_ids=[1, 2, 3, 4])


def test_collator_pads_right_and_masks_padding():
    torch = pytest.importorskip("torch")
    ds = ListDataset([{"input_ids": [5, 6, 7], "labels": [-100, -100, 7]}, {"input_ids": [5, 9], "labels": [-100, 9]}])
    b = PadCollator(pad_token_id=0)([ds[0], ds[1]])
    assert b["input_ids"].tolist() == [[5, 6, 7], [5, 9, 0]]
    assert b["labels"].tolist() == [[-100, -100, 7], [-100, 9, -100]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_parse_letter_markdown_bold_is_exact():
    assert parse_letter("**A**") == ("A", "exact")
    assert parse_letter("The answer is **C**.") == ("C", "phrase")


@pytest.mark.parametrize("text,expected", [
    # real responses from the full frontier pass
    ("A painless bilateral genu valgum in a 4-year-old most likely shows metaphyseal cupping.\n\nD", ("D", "line")),
    ("A papule on the hand that ulcerates in an abattoir worker suggests cutaneous anthrax.\n\nB", ("B", "line")),
    ("Zygomatic bone articulates with frontal, maxilla, temporal, sphenoid. Not nasal.\n\n**C**", ("C", "line")),
    ("SE = 1/\u221a100 = 0.1\n\nB", ("B", "line")),
    ("**D**\n\nThe question looks garbled: the original answer sequences seem to be missing.", ("D", "line")),
    ("A\nExample 1 of 2", ("A", "line")),
    # the article "A" in prose must never be read as an answer
    ("A tack placed below and lateral to the iliopubic tract lies in the triangle of pain.", (None, "fail")),
    ("", (None, "fail")),
    ("B\n\nOn reflection the best option is\n\nC", (None, "fail")),       # conflicting letters
    ("Both look plausible, but the answer is D.", ("D", "phrase")),
    ("The answer is a matter of debate.", (None, "fail")),                  # lowercase article, not a letter
])
def test_parse_letter_cases(text, expected):
    assert parse_letter(text) == expected


def _table(logit_rows):
    rows = []
    for k, lg in enumerate(logit_rows):
        rows.append({"item_id": f"i{k}", "presentation": 0, "perm": "0123", "gold_idx": 0, "subject": "S",
                     "chosen_orig": int(np.argmax(lg)), "correct": int(np.argmax(lg)) == 0, "letter_mass": 1.0,
                     **{f"logit_{L}": v for L, v in zip(LETTERS, lg)}})
    return pd.DataFrame(rows)


def test_logodds_matches_msp_and_does_not_saturate():
    s = item_signals(_table([[2.0, 1.0, 0.0, -1.0], [40.0, 0.0, 0.0, 0.0], [60.0, 0.0, 0.0, 0.0]]))
    p = s["msp"].to_numpy()
    assert np.isclose(s["msp_logodds"].iloc[0], np.log(p[0] / (1 - p[0])))
    assert p[1] == p[2] == 1.0                                   # msp saturates ...
    assert s["msp_logodds"].iloc[2] > s["msp_logodds"].iloc[1]   # ... log-odds still ranks them


def test_logodds_with_tied_top_letters_is_log_one_over_rest():
    s = item_signals(_table([[3.0, 3.0, 0.0, 0.0]]))
    p = s["msp"].iloc[0]
    assert np.isclose(s["msp_logodds"].iloc[0], np.log(p / (1 - p)))
