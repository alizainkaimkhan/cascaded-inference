"""QLoRA supervised fine-tuning on MedMCQA, with loss only on the answer letter.

Every training example is built from exactly the string the evaluation harness reads
logits after (prompts.chat_prompt), followed by the gold letter and the end-of-turn
token. Labels are -100 on every prompt token, so the only supervised positions are the
letter and the end-of-turn token. The letter position is therefore *the same position*
that run_small.py reads confidence from.

We use plain transformers.Trainer plus a tiny collator rather than trl's SFTTrainer: the
masking is explicit and testable here, and it does not depend on trl's API, which changes
often.
"""
from __future__ import annotations

from dataclasses import dataclass

from .prompts import IDENTITY, LETTERS, gold_letter, sft_example, user_message


def build_examples(items, tokenizer, prompt_version: str, enable_thinking: bool,
                   letter_ids: list[int]) -> list[dict]:
    """items: canonical DataFrame. Returns [{input_ids, labels}] in the original option order."""
    out = []
    for it in items.itertuples(index=False):
        ex = sft_example(tokenizer, user_message(it.question, it.options, IDENTITY, prompt_version),
                         gold_letter(int(it.gold_idx), IDENTITY), enable_thinking)
        p = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        c = tokenizer.encode(ex["completion"], add_special_tokens=False)
        if c[0] not in letter_ids:
            raise AssertionError(f"completion does not start with a letter token: {c[:3]} for {it.item_id}")
        out.append({"input_ids": p + c, "labels": [-100] * len(p) + c, "item_id": it.item_id})
    return out


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])  # right padding is fine for training
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class ListDataset:
    """Map-style dataset (Trainer/DataLoader only need __len__ and __getitem__)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"input_ids": r["input_ids"], "labels": r["labels"]}

    def lengths(self) -> list[int]:
        return [len(r["input_ids"]) for r in self.rows]


def letter_ids_for(tokenizer) -> list[int]:
    ids = [tokenizer.encode(L, add_special_tokens=False) for L in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]
