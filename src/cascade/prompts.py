"""Prompt text, option permutations, and the exact strings the small model sees.

Presentations
-------------
Each item is shown under 10 option orders: presentation 0 is the original order, 1..9 are
distinct non-identity permutations drawn deterministically from (item_id, seed). They are a
pure function of the item ID, so every model (base, NF4 base, each fine-tuned seed) sees
exactly the same orders.

A permutation `perm` is a tuple where perm[slot] = original option index shown at that
slot. Slot 0 is labelled A. So the gold letter in a presentation is
LETTERS[perm.index(gold_idx)], and a chosen slot s maps back to original option perm[s].
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

LETTERS = ("A", "B", "C", "D")
IDENTITY = (0, 1, 2, 3)
_ALL_PERMS = [p for p in itertools.permutations(range(4)) if p != IDENTITY]  # 23 orders

PROMPT_V1 = (
    "Answer the following multiple-choice question about medicine. "
    "Reply with only the letter of the correct option (A, B, C, or D).\n\n"
    "Question: {question}\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}"
)
TEMPLATES = {"v1": PROMPT_V1}


def item_permutations(item_id: str, k: int, seed: int) -> list[tuple[int, ...]]:
    """[identity, k distinct non-identity permutations], fixed per (item_id, seed)."""
    if not 0 <= k <= len(_ALL_PERMS):
        raise ValueError(f"k must be in [0, {len(_ALL_PERMS)}]")
    h = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    picks = rng.choice(len(_ALL_PERMS), size=k, replace=False)
    return [IDENTITY] + [_ALL_PERMS[i] for i in picks]


def perm_str(perm) -> str:
    return "".join(str(i) for i in perm)


def user_message(question: str, options: list[str], perm=IDENTITY, version: str = "v1") -> str:
    shown = [options[i] for i in perm]
    return TEMPLATES[version].format(question=question.strip(), **{L: o.strip() for L, o in zip(LETTERS, shown)})


def gold_letter(gold_idx: int, perm) -> str:
    return LETTERS[list(perm).index(gold_idx)]


def slot_to_original(slot: int, perm) -> int:
    return perm[slot]


def chat_prompt(tokenizer, user_text: str, enable_thinking: bool = False) -> str:
    """The exact string whose next token is read as the answer.

    For Qwen3 with enable_thinking=False the template ends with
    '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n', so the next token is the letter."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def sft_example(tokenizer, user_text: str, answer_letter: str, enable_thinking: bool = False) -> dict:
    """Prompt/completion pair for completion-only SFT.

    The prompt is byte-identical to what inference uses, so the trained position is the
    position read at evaluation. We build it by string concatenation rather than rendering
    an assistant turn through the chat template, because Qwen3's template treats historical
    assistant turns differently from the generation prompt."""
    return {
        "prompt": chat_prompt(tokenizer, user_text, enable_thinking),
        "completion": answer_letter + tokenizer.eos_token,  # "<|im_end|>" for Qwen3 instruct
    }
