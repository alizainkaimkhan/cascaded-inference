"""Exact option-letter logits from the small model (HF transformers, no vLLM).

One forward pass per (item, presentation); we read the logits at the last prompt position,
i.e. the first assistant token after Qwen3's empty <think></think> block. Output is one
row per (item, presentation) -- the per-item table every later analysis reads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .prompts import LETTERS, chat_prompt, gold_letter, item_permutations, perm_str, user_message

THINK_SUFFIX = "</think>\n\n"


@dataclass
class LetterTokens:
    ids: list[int]          # token id for "A", "B", "C", "D" in the position after the prompt


def check_letter_tokens(tokenizer, sample_prompt: str, enable_thinking: bool) -> LetterTokens:
    """Fail loudly if the letter position is not what we think it is."""
    if not enable_thinking and not sample_prompt.endswith(THINK_SUFFIX):
        raise AssertionError(
            f"chat template did not end with the empty think block; tail={sample_prompt[-60:]!r}"
        )
    base = tokenizer.encode(sample_prompt, add_special_tokens=False)
    ids = []
    for L in LETTERS:
        single = tokenizer.encode(L, add_special_tokens=False)
        if len(single) != 1:
            raise AssertionError(f"letter {L!r} is {len(single)} tokens: {single}")
        joined = tokenizer.encode(sample_prompt + L, add_special_tokens=False)
        if joined != base + single:
            raise AssertionError(
                f"appending {L!r} re-tokenizes the prompt boundary: tail {joined[-4:]} vs {base[-3:] + single}"
            )
        ids.append(single[0])
    if len(set(ids)) != 4:
        raise AssertionError(f"letter ids not distinct: {ids}")
    return LetterTokens(ids)


def load_model(model_path: str, precision: str = "bf16", adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if precision == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision != "bf16":
        raise ValueError(f"precision must be bf16 or nf4, got {precision!r}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def build_requests(items: pd.DataFrame, tokenizer, n_presentations: int, perm_seed: int,
                   prompt_version: str, enable_thinking: bool) -> pd.DataFrame:
    rows = []
    for it in items.itertuples(index=False):
        perms = item_permutations(it.item_id, n_presentations - 1, perm_seed)
        for p_idx, perm in enumerate(perms):
            text = chat_prompt(tokenizer, user_message(it.question, it.options, perm, prompt_version), enable_thinking)
            rows.append(
                {
                    "item_id": it.item_id,
                    "presentation": p_idx,
                    "perm": perm_str(perm),
                    "gold_idx": int(it.gold_idx),
                    "gold_letter": gold_letter(int(it.gold_idx), perm),
                    "subject": it.subject,
                    "prompt": text,
                }
            )
    return pd.DataFrame(rows)


def score(requests: pd.DataFrame, model, tokenizer, letters: LetterTokens, batch_size: int = 32,
          log_every: int = 50) -> pd.DataFrame:
    import torch

    lengths = requests["prompt"].map(lambda s: len(tokenizer.encode(s, add_special_tokens=False)))
    order = np.argsort(-lengths.to_numpy(), kind="stable")  # longest first: OOM shows up immediately
    letter_ids = torch.tensor(letters.ids, device=model.device)
    out = {k: np.empty(len(requests)) for k in ("logit_A", "logit_B", "logit_C", "logit_D", "letter_mass")}
    top_id = np.empty(len(requests), dtype=np.int64)

    n_batches = math.ceil(len(order) / batch_size)
    with torch.inference_mode():
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            enc = tokenizer(requests["prompt"].iloc[idx].tolist(), return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:  # older transformers
                logits = model(**enc).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            sel = logits[:, letter_ids]
            out_np = sel.cpu().numpy()
            for j, L in enumerate(LETTERS):
                out[f"logit_{L}"][idx] = out_np[:, j]
            out["letter_mass"][idx] = logp[:, letter_ids].exp().sum(-1).cpu().numpy()
            top_id[idx] = logits.argmax(-1).cpu().numpy()
            if log_every and (b % log_every == 0 or b == n_batches - 1):
                print(f"batch {b + 1}/{n_batches}", flush=True)

    res = requests.drop(columns=["prompt"]).copy()
    for k, v in out.items():
        res[k] = v
    res["n_prompt_tokens"] = lengths.to_numpy()
    res["top_token_id"] = top_id
    res["top_is_letter"] = np.isin(top_id, letters.ids)
    L = res[["logit_A", "logit_B", "logit_C", "logit_D"]].to_numpy()
    res["chosen_slot"] = L.argmax(1)
    res["chosen_letter"] = [LETTERS[s] for s in res["chosen_slot"]]
    res["chosen_orig"] = [int(p[s]) for p, s in zip(res["perm"], res["chosen_slot"])]
    res["correct"] = res["chosen_orig"] == res["gold_idx"]
    return res
