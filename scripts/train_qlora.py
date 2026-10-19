#!/usr/bin/env python
"""QLoRA fine-tuning of the pinned Qwen3-8B on the frozen ft_train split.

    python scripts/train_qlora.py --seed 1
    python scripts/train_qlora.py --seed 1 --max-steps 20 --run-name smoke   # quick check

Writes checkpoints/<run>/adapter/ (the LoRA weights), plus train_meta.json and
log_history.json (training and ft_dev loss curves, GPU-hours, peak memory). Evaluate with:

    python scripts/run_small.py --split test --run ft_s1 --precision nf4 --adapter checkpoints/ft_s1/adapter
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import git_commit, git_dirty, load_config, repo_path  # noqa: E402
from cascade.data import load_split  # noqa: E402
from cascade.finetune import ListDataset, PadCollator, build_examples, letter_ids_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-name", default=None, help="default: ft_s<seed>")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke tests); -1 = full epochs")
    ap.add_argument("--limit-train", type=int, default=None, help="first N training items only (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    cfg = load_config()
    ft, sm = cfg["finetune"], cfg["small_model"]
    run = args.run_name or f"ft_s{args.seed}"
    out_dir = repo_path(ft["output_dir"]) / run
    if (out_dir / "adapter").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/adapter exists (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = sm.get("local_path") or sm["hf_id"]

    transformers.set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letters = letter_ids_for(tok)

    raw, splits = repo_path(cfg["paths"]["raw"]), repo_path(cfg["paths"]["splits"])
    train_items = load_split("ft_train", raw, splits)
    if args.limit_train:
        train_items = train_items.head(args.limit_train)
    dev_items = load_split("ft_dev", raw, splits)
    pv, think = cfg["prompt"]["version"], sm["enable_thinking"]
    train_ds = ListDataset(build_examples(train_items, tok, pv, think, letters))
    dev_ds = ListDataset(build_examples(dev_items, tok, pv, think, letters))
    lens = train_ds.lengths()
    print(f"train {len(train_ds)} / dev {len(dev_ds)} examples; tokens per example "
          f"median {sorted(lens)[len(lens)//2]}, max {max(lens)}")
    print("--- first example, decoded (supervised tokens in [[ ]]) ---")
    r = train_ds.rows[0]
    first_sup = next(i for i, l in enumerate(r["labels"]) if l != -100)
    print(tok.decode(r["input_ids"][:first_sup]) + "[[" + tok.decode(r["input_ids"][first_sup:]) + "]]")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft["lora_dropout"],
        target_modules=ft["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        seed=args.seed, data_seed=args.seed,
        num_train_epochs=ft["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=ft["per_device_batch_size"],
        per_device_eval_batch_size=ft["per_device_batch_size"],
        gradient_accumulation_steps=ft["grad_accum"],
        learning_rate=ft["learning_rate"], lr_scheduler_type=ft["lr_scheduler"],
        warmup_ratio=ft["warmup_ratio"], weight_decay=ft["weight_decay"],
        max_grad_norm=ft["max_grad_norm"], optim="adamw_torch",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=ft["eval_every_steps"],
        logging_steps=10, save_strategy="no", report_to="none",
        group_by_length=True, remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=PadCollator(tok.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate()
    secs = time.time() - t0

    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    import peft
    meta = {
        "run": run, "seed": args.seed, "n_train": len(train_ds), "n_dev": len(dev_ds),
        "max_steps": args.max_steps, "global_steps": trainer.state.global_step,
        "final_dev_loss": final_eval.get("eval_loss"),
        "train_seconds": secs, "gpu_hours": secs / 3600,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "finetune_config": ft, "model_path": model_path, "revision": sm.get("revision"),
        "prompt_version": pv,
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "python": platform.python_version(), "git_commit": git_commit(), "git_dirty": git_dirty(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("global_steps", "final_dev_loss", "gpu_hours", "peak_mem_gb")}, indent=2))


if __name__ == "__main__":
    main()
