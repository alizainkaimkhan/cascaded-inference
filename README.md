# Does Domain Fine-Tuning Change the Cascade?

CSE-645 project (Kaimkhani, Musa). Qwen3-8B, QLoRA-fine-tuned on MedMCQA, cascaded to a frontier LLM.
We measure how fine-tuning changes calibration, discrimination, and the cascade's cost–accuracy frontier.

## Design rule

**Every model run writes one per-item table and nothing else.** Everything downstream
(confidence signals, temperature scaling, the logistic combiner, both cascade regimes,
bootstraps, contamination strata) is offline pandas over those cached tables.

| Table | Written by | One row per |
|---|---|---|
| `results/<run>/<split>.parquet` | `run_small.py` | (item, presentation): four letter logits, letter mass, chosen option, gold |
| `results/frontier/<split>.parquet` | `run_frontier.py` | item: parsed letter, correctness, token counts |
| `results/frontier_cache.sqlite` | `run_frontier.py` | (model, item, prompt hash): raw response. **Never delete; back it up.** |

## Layout

```
configs/default.yaml     all settings; configs/pinned.json (generated) pins revisions
src/cascade/
  data.py                canonical schema, dedup, frozen splits
  prompts.py             prompt template, the 10 fixed option orders, SFT prompt/completion
  small_model.py         exact letter logits via HF transformers, with tokenization self-checks
  frontier.py            provider-agnostic API wrapper + write-once SQLite cache
  signals.py             per-item MSP and permutation agreement
  metrics.py             ECE (equal-mass), Brier, AUROC, AURC/E-AURC, bootstrap (tie-aware)
  cascade.py             budget curve, threshold regime, oracle, 2x2 table
scripts/                 stage_assets, make_splits, run_small, run_frontier, pilot_report
slurm/run_small.sbatch   one H100, offline
tests/                   pytest (synthetic data; no GPU or network)
```

## Week-1 runbook

**1. Environment (LARCC login node, once)**
```bash
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n cascade python=3.12 && conda activate cascade
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch
pip install -r requirements.txt
pip freeze > requirements.lock     # commit this
python -m pytest -q
```

**2. Stage data and weights (login node, has internet)**
```bash
export HF_HOME=$HOME/hf_cache
python scripts/stage_assets.py     # writes data/raw/*.parquet and configs/pinned.json
python scripts/make_splits.py      # writes data/splits/*.csv + manifest.json
git add configs/pinned.json data/splits && git commit -m "Pin revisions, freeze splits"
```
Read the dedup report `make_splits.py` prints. In particular, check how many validation items
were dropped and how many are `choice_type == multi`.

**3. Small model on the pilot (compute node)**
```bash
mkdir -p logs
sbatch slurm/run_small.sbatch pilot base_bf16
# then check: letter_mass_median ~ 1.0, frac_top_token_is_letter ~ 1.0, and the printed prompt
```

**4. Frontier on the pilot (laptop or login node)**

First set `frontier.provider` and `frontier.model` in `configs/pinned.json`, and record the access date there.
```bash
export ANTHROPIC_API_KEY=...       # or OPENAI_API_KEY
python scripts/run_frontier.py --split pilot --dry-run
python scripts/run_frontier.py --split pilot
```

**5. Go / no-go**
```bash
python scripts/pilot_report.py --run base_bf16 --split pilot
```

## Conventions

- MedMCQA `cop` is 0-indexed (0 = `opa`). MedQA items get IDs `medqa-test-NNNN` by row order of the pinned revision.
- Presentation 0 is the original option order; 1–9 are fixed per item ID and shared by all models.
- Dedup key defaults to normalized stem + sorted options (see `configs/default.yaml`).
  Duplicate groups whose copies disagree on the answer are dropped entirely.
- Before/after fine-tuning comparisons use the same precision. Base is scored in both bf16 and NF4,
  and the quantization effect is reported separately.

## Acknowledgment (required for any publication or presentation using LARCC)

"This research was supported in part by the U.S. National Science Foundation (NSF) under grants
OAC2430270 and OAC2322248, and the University of Louisville's Research Computing team."
