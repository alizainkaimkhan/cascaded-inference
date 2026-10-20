# Onboarding: Cascaded Inference Project

**Read this first.** The study asks whether QLoRA fine-tuning Qwen3-8B on MedMCQA changes how well its confidence routes questions to a frontier model (Claude Opus 5.5). The proposal is the source of truth for the design. The latest status report has the current results.

## 1. Access you need

| What | How |
|---|---|
| GitHub repo (private) | Ask Ali to add you as a collaborator on `alizainkaimkhan/cascaded-inference`. |
| LARCC account | UofL Research Computing (Service Desk ticket). Off campus, connect the UofL VPN first. Every SSH, scp or rsync connection asks for Duo approval. |
| GitHub from LARCC | Add an SSH key from your LARCC account to **your own** GitHub account (Settings → SSH keys). Don't reuse Ali's deploy key. |
| Anthropic API key | **Not needed.** All frontier answers are already cached in `results/frontier_cache.sqlite`. Only add new frontier calls if the team agrees to, because they cost money. |

## 2. Repo map

```
configs/default.yaml     every setting; configs/pinned.json pins exact model/data revisions (committed)
data/splits/             frozen item-ID lists + manifest with hashes (committed; never regenerate)
data/raw/                dataset parquet files (git-ignored; re-download with stage_assets.py)
src/cascade/             library: data, prompts, small_model, frontier, finetune, signals, metrics, cascade
scripts/                 stage_assets, make_splits, run_small, run_frontier, train_qlora, pilot_report
slurm/                   run_small.sbatch (evaluation), train_qlora.sbatch (fine-tuning)
results/                 per-item tables (git-ignored, regenerable) + frontier_cache.sqlite (committed)
checkpoints/             LoRA adapters (git-ignored)
tests/                   pytest; run before every commit
docs/                    this file
```

**Ground rules**

1. **Each model run writes one table with a row per item:** `results/<run>/<split>.parquet`. Analysis never reruns a model; it reads these tables.
2. **The splits are frozen.** `make_splits.py` refuses to overwrite them, so never pass `--force`.
3. **Compare fine-tuned runs against `base_nf4`, never `base_bf16`.** QLoRA trains on an NF4 base, so comparing against bf16 would mix in quantization effects.
4. **Never delete or rebuild `frontier_cache.sqlite`.** It is the only result that costs money to regenerate.
5. **Commit code, configs, splits, `requirements.lock`, and the cache. Never commit API keys or `results/*.parquet`.**

## 3. LARCC setup (once, about 30 minutes)

```bash
ssh <ulink>@larcc.hpc.louisville.edu

# 1. Make login shells load the environment tools (SSH logins skip ~/.bashrc without this)
cat >> ~/.bash_profile <<'EOF'
[ -f ~/.bashrc ] && . ~/.bashrc
EOF
cat >> ~/.bashrc <<'EOF'
type module &>/dev/null && module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
type conda &>/dev/null && source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=$HOME/hf_cache
export PYTHONNOUSERSITE=1
EOF
source ~/.bashrc          # confirm the module name with `module avail miniforge`

# 2. Code
git clone git@github.com:alizainkaimkhan/cascaded-inference.git ~/cascaded-inference
cd ~/cascaded-inference

# 3. Environment (exact versions from the lock file)
conda create -y -n cascade python=3.12
conda activate cascade
which python                              # MUST be ~/.conda/envs/cascade/bin/python
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0+cu128 12.8
python -m pytest -q                       # all tests pass (one is skipped without a GPU)

# 4. Data and model, at the pinned revisions (login node only; compute nodes have no internet)
python scripts/stage_assets.py            # ~16 GB; downloads the exact revisions in configs/pinned.json
                                          # and writes your model path to configs/local.json (git-ignored).
                                          # It must print "pins unchanged"; if not, stop and ask.
git status --short                        # should show nothing new to commit

# 5. Check the GPU works (about 1 minute)
srun --partition=gpu --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
  bash -c 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cascade; python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
# expect: True NVIDIA H100 NVL
```

## 4. Day-to-day

```bash
conda activate cascade && cd ~/cascaded-inference && git pull

# Evaluate a model on one split (one H100, a few minutes)
mkdir -p logs
sbatch slurm/run_small.sbatch test base_nf4 --precision nf4
sbatch slurm/run_small.sbatch test ft_s1 --precision nf4 --adapter checkpoints/ft_s1/adapter

# Fine-tune one seed (one H100, about 1–2 hours)
sbatch slurm/train_qlora.sbatch 2

# Monitor
squeue -u $USER                     # PD = waiting, R = running; QOSMaxNodePerUserLimit = you hold 2 nodes
tail -f logs/<job-name>-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode

# Small model vs frontier report for any run and split (CPU only)
python scripts/pilot_report.py --run base_nf4 --split test
```
Results tables aren't in git. To analyse runs that someone else did, re-run the evaluation job (a few minutes) or ask them to copy the `results/<run>/` folder to you.

## 5. Mistakes that already cost us time

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` after login | `~/.bashrc` isn't read by SSH login shells | The `~/.bash_profile` line in step 3.1 |
| `torch.cuda.is_available()` is False, or `libcudart.so.12` / `libcudnn.so.9` not found | A CUDA 13 torch build from PyPI, or CUDA libraries from two builds mixed in one env | Rebuild the env from `requirements.lock` with the cu128 index. Never patch NVIDIA packages by hand. |
| `which python` shows the shared miniforge path while `(cascade)` is active | The env's `bin/` is broken or missing | `conda env remove` won't work then; move the folder aside, recreate, and reinstall from the lock |
| Packages show up from `~/.local/...` | `pip install` ran outside the env (or with `--user`) | Always use `python -m pip`; keep `PYTHONNOUSERSITE=1` |
| `CondaError: Run 'conda init'` inside tmux | A new tmux window is a fresh shell | `source "$(conda info --base)/etc/profile.d/conda.sh"` then `conda activate cascade` |
| `pip freeze` lock fails to install (`@ file:///...`) | `pip freeze` records conda-built packages as local file paths | Regenerate with `python -m pip list --format=freeze > requirements.lock` |
| A job dies at `conda activate` | `set -u` in a job script | Job scripts use `set -eo pipefail` (no `-u`) |

**Cluster rules:** never run heavy work on the login node. Long interactive work goes in `tmux`; Slurm jobs keep running if you disconnect. Each user can hold at most 2 nodes. **Nothing on LARCC is backed up**, so push to GitHub. Student accounts expire at the end of the semester, so copy anything you need before then.

**Acknowledgment** (required in any paper or presentation): "This research was supported in part by the U.S. National Science Foundation (NSF) under grants OAC2430270 and OAC2322248, and the University of Louisville's Research Computing team."
