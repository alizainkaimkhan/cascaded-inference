#!/bin/bash
# QLoRA fine-tuning, one seed per job, one H100.
#   mkdir -p logs
#   sbatch slurm/train_qlora.sbatch 1                                   # full run, seed 1
#   sbatch slurm/train_qlora.sbatch 1 --max-steps 20 --run-name smoke   # ~5 min smoke test
#SBATCH --job-name=train_qlora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
SEED=$1; shift

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym   # confirm with `module avail`
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cascade

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR"
echo "python: $(which python)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train_qlora.py --seed "$SEED" "$@"
