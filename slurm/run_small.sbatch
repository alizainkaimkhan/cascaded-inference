#!/bin/bash
# Score one split with the small model on one H100 (half a GPU node).
#   mkdir -p logs
#   sbatch slurm/run_small.sbatch pilot base_bf16
#   sbatch slurm/run_small.sbatch calibration base_nf4 --precision nf4
#SBATCH --job-name=run_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
SPLIT=$1; RUN=$2; shift 2

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym   # confirm with `module avail`
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cascade

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/run_small.py --split "$SPLIT" --run "$RUN" "$@"
