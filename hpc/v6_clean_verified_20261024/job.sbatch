#!/bin/bash
#SBATCH --job-name=lltc4j-v6-clean-staged
#SBATCH --account=staruser
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
test "${LLTC4J_NO_ADDITIONAL_CHARGE_VERIFIED:-no}" = yes
export HF_HOME=/users/mhasan29/hf_cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=20261024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /users/mhasan29/projects/lltc4j
run="runs/v6-clean-${SLURM_JOB_ID}"
mkdir "$run"
.venv/bin/python -m pip freeze > "$run/packages.txt"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv > "$run/gpu.csv"
export LLTC4J_MODEL_CACHE="/tmp/lltc4j-v6-clean-${SLURM_JOB_ID}"
.venv/bin/python -u hpc/v6_clean_verified_20261024/stage.py --destination "$LLTC4J_MODEL_CACHE" --record "$run/staging_manifest.json"
# Smoke validation uses only the first two existing development hunks, no labels.
.venv/bin/python -u hpc/v6_clean_verified_20261024/predict.py predict --model qwen3_14b --limit 2 --inputs data/prepared_v6_dev/inputs.jsonl --output "$run/smoke/qwen3_14b"
for repeat in 1 2; do
 for model in qwen25 qwen3 qwen25_14b qwen3_14b; do
  .venv/bin/python -u hpc/v6_clean_verified_20261024/predict.py predict --model "$model" --inputs data/prepared_v6_dev/inputs.jsonl --output "$run/repeat_${repeat}/$model"
 done
done
