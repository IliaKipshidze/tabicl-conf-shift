#!/usr/bin/env bash
#SBATCH --job-name=tabicl-gu-smoke
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-smoke-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

CLUSTER_ROOT="${CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
REPO_DIR="${REPO_DIR:-${CLUSTER_ROOT}/tabicl-conf-shift}"
CONDA_ROOT="${CONDA_ROOT:-${CLUSTER_ROOT}/miniconda3}"
ENV_PATH="${ENV_PATH:-${CONDA_ROOT}/envs/tabicl-conf-shift}"

GRAPH_U_DATASETS="${GRAPH_U_DATASETS:-20}"
GRAPH_U_QUERY_LOCATION="${GRAPH_U_QUERY_LOCATION:-2.0}"
GRAPH_U_QUERY_SCALE="${GRAPH_U_QUERY_SCALE:-1.5}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${REPO_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

hostname
nvidia-smi
python --version
python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# The diagnostic captures the hidden U only in memory while each dataset is
# generated. It does not add U to X/y or expose it to the model.
python -u scripts/smoke_graph_u.py \
    --datasets "${GRAPH_U_DATASETS}" \
    --query-location "${GRAPH_U_QUERY_LOCATION}" \
    --query-scale "${GRAPH_U_QUERY_SCALE}"

