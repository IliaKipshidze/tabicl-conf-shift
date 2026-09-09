#!/usr/bin/env bash
# Create/update the Linux environment used for TabICL pre-training at CISPA.
# Run this once from the login node after cloning the repository.

set -euo pipefail

CLUSTER_ROOT="${CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
REPO_DIR="${REPO_DIR:-${CLUSTER_ROOT}/tabicl-conf-shift}"
CONDA_ROOT="${CONDA_ROOT:-${CLUSTER_ROOT}/miniconda3}"
ENV_PATH="${ENV_PATH:-${CONDA_ROOT}/envs/tabicl-conf-shift}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    exit 1
fi
if [[ ! -f "${REPO_DIR}/environment.cispa.yml" ]]; then
    echo "Repository not found at ${REPO_DIR}" >&2
    echo "Set REPO_DIR if the clone is stored under another name." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${REPO_DIR}"

if [[ -d "${ENV_PATH}" ]]; then
    echo "Updating existing environment: ${ENV_PATH}"
    conda env update --prefix "${ENV_PATH}" --file environment.cispa.yml
else
    echo "Creating environment: ${ENV_PATH}"
    conda env create --prefix "${ENV_PATH}" --file environment.cispa.yml
fi

conda activate "${ENV_PATH}"

# The pretrain extra supplies transformers, xgboost, and wandb. The test extra
# is small and lets us run focused checks on the cluster before a long job.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CLUSTER_ROOT}/pip-cache}"
mkdir -p "${PIP_CACHE_DIR}"
python -m pip install --upgrade pip

# CISPA's A100 nodes currently expose NVIDIA driver 535 (CUDA 12.x). Installing
# the unbounded ``torch>=2.2`` project dependency directly can select a CUDA 13
# wheel, which requires driver 580 or newer. Pin the official CUDA 12.1 build
# first; it satisfies TabICL's requirement and prevents the subsequent editable
# install from replacing it.
python -m pip install "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[pretrain,test]"

python - <<'PY'
import sys
import torch
import tabicl

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA visible on login node: {torch.cuda.is_available()}")
print(f"TabICL import: {tabicl.__file__}")
PY

echo "Environment ready: ${ENV_PATH}"
echo "CUDA availability will be checked again inside the GPU smoke job."
