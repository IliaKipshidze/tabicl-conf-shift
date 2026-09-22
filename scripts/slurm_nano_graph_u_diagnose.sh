#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-diag
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-diag-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-diag-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Read-only diagnostics for the completed shifted Nano-Graph-U experiment.
# This job does not edit the training/evaluation dumps or saved checkpoints.
# It scores several checkpoints on the same frozen eval banks, compares an
# ExtraTrees support-to-query baseline, and overfits one training table using
# a fresh model. Each report is written once under a job-specific directory.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_TRAIN_TAG="${TRAIN_TAG:-shift_loc_2p0_scale_1p5_full_2500steps_b32_r150_f5}"
NANO_CKPT_DIR="${CHECKPOINT_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/${NANO_TRAIN_TAG}}"
NANO_TRAIN_DUMP="${TRAIN_DUMP:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/${NANO_TRAIN_TAG}/train.h5}"
NANO_SHIFT_EVAL_DUMP="${SHIFT_EVAL_DUMP:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/evaluation/loc_2p0_scale_1p5_seed424242_100steps_b32/tasks.h5}"
NANO_IDENTITY_EVAL_DUMP="${IDENTITY_EVAL_DUMP:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/evaluation/loc_0p0_scale_1p0_seed424242_100steps_b32/tasks.h5}"
NANO_DIAG_MAX_TASKS="${MAX_TASKS:-128}"
NANO_OVERFIT_STEPS="${OVERFIT_STEPS:-1000}"
NANO_OVERFIT_TABLE_INDEX="${OVERFIT_TABLE_INDEX:-0}"
NANO_DIAG_OUTPUT_DIR="${OUTPUT_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/diagnostics/job-${SLURM_JOB_ID}}"

for NANO_REQUIRED_FILE in \
    "${NANO_CKPT_DIR}/step-250.pt" \
    "${NANO_CKPT_DIR}/step-1000.pt" \
    "${NANO_CKPT_DIR}/step-2500.pt" \
    "${NANO_TRAIN_DUMP}" \
    "${NANO_SHIFT_EVAL_DUMP}" \
    "${NANO_IDENTITY_EVAL_DUMP}"; do
    if [[ ! -f "${NANO_REQUIRED_FILE}" ]]; then
        echo "Required Nano diagnostic input does not exist: ${NANO_REQUIRED_FILE}" >&2
        exit 1
    fi
done

# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python - <<'PY'
import h5py
import schedulefree
import sklearn
import torch

print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"h5py: {h5py.__version__}; sklearn: {sklearn.__version__}; schedulefree import: OK")
PY

mkdir -p "${NANO_DIAG_OUTPUT_DIR}"
echo "Nano diagnostic reports: ${NANO_DIAG_OUTPUT_DIR}"
echo "Comparing checkpoints on ${NANO_DIAG_MAX_TASKS} frozen tables per condition"

NANO_CHECKPOINT_ARGS=(
    --checkpoint "${NANO_CKPT_DIR}/step-250.pt"
    --checkpoint "${NANO_CKPT_DIR}/step-1000.pt"
    --checkpoint "${NANO_CKPT_DIR}/step-2500.pt"
)

python -u -m tabicl.nano_graph_u.diagnose \
    "${NANO_CHECKPOINT_ARGS[@]}" \
    --dump "${NANO_SHIFT_EVAL_DUMP}" \
    --max-tasks "${NANO_DIAG_MAX_TASKS}" \
    --device cuda \
    --output "${NANO_DIAG_OUTPUT_DIR}/shifted_eval.json"

python -u -m tabicl.nano_graph_u.diagnose \
    "${NANO_CHECKPOINT_ARGS[@]}" \
    --dump "${NANO_IDENTITY_EVAL_DUMP}" \
    --max-tasks "${NANO_DIAG_MAX_TASKS}" \
    --device cuda \
    --output "${NANO_DIAG_OUTPUT_DIR}/identity_eval.json"

echo "Overfitting training table ${NANO_OVERFIT_TABLE_INDEX} for ${NANO_OVERFIT_STEPS} updates"
python -u -m tabicl.nano_graph_u.overfit \
    --dump "${NANO_TRAIN_DUMP}" \
    --table-index "${NANO_OVERFIT_TABLE_INDEX}" \
    --steps "${NANO_OVERFIT_STEPS}" \
    --device cuda \
    --output "${NANO_DIAG_OUTPUT_DIR}/fixed_table_overfit.json"

echo "Nano diagnostics completed: ${NANO_DIAG_OUTPUT_DIR}"
