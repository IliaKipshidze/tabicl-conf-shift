#!/usr/bin/env bash
#SBATCH --job-name=nano-graph-u
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-graph-u-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-graph-u-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Separate paper-scale Nano-Graph-U experiment. Does not invoke tabicl.train or
# use its checkpoint directory. The safe default is a tiny end-to-end smoke.
# Submit RUN_MODE=full,CONDITION=shift|identity for the 2,500-step paper recipe.
# Generation is CPU-only but shares this known-working GPU allocation; the GPU
# will be idle until pre-generation finishes. Both phases resume on resubmit.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_RUN_MODE="${RUN_MODE:-smoke}"
NANO_CONDITION="${CONDITION:-shift}"
NANO_DATA_SEED="${DATA_SEED:-42}"
NANO_MODEL_SEED="${MODEL_SEED:-42}"
NANO_LEARNING_RATE="${LEARNING_RATE:-0.004}"

case "${NANO_RUN_MODE}" in
    smoke)
        NANO_STEPS="${STEPS:-2}"
        NANO_BATCH_SIZE="${BATCH_SIZE:-2}"
        NANO_ROWS="${ROWS:-64}"
        NANO_FEATURES="${FEATURES:-3}"
        NANO_SAVE_EVERY="${SAVE_EVERY:-1}"
        NANO_WORKERS="${WORKERS:-1}"
        ;;
    full)
        NANO_STEPS="${STEPS:-2500}"
        NANO_BATCH_SIZE="${BATCH_SIZE:-32}"
        NANO_ROWS="${ROWS:-150}"
        NANO_FEATURES="${FEATURES:-5}"
        NANO_SAVE_EVERY="${SAVE_EVERY:-250}"
        NANO_WORKERS="${WORKERS:-8}"
        ;;
    *)
        echo "RUN_MODE must be smoke or full" >&2
        exit 1
        ;;
esac

case "${NANO_CONDITION}" in
    identity)
        NANO_LOCATION=0.0
        NANO_SCALE=1.0
        ;;
    shift)
        NANO_LOCATION="${QUERY_LOCATION:-2.0}"
        NANO_SCALE="${QUERY_SCALE:-1.5}"
        ;;
    *)
        echo "CONDITION must be identity or shift" >&2
        exit 1
        ;;
esac

if ! [[ "${NANO_LOCATION}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] ||
   ! [[ "${NANO_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "QUERY_LOCATION and QUERY_SCALE must be decimal numbers" >&2
    exit 1
fi
if ! [[ "${NANO_MODEL_SEED}" =~ ^[0-9]+$ ]]; then
    echo "MODEL_SEED must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${NANO_LEARNING_RATE}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "LEARNING_RATE must be a positive decimal number" >&2
    exit 1
fi

NANO_SHIFT_TAG="loc_${NANO_LOCATION}_scale_${NANO_SCALE}"
NANO_SHIFT_TAG="${NANO_SHIFT_TAG//-/m}"
NANO_SHIFT_TAG="${NANO_SHIFT_TAG//./p}"
NANO_RUN_TAG="${NANO_CONDITION}_${NANO_SHIFT_TAG}_${NANO_RUN_MODE}_${NANO_STEPS}steps_b${NANO_BATCH_SIZE}_r${NANO_ROWS}_f${NANO_FEATURES}"
NANO_DUMP="${DUMP_PATH:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/${NANO_RUN_TAG}/train.h5}"
NANO_LEARNING_RATE_TAG="${NANO_LEARNING_RATE//-/m}"
NANO_LEARNING_RATE_TAG="${NANO_LEARNING_RATE_TAG//+/p}"
NANO_LEARNING_RATE_TAG="${NANO_LEARNING_RATE_TAG//./p}"
NANO_LEGACY_CHECKPOINT_DIR="${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/${NANO_RUN_TAG}"
if [[ "${NANO_MODEL_SEED}" == "42" && "${NANO_LEARNING_RATE}" == "0.004" ]]; then
    # Preserve the original paper-recipe path so existing runs still resume.
    NANO_DEFAULT_CHECKPOINT_DIR="${NANO_LEGACY_CHECKPOINT_DIR}"
else
    # A different optimizer configuration must never resume the legacy model.
    NANO_DEFAULT_CHECKPOINT_DIR="${NANO_LEGACY_CHECKPOINT_DIR}/models/seed_${NANO_MODEL_SEED}_lr_${NANO_LEARNING_RATE_TAG}"
fi
NANO_CHECKPOINT_DIR="${CHECKPOINT_DIR:-${NANO_DEFAULT_CHECKPOINT_DIR}}"

# A repeated submission must not write the same dump or checkpoint concurrently.
mkdir -p "${NANO_DUMP%/*}" "${NANO_CHECKPOINT_DIR}"
exec 9>"${NANO_DUMP}.job.lock"
if ! flock -n 9; then
    echo "Another Nano job is using prior dump ${NANO_DUMP}" >&2
    exit 1
fi
exec 8>"${NANO_CHECKPOINT_DIR}/.job.lock"
if ! flock -n 8; then
    echo "Another Nano job is using checkpoints ${NANO_CHECKPOINT_DIR}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python - "${NANO_LEARNING_RATE}" <<'PY'
import h5py
import math
import schedulefree
import sys
import torch

learning_rate = float(sys.argv[1])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit("LEARNING_RATE must be finite and positive")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"h5py: {h5py.__version__}; schedulefree import: OK")
PY

echo "Nano mode=${NANO_RUN_MODE} condition=${NANO_CONDITION} location=${NANO_LOCATION} scale=${NANO_SCALE}"
echo "Model seed=${NANO_MODEL_SEED} learning rate=${NANO_LEARNING_RATE}"
echo "Prior dump=${NANO_DUMP}"
echo "Checkpoint directory=${NANO_CHECKPOINT_DIR}"

NANO_GENERATE_ARGS=(
    generate --dump "${NANO_DUMP}"
    --steps "${NANO_STEPS}" --batch-size "${NANO_BATCH_SIZE}"
    --rows "${NANO_ROWS}" --features "${NANO_FEATURES}"
    --seed "${NANO_DATA_SEED}" --n-jobs 1 --workers "${NANO_WORKERS}"
    --query-location "${NANO_LOCATION}" --query-scale "${NANO_SCALE}"
)
if [[ -f "${NANO_DUMP}" ]]; then
    NANO_GENERATE_ARGS+=(--resume)
fi
python -u -m tabicl.nano_graph_u "${NANO_GENERATE_ARGS[@]}"

NANO_TRAIN_ARGS=(
    train --dump "${NANO_DUMP}" --checkpoint-dir "${NANO_CHECKPOINT_DIR}"
    --steps "${NANO_STEPS}" --batch-size "${NANO_BATCH_SIZE}"
    --seed "${NANO_MODEL_SEED}" --device cuda
    --learning-rate "${NANO_LEARNING_RATE}"
    --save-every "${NANO_SAVE_EVERY}" --log-every 25
)
if [[ -f "${NANO_CHECKPOINT_DIR}/latest.pt" ]]; then
    NANO_TRAIN_ARGS+=(--resume)
fi
python -u -m tabicl.nano_graph_u "${NANO_TRAIN_ARGS[@]}"
