#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-pilot
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pilot-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pilot-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# Train a fresh Nano model on an already completed Graph-U HDF5 dump. This is
# intended for short seed/learning-rate pilots and never runs data generation.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_TRAIN_DUMP="${DUMP_PATH:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/shift_loc_2p0_scale_1p5_full_2500steps_b32_r150_f5/train.h5}"
NANO_STEPS="${STEPS:-250}"
NANO_BATCH_SIZE="${BATCH_SIZE:-32}"
NANO_MODEL_SEED="${MODEL_SEED:-0}"
NANO_LEARNING_RATE="${LEARNING_RATE:-0.004}"
NANO_SAVE_EVERY="${SAVE_EVERY:-50}"
NANO_LOG_EVERY="${LOG_EVERY:-25}"

for NANO_INTEGER_SETTING in \
    "STEPS=${NANO_STEPS}" \
    "BATCH_SIZE=${NANO_BATCH_SIZE}" \
    "SAVE_EVERY=${NANO_SAVE_EVERY}" \
    "LOG_EVERY=${NANO_LOG_EVERY}"; do
    NANO_SETTING_NAME="${NANO_INTEGER_SETTING%%=*}"
    NANO_SETTING_VALUE="${NANO_INTEGER_SETTING#*=}"
    if ! [[ "${NANO_SETTING_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${NANO_SETTING_NAME} must be a positive integer" >&2
        exit 1
    fi
done
if ! [[ "${NANO_MODEL_SEED}" =~ ^[0-9]+$ ]]; then
    echo "MODEL_SEED must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${NANO_LEARNING_RATE}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "LEARNING_RATE must be a positive decimal number" >&2
    exit 1
fi
if [[ "${NANO_TRAIN_DUMP}" != /* ]]; then
    echo "DUMP_PATH must be absolute so the same file is checked and trained on" >&2
    exit 1
fi
if [[ ! -f "${NANO_TRAIN_DUMP}" ]]; then
    echo "Completed Nano training dump does not exist: ${NANO_TRAIN_DUMP}" >&2
    exit 1
fi

NANO_DUMP_TAG="$(basename "$(dirname "${NANO_TRAIN_DUMP}")")"
NANO_LR_TAG="${NANO_LEARNING_RATE//-/m}"
NANO_LR_TAG="${NANO_LR_TAG//+/p}"
NANO_LR_TAG="${NANO_LR_TAG//./p}"
NANO_DEFAULT_CHECKPOINT_DIR="${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/pilots/${NANO_DUMP_TAG}/seed_${NANO_MODEL_SEED}_lr_${NANO_LR_TAG}_${NANO_STEPS}steps"
NANO_CHECKPOINT_DIR="${CHECKPOINT_DIR:-${NANO_DEFAULT_CHECKPOINT_DIR}}"
if [[ "${NANO_CHECKPOINT_DIR}" != /* ]]; then
    echo "CHECKPOINT_DIR must be absolute so its lock protects the output directory" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python - "${NANO_TRAIN_DUMP}" "${NANO_STEPS}" "${NANO_BATCH_SIZE}" "${NANO_LEARNING_RATE}" <<'PY'
import json
import math
import sys

import h5py
import schedulefree
import torch

dump_path, requested_steps, requested_batch_size, learning_rate = sys.argv[1:]
requested_steps = int(requested_steps)
requested_batch_size = int(requested_batch_size)
learning_rate = float(learning_rate)
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit("LEARNING_RATE must be finite and positive")
with h5py.File(dump_path, "r") as stream:
    if "metadata_json" not in stream.attrs or "committed_steps" not in stream.attrs:
        raise SystemExit("DUMP_PATH is not a committed Nano-Graph-U dump")
    metadata = json.loads(stream.attrs["metadata_json"])
    committed_steps = int(stream.attrs["committed_steps"])
if committed_steps != int(metadata["steps"]):
    raise SystemExit(
        f"DUMP_PATH is incomplete: {committed_steps}/{metadata['steps']} batches committed"
    )
if requested_steps > committed_steps:
    raise SystemExit(
        f"STEPS={requested_steps} exceeds {committed_steps} committed dump batches"
    )
if requested_batch_size != int(metadata["batch_size"]):
    raise SystemExit(
        f"BATCH_SIZE={requested_batch_size} differs from dump batch size "
        f"{metadata['batch_size']}"
    )

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"h5py: {h5py.__version__}; schedulefree import: OK")
print(
    f"Prior dump complete: {committed_steps} batches; "
    f"pilot will consume the first {requested_steps}"
)
PY

mkdir -p "${NANO_CHECKPOINT_DIR}"
exec 9>"${NANO_CHECKPOINT_DIR}/.job.lock"
if ! flock -n 9; then
    echo "Another Nano pilot is using ${NANO_CHECKPOINT_DIR}" >&2
    exit 1
fi

echo "Existing prior dump=${NANO_TRAIN_DUMP}"
echo "Pilot checkpoint directory=${NANO_CHECKPOINT_DIR}"
echo "Pilot steps=${NANO_STEPS} batch=${NANO_BATCH_SIZE} seed=${NANO_MODEL_SEED} learning_rate=${NANO_LEARNING_RATE}"

NANO_TRAIN_ARGS=(
    train
    --dump "${NANO_TRAIN_DUMP}"
    --checkpoint-dir "${NANO_CHECKPOINT_DIR}"
    --steps "${NANO_STEPS}"
    --batch-size "${NANO_BATCH_SIZE}"
    --seed "${NANO_MODEL_SEED}"
    --device cuda
    --learning-rate "${NANO_LEARNING_RATE}"
    --save-every "${NANO_SAVE_EVERY}"
    --log-every "${NANO_LOG_EVERY}"
)
if [[ -f "${NANO_CHECKPOINT_DIR}/latest.pt" ]]; then
    NANO_TRAIN_ARGS+=(--resume)
fi
python -u -m tabicl.nano_graph_u "${NANO_TRAIN_ARGS[@]}"
