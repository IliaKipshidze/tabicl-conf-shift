#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-eval
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-eval-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-eval-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Evaluate any Nano checkpoint on a frozen, independently seeded Graph-U bank.
# Run once per evaluation condition and checkpoint. The same eval dump path is
# reused for all checkpoints, so their task sets are identical.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_EVAL_CHECKPOINT="${CHECKPOINT:-}"
NANO_EVAL_CONDITION="${EVAL_CONDITION:-identity}"
NANO_EVAL_SEED="${EVAL_SEED:-424242}"
NANO_EVAL_STEPS="${EVAL_STEPS:-100}"
NANO_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NANO_EVAL_WORKERS="${EVAL_WORKERS:-8}"

if [[ -z "${NANO_EVAL_CHECKPOINT}" ]]; then
    echo "Set CHECKPOINT to a Nano-Graph-U latest.pt or step-*.pt file" >&2
    exit 1
fi
case "${NANO_EVAL_CONDITION}" in
    identity)
        NANO_EVAL_LOCATION=0.0
        NANO_EVAL_SCALE=1.0
        ;;
    shift)
        NANO_EVAL_LOCATION="${EVAL_QUERY_LOCATION:-2.0}"
        NANO_EVAL_SCALE="${EVAL_QUERY_SCALE:-1.5}"
        ;;
    *)
        echo "EVAL_CONDITION must be identity or shift" >&2
        exit 1
        ;;
esac
if ! [[ "${NANO_EVAL_LOCATION}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] ||
   ! [[ "${NANO_EVAL_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "EVAL_QUERY_LOCATION and EVAL_QUERY_SCALE must be decimal numbers" >&2
    exit 1
fi

NANO_EVAL_TAG="loc_${NANO_EVAL_LOCATION}_scale_${NANO_EVAL_SCALE}"
NANO_EVAL_TAG="${NANO_EVAL_TAG//-/m}"
NANO_EVAL_TAG="${NANO_EVAL_TAG//./p}"
NANO_EVAL_DUMP="${EVAL_DUMP:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/evaluation/${NANO_EVAL_TAG}_seed${NANO_EVAL_SEED}_${NANO_EVAL_STEPS}steps_b${NANO_EVAL_BATCH_SIZE}/tasks.h5}"
NANO_EVAL_RESULT="${EVAL_RESULT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/eval-${SLURM_JOB_ID}.json}"

# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python - <<'PY'
import h5py
import schedulefree
import torch

print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"h5py: {h5py.__version__}; schedulefree import: OK")
PY

# The lock prevents two model-evaluation jobs from writing the shared bank at
# the same time. Once generated, each job merely validates the completed bank.
mkdir -p "$(dirname "${NANO_EVAL_DUMP}")" "$(dirname "${NANO_EVAL_RESULT}")"
exec 9>"${NANO_EVAL_DUMP}.lock"
flock -x 9
NANO_GENERATE_ARGS=(
    generate --dump "${NANO_EVAL_DUMP}"
    --steps "${NANO_EVAL_STEPS}" --batch-size "${NANO_EVAL_BATCH_SIZE}"
    --rows 150 --features 5 --seed "${NANO_EVAL_SEED}" --n-jobs 1
    --workers "${NANO_EVAL_WORKERS}"
    --query-location "${NANO_EVAL_LOCATION}" --query-scale "${NANO_EVAL_SCALE}"
)
if [[ -f "${NANO_EVAL_DUMP}" ]]; then
    NANO_GENERATE_ARGS+=(--resume)
fi
python -u -m tabicl.nano_graph_u "${NANO_GENERATE_ARGS[@]}"

# Keep the lock while evaluating: another job's HDF5 r+ resume check must not
# open the same dump while this process is reading it.

python -u -m tabicl.nano_graph_u.evaluate \
    --checkpoint "${NANO_EVAL_CHECKPOINT}" \
    --dump "${NANO_EVAL_DUMP}" \
    --batch-size "${NANO_EVAL_BATCH_SIZE}" \
    --device cuda \
    --output "${NANO_EVAL_RESULT}"
