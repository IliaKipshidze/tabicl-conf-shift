#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-tune
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-tune-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-tune-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-100%2

# Five repetitions x two training conditions x ten learning rates.
# Each job trains to step 2500 and scores only its matching validation bank.
# Preview without reading data or activating Conda: bash <this-script> --dry-run
# Submit: sbatch scripts/slurm_nano_graph_u_tune.sh

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions}"
NANO_SEARCH_ROOT="${NANO_SEARCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false
NANO_LEARNING_RATES=(0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 0.002 0.003 0.004)

if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi
if [[ "${NANO_DATA_ROOT}" != /* || "${NANO_SEARCH_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT and NANO_SEARCH_ROOT must be absolute paths" >&2
    exit 1
fi

configure_trial() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 100" >&2
        exit 1
    fi
    NANO_REPETITION=$(( (NANO_TASK_ID - 1) / 20 + 1 ))
    NANO_MODEL_SEED=$(( 41 + NANO_REPETITION ))
    NANO_LR_INDEX=$(( (NANO_TASK_ID - 1) % 10 ))
    NANO_LEARNING_RATE="${NANO_LEARNING_RATES[NANO_LR_INDEX]}"
    if (( (NANO_TASK_ID - 1) % 20 < 10 )); then
        NANO_CONDITION=identity
    else
        NANO_CONDITION=shift
    fi
    NANO_LR_TAG="${NANO_LEARNING_RATE//./p}"
    NANO_CANDIDATE_DIR="${NANO_SEARCH_ROOT}/rep_${NANO_REPETITION}/${NANO_CONDITION}/lr_${NANO_LR_TAG}"
    NANO_TRIAL_ARGS=(
        trial --task-id "${NANO_TASK_ID}"
        --data-root "${NANO_DATA_ROOT}" --output-root "${NANO_SEARCH_ROOT}"
        --device cuda
    )
}

print_plan() {
    echo "trial=${NANO_TASK_ID} repetition=${NANO_REPETITION} condition=${NANO_CONDITION} model_seed=${NANO_MODEL_SEED} learning_rate=${NANO_LEARNING_RATE} steps=2500"
    echo "Candidate directory=${NANO_CANDIDATE_DIR}"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.tune "${NANO_TRIAL_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_TASK_ID}" ]]; then
        configure_trial "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..100}; do
            configure_trial "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_trial "${NANO_TASK_ID}"
print_plan
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "${NANO_CANDIDATE_DIR}"
exec 9>"${NANO_CANDIDATE_DIR}/.job.lock"
if ! flock -n 9; then
    echo "Another tuning job is using ${NANO_CANDIDATE_DIR}" >&2
    exit 1
fi
python -u -m tabicl.nano_graph_u.tune "${NANO_TRIAL_ARGS[@]}"
echo "TUNING_JOB_COMPLETE task=${NANO_TASK_ID} directory=${NANO_CANDIDATE_DIR}"
