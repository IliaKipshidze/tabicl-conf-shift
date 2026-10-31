#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-sampled-tune
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-tune-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-tune-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-200%2

# Tasks 1--100 confounding, 101--200 covariate: five repetitions,
# two conditions, ten learning rates. Each fit uses 2500 updates.
# Matching-validation AUROC selects learning rates later; test banks are unused.
# Preview: bash scripts/slurm_nano_graph_u_sampled_tune.sh --dry-run
# Retry only failed tasks: sbatch --array=<indices> <this-script>

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_EXPERIMENT_ROOT="${NANO_EXPERIMENT_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-sampled-experiment}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false
if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi
if [[ "${NANO_EXPERIMENT_ROOT}" != /* ]]; then
    echo "NANO_EXPERIMENT_ROOT must be an absolute path" >&2
    exit 1
fi

configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]{0,2}$ ]] || (( NANO_TASK_ID > 200 )); then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 200" >&2
        exit 1
    fi
    NANO_LOCAL_ID=$(( (NANO_TASK_ID - 1) % 100 + 1 ))
    if (( NANO_TASK_ID <= 100 )); then NANO_MODE=confounding; else NANO_MODE=covariate; fi
    NANO_REPETITION=$(( (NANO_LOCAL_ID - 1) / 20 + 1 ))
    NANO_LR_INDEX=$(( (NANO_LOCAL_ID - 1) % 10 ))
    NANO_LEARNING_RATES=(0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 0.002 0.003 0.004)
    NANO_LEARNING_RATE="${NANO_LEARNING_RATES[NANO_LR_INDEX]}"
    if (( (NANO_LOCAL_ID - 1) % 20 < 10 )); then NANO_CONDITION=identity; else NANO_CONDITION=shift; fi
    NANO_CANDIDATE="${NANO_EXPERIMENT_ROOT}/checkpoints/${NANO_MODE}/rep_${NANO_REPETITION}/${NANO_CONDITION}/lr_${NANO_LEARNING_RATE//./p}"
    NANO_ARGS=(tune --root "${NANO_EXPERIMENT_ROOT}" --task-id "${NANO_TASK_ID}" --device cuda)
}

print_plan() {
    echo "trial=${NANO_TASK_ID} local_trial=${NANO_LOCAL_ID} graph_u_mode=${NANO_MODE} repetition=${NANO_REPETITION} condition=${NANO_CONDITION} model_seed=$(( 41 + NANO_REPETITION )) learning_rate=${NANO_LEARNING_RATE} steps=2500"
    echo "Candidate directory=${NANO_CANDIDATE}"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_TASK_ID}" ]]; then
        configure_task "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..200}; do
            configure_task "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_task "${NANO_TASK_ID}"
print_plan

# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
echo "SAMPLED_TUNE_COMPLETE task=${NANO_TASK_ID:-single} root=${NANO_EXPERIMENT_ROOT}"
