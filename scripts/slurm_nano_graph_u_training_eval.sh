#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-train-eval
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-train-eval-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-train-eval-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-50%2
#SBATCH --export=ALL

# Evaluate the already selected fixed-sweep models on their own saved training
# banks (80,000 tasks each). This performs no training, selection, or generation.
# Tasks 1-10 evaluate the original IID models once; tasks 11-50 evaluate the
# shifted winners for the four fixed pairs. The Python controller writes the
# training summaries after all 50 distinct model/bank evaluations are complete.
# Preview all tasks without touching files: bash <this-script> --dry-run

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_FIXED_ROOT="${NANO_FIXED_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments}"
NANO_CPUS="${SLURM_CPUS_PER_TASK:-2}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false

if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi

for NANO_PATH_SETTING in \
    "NANO_CLUSTER_ROOT=${NANO_CLUSTER_ROOT}" "NANO_FIXED_ROOT=${NANO_FIXED_ROOT}" \
    "NANO_REPO_DIR=${NANO_REPO_DIR}" "NANO_CONDA_ROOT=${NANO_CONDA_ROOT}" \
    "NANO_ENV_PATH=${NANO_ENV_PATH}"; do
    if [[ "${NANO_PATH_SETTING#*=}" != /* ]]; then
        echo "${NANO_PATH_SETTING%%=*} must be an absolute path" >&2
        exit 1
    fi
done
if ! [[ "${NANO_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 1
fi

configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^([1-9]|[1-4][0-9]|50)$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 50" >&2
        exit 1
    fi
    NANO_CONDITION=identity
    NANO_PAIR_ID=
    NANO_LOCATION=
    NANO_SCALE=
    local NANO_MODEL_INDEX=$(( NANO_TASK_ID - 1 ))
    if (( NANO_TASK_ID > 10 )); then
        NANO_CONDITION=shift
        NANO_PAIR_ID=$(( (NANO_TASK_ID - 11) / 10 + 1 ))
        NANO_MODEL_INDEX=$(( (NANO_TASK_ID - 11) % 10 ))
        case "${NANO_PAIR_ID}" in
            1) NANO_LOCATION=4; NANO_SCALE=1.5 ;;
            2) NANO_LOCATION=20; NANO_SCALE=2 ;;
            3) NANO_LOCATION=60; NANO_SCALE=3 ;;
            4) NANO_LOCATION=100; NANO_SCALE=4 ;;
        esac
    fi
    NANO_MODE=confounding
    if (( NANO_MODEL_INDEX >= 5 )); then NANO_MODE=covariate; fi
    NANO_REPETITION=$(( NANO_MODEL_INDEX % 5 + 1 ))
    NANO_ARGS=(--root "${NANO_FIXED_ROOT}" --task-id "${NANO_TASK_ID}" --device cuda)
}

print_plan() {
    printf 'task=%s condition=%s mode=%s repetition=%s' \
        "${NANO_TASK_ID}" "${NANO_CONDITION}" "${NANO_MODE}" "${NANO_REPETITION}"
    if [[ "${NANO_CONDITION}" == shift ]]; then
        printf ' pair=%s location=%s scale=%s' "${NANO_PAIR_ID}" "${NANO_LOCATION}" "${NANO_SCALE}"
    else
        printf ' IID result reused for all four pairs'
    fi
    printf '; own saved training bank; tasks=80000\n'
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.training_eval "${NANO_ARGS[@]}"
    printf '\n'
}

print_outputs() {
    echo "Evaluation only: existing checkpoints and training banks are read-only; existing test CSVs are unchanged"
    echo "After all 50 evaluations: ${NANO_FIXED_ROOT}/results/training_summary.csv (16 cells)"
    echo "After all 50 evaluations: ${NANO_FIXED_ROOT}/results/training_per_repetition.csv (80 rows; IID results reused per pair)"
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then
        configure_task "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..50}; do
            configure_task "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    print_outputs
    exit 0
fi

configure_task "${NANO_TASK_ID}"
print_plan
print_outputs
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

python -u -m tabicl.nano_graph_u.training_eval "${NANO_ARGS[@]}"
echo "TRAINING_EVAL_JOB_COMPLETE task=${NANO_TASK_ID} root=${NANO_FIXED_ROOT}"
