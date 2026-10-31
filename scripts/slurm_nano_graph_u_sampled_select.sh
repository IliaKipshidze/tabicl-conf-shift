#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-sampled-select
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-select-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-select-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=2
#SBATCH --array=1-2%2

# Task 1 selects confounding winners; task 2 selects covariate winners.
# Requires every tuning result for its mode; keeps 20 winners in total.
# Preview: bash scripts/slurm_nano_graph_u_sampled_select.sh --dry-run
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
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]{0,2}$ ]] || (( NANO_TASK_ID > 2 )); then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 2" >&2
        exit 1
    fi
    if (( NANO_TASK_ID == 1 )); then NANO_MODE=confounding; else NANO_MODE=covariate; fi
    NANO_ARGS=(select --root "${NANO_EXPERIMENT_ROOT}" --task-id "${NANO_TASK_ID}")
}

print_plan() {
    echo "selection=${NANO_TASK_ID} graph_u_mode=${NANO_MODE} candidates=100 winners=10"
    echo "Winners=${NANO_EXPERIMENT_ROOT}/checkpoints/${NANO_MODE}/winners/winners.json"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_TASK_ID}" ]]; then
        configure_task "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..2}; do
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

python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
echo "SAMPLED_SELECT_COMPLETE task=${NANO_TASK_ID:-single} root=${NANO_EXPERIMENT_ROOT}"
