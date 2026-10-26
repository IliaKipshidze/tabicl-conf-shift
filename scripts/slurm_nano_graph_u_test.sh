#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-test
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-test-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-test-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-20%2

# Test the ten retained winners on both frozen test conditions. No training,
# data generation, or winner selection. Four tasks per repetition:
# identity->identity, identity->shift, shift->identity, shift->shift.
# Submit: sbatch scripts/slurm_nano_graph_u_test.sh
# Use the same GRAPH_U_MODE as generation and tuning.
# Preview all tasks: bash scripts/slurm_nano_graph_u_test.sh --dry-run

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_GRAPH_U_MODE="${GRAPH_U_MODE:-confounding}"
case "${NANO_GRAPH_U_MODE}" in
    confounding) NANO_MODE_SUFFIX="" ;;
    covariate) NANO_MODE_SUFFIX="_covariate" ;;
    *) echo "GRAPH_U_MODE must be confounding or covariate" >&2; exit 1 ;;
esac
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions${NANO_MODE_SUFFIX}}"
NANO_SEARCH_ROOT="${NANO_SEARCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search${NANO_MODE_SUFFIX}}"
NANO_TEST_ROOT="${NANO_TEST_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests${NANO_MODE_SUFFIX}}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false

if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi
if [[ "${NANO_DATA_ROOT}" != /* || "${NANO_SEARCH_ROOT}" != /* || "${NANO_TEST_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT, NANO_SEARCH_ROOT and NANO_TEST_ROOT must be absolute paths" >&2
    exit 1
fi

configure_test() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^([1-9]|1[0-9]|20)$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 20" >&2
        exit 1
    fi
    NANO_REPETITION=$(( (NANO_TASK_ID - 1) / 4 + 1 ))
    NANO_SLOT=$(( (NANO_TASK_ID - 1) % 4 ))
    NANO_TEST_SEED=$(( 2000 + NANO_REPETITION ))
    if (( NANO_SLOT < 2 )); then
        NANO_TRAIN_CONDITION=identity
    else
        NANO_TRAIN_CONDITION=shift
    fi
    if (( NANO_SLOT % 2 == 0 )); then
        NANO_TEST_CONDITION=identity
    else
        NANO_TEST_CONDITION=shift
    fi
    NANO_CHECKPOINT="${NANO_SEARCH_ROOT}/winners/rep_${NANO_REPETITION}/${NANO_TRAIN_CONDITION}/model.pt"
    NANO_DUMP="${NANO_DATA_ROOT}/rep_${NANO_REPETITION}/${NANO_TEST_CONDITION}_test_seed${NANO_TEST_SEED}_100steps_b32_r150_f5.h5"
    NANO_RESULT="${NANO_TEST_ROOT}/rep_${NANO_REPETITION}/${NANO_TRAIN_CONDITION}_on_${NANO_TEST_CONDITION}.json"
    NANO_TEST_ARGS=(
        --task-id "${NANO_TASK_ID}"
        --data-root "${NANO_DATA_ROOT}" --search-root "${NANO_SEARCH_ROOT}"
        --result-root "${NANO_TEST_ROOT}" --device cuda
        --graph-u-mode "${NANO_GRAPH_U_MODE}"
    )
}

print_plan() {
    echo "test=${NANO_TASK_ID} repetition=${NANO_REPETITION} training=${NANO_TRAIN_CONDITION} evaluation=${NANO_TEST_CONDITION} test_seed=${NANO_TEST_SEED} tables=3200"
    echo "Checkpoint=${NANO_CHECKPOINT}"
    echo "Test bank=${NANO_DUMP}"
    echo "Result=${NANO_RESULT}"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.test_winners "${NANO_TEST_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_TASK_ID}" ]]; then
        configure_test "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..20}; do
            configure_test "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_test "${NANO_TASK_ID}"
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

mkdir -p "${NANO_RESULT%/*}"
exec 9>"${NANO_RESULT}.job.lock"
if ! flock -n 9; then
    echo "Another test job is using ${NANO_RESULT}" >&2
    exit 1
fi
python -u -m tabicl.nano_graph_u.test_winners "${NANO_TEST_ARGS[@]}"
echo "TEST_JOB_COMPLETE task=${NANO_TASK_ID} result=${NANO_RESULT}"
