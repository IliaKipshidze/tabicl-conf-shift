#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-testshift-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-testshift-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-testshift-data-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-8%2
#SBATCH --export=ALL

# Generate eight new fixed-shift test banks; existing data and models are read-only.
# After all eight tasks complete, submit slurm_nano_graph_u_test_shift_evaluate.sh.
# Preview without writing files: bash <this-script> --dry-run

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_FIXED_ROOT="${NANO_FIXED_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments}"
NANO_EPOCH_SOURCE_ROOT="${NANO_EPOCH_SOURCE_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-experiments-shuffled}"
NANO_TEST_SHIFT_ROOT="${NANO_TEST_SHIFT_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-test-shift-1p5}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false
if [[ "$#" == 1 && "${1:-}" == --dry-run ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2; exit 1
fi
for NANO_NAME in NANO_CLUSTER_ROOT NANO_FIXED_ROOT NANO_EPOCH_SOURCE_ROOT NANO_TEST_SHIFT_ROOT NANO_REPO_DIR NANO_CONDA_ROOT NANO_ENV_PATH; do
    if [[ "${!NANO_NAME}" != /* ]]; then
        echo "${NANO_NAME} must be an absolute path" >&2; exit 1
    fi
done
normalize_path() {
    local NANO_COMPONENT
    local -a NANO_COMPONENTS NANO_NORMALIZED=()
    IFS=/ read -r -a NANO_COMPONENTS <<< "$1"
    for NANO_COMPONENT in "${NANO_COMPONENTS[@]}"; do
        case "${NANO_COMPONENT}" in
            ''|.) ;;
            ..) if (( ${#NANO_NORMALIZED[@]} )); then unset 'NANO_NORMALIZED[-1]'; fi ;;
            *) NANO_NORMALIZED+=("${NANO_COMPONENT}") ;;
        esac
    done
    local IFS=/
    printf '/%s' "${NANO_NORMALIZED[*]}"
}
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_TEST_SHIFT_ROOT}")"
for NANO_SOURCE in "${NANO_FIXED_ROOT}" "${NANO_EPOCH_SOURCE_ROOT}" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations"; do
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* \
        || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_TEST_SHIFT_ROOT must not overlap read-only sources: ${NANO_SOURCE}" >&2; exit 1
    fi
done

NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
for NANO_INTEGER in "${NANO_CPUS}" "${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER}" =~ ^[1-9][0-9]*$ ]]; then
        echo "CPUS and WORKERS must be positive integers" >&2; exit 1
    fi
done
if (( NANO_WORKERS > NANO_CPUS )); then
    echo "WORKERS must not exceed allocated CPUs" >&2; exit 1
fi
configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-8]$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 8" >&2; exit 1
    fi
    NANO_PAIR_ID=$(( (NANO_TASK_ID - 1) / 2 + 1 ))
    NANO_MODE=confounding
    if (( NANO_TASK_ID % 2 == 0 )); then NANO_MODE=covariate; fi
    case "${NANO_PAIR_ID}" in
        1) NANO_LOCATION=4; NANO_SCALE=1.5; NANO_TEST_LOCATION=6; NANO_TEST_SCALE=2.25 ;;
        2) NANO_LOCATION=20; NANO_SCALE=2; NANO_TEST_LOCATION=30; NANO_TEST_SCALE=3 ;;
        3) NANO_LOCATION=60; NANO_SCALE=3; NANO_TEST_LOCATION=90; NANO_TEST_SCALE=4.5 ;;
        4) NANO_LOCATION=100; NANO_SCALE=4; NANO_TEST_LOCATION=150; NANO_TEST_SCALE=6 ;;
    esac

    NANO_ARGS=(generate --cluster-root "${NANO_CLUSTER_ROOT}" --fixed-root "${NANO_FIXED_ROOT}"
        --source-root "${NANO_EPOCH_SOURCE_ROOT}" --root "${NANO_TEST_SHIFT_ROOT}"
        --task-id "${NANO_TASK_ID}" --workers "${NANO_WORKERS}")
}

print_plan() {
    echo "task=${NANO_TASK_ID} train_location=${NANO_LOCATION} train_scale=${NANO_SCALE} test_location=${NANO_TEST_LOCATION} test_scale=${NANO_TEST_SCALE} repetition=1"
    echo "Existing models=${NANO_EPOCH_SOURCE_ROOT}; new outputs=${NANO_TEST_SHIFT_ROOT}"
    echo "Generate one shifted test bank: mode=${NANO_MODE} seed=6001 tasks=3200 batches=100 batch=32 rows=150 features=5; no training/validation/IID generation"
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.epoch_test_shift "${NANO_ARGS[@]}"; printf '\n'
}
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then
        configure_task "${NANO_TASK_ID}"; print_plan
    else
        for NANO_PREVIEW_ID in {1..8}; do configure_task "${NANO_PREVIEW_ID}"; print_plan; done
    fi
    exit 0
fi
configure_task "${NANO_TASK_ID}"
print_plan
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -u -m tabicl.nano_graph_u.epoch_test_shift "${NANO_ARGS[@]}"
echo "TEST_SHIFT_GENERATION_JOB_COMPLETE task=${NANO_TASK_ID} root=${NANO_TEST_SHIFT_ROOT}"
