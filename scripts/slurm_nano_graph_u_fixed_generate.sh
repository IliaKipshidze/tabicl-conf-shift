#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-fixed-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-fixed-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-fixed-data-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4%2
#SBATCH --export=ALL

# Submit: sbatch scripts/slurm_nano_graph_u_fixed_generate.sh
# Each task generates the shifted train/validation/test banks for one fixed
# (location, scale) pair, for both confounding and covariate modes.
# After this array completes, submit slurm_nano_graph_u_fixed_experiments.sh.
# Original fixed IID banks, winners, and reports are read-only inputs.
# Preview all four tasks without touching files: bash <this-script> --dry-run
# Resubmission resumes complete stages through the Python controller.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_FIXED_ROOT="${NANO_FIXED_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments}"
NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
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
for NANO_INTEGER_SETTING in "SLURM_CPUS_PER_TASK=${NANO_CPUS}" "WORKERS=${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER_SETTING#*=}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${NANO_INTEGER_SETTING%%=*} must be a positive integer" >&2
        exit 1
    fi
done

if (( NANO_WORKERS > NANO_CPUS )); then
    echo "WORKERS must not exceed allocated SLURM_CPUS_PER_TASK=${NANO_CPUS}" >&2
    exit 1
fi

# Normalize lexically with Bash builtins so dry-run does not inspect files or
# require Python. The controller also validates resolved paths before writing.
normalize_path() {
    local NANO_COMPONENT
    local -a NANO_COMPONENTS NANO_NORMALIZED=()
    IFS=/ read -r -a NANO_COMPONENTS <<< "$1"
    for NANO_COMPONENT in "${NANO_COMPONENTS[@]}"; do
        case "${NANO_COMPONENT}" in
            ''|.) ;;
            ..)
                if (( ${#NANO_NORMALIZED[@]} )); then
                    unset "NANO_NORMALIZED[$(( ${#NANO_NORMALIZED[@]} - 1 ))]"
                fi
                ;;
            *) NANO_NORMALIZED+=("${NANO_COMPONENT}") ;;
        esac
    done
    local IFS=/
    printf '/%s' "${NANO_NORMALIZED[*]}"
}

NANO_IID_BANK_ROOTS=(
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions"
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions_covariate"
)
NANO_IID_SEARCH_ROOTS=(
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search"
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search_covariate"
)
NANO_IID_RESULT_ROOTS=(
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests"
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_covariate"
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_confounding_on_covariate"
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_covariate_on_confounding"
)
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_FIXED_ROOT}")"
for NANO_SOURCE_PATH in "${NANO_IID_BANK_ROOTS[@]}" "${NANO_IID_SEARCH_ROOTS[@]}" "${NANO_IID_RESULT_ROOTS[@]}"; do
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE_PATH}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* \
        || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_FIXED_ROOT must differ from and not overlap original read-only source roots: ${NANO_SOURCE_PATH}" >&2
        exit 1
    fi
done

configure_pair() {
    NANO_TASK_ID="$1"
    case "${NANO_TASK_ID}" in
        1) NANO_LOCATION=4; NANO_SCALE=1.5; NANO_PAIR_TAG=loc_4_scale_1p5 ;;
        2) NANO_LOCATION=20; NANO_SCALE=2; NANO_PAIR_TAG=loc_20_scale_2 ;;
        3) NANO_LOCATION=60; NANO_SCALE=3; NANO_PAIR_TAG=loc_60_scale_3 ;;
        4) NANO_LOCATION=100; NANO_SCALE=4; NANO_PAIR_TAG=loc_100_scale_4 ;;
        *) echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 4" >&2; exit 1 ;;
    esac
    NANO_ARGS=(
        generate
        --cluster-root "${NANO_CLUSTER_ROOT}" --root "${NANO_FIXED_ROOT}"
        --task-id "${NANO_TASK_ID}" --workers "${NANO_WORKERS}"
    )
}

print_sources() {
    echo "Original fixed IID sources (read-only; reused without retraining):"
    local NANO_MODE_INDEX NANO_REPETITION NANO_MODE
    for NANO_MODE_INDEX in 0 1; do
        NANO_MODE=confounding
        if (( NANO_MODE_INDEX == 1 )); then NANO_MODE=covariate; fi
        echo "Original IID banks mode=${NANO_MODE} root=${NANO_IID_BANK_ROOTS[NANO_MODE_INDEX]}"
        echo "Original IID winners mode=${NANO_MODE} root=${NANO_IID_SEARCH_ROOTS[NANO_MODE_INDEX]}/winners"
        for NANO_REPETITION in {1..5}; do
            echo "Original IID model mode=${NANO_MODE} repetition=${NANO_REPETITION} path=${NANO_IID_SEARCH_ROOTS[NANO_MODE_INDEX]}/winners/rep_${NANO_REPETITION}/identity/model.pt"
        done
    done
    for NANO_SOURCE_PATH in "${NANO_IID_RESULT_ROOTS[@]}"; do
        echo "Original IID test results root=${NANO_SOURCE_PATH}"
    done
}

print_plan() {
    echo "pair=${NANO_TASK_ID} location=${NANO_LOCATION} scale=${NANO_SCALE} modes=confounding,covariate repetitions=5 learning_rates=10"
    echo "Pair output=${NANO_FIXED_ROOT}/${NANO_PAIR_TAG}"
    echo "New banks=22; generation only; original IID banks reused read-only"
    echo "Stages=generate shifted train/validation/test banks"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.fixed_experiments "${NANO_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    print_sources
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then
        configure_pair "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..4}; do
            configure_pair "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_pair "${NANO_TASK_ID}"
print_sources
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

python -u -m tabicl.nano_graph_u.fixed_experiments "${NANO_ARGS[@]}"
echo "FIXED_GENERATION_JOB_COMPLETE task=${NANO_TASK_ID} output=${NANO_FIXED_ROOT}/${NANO_PAIR_TAG}"
