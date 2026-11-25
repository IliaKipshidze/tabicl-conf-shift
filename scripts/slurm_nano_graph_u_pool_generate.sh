#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-pool-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-data-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-66%2

# Run this stage only after all tasks in the preceding stage have completed.
# Preview without creating files: bash ${BASH_SOURCE[0]} --dry-run
set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_POOL_ROOT="${NANO_POOL_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-common-support-experiment}"
NANO_DRY_RUN=false
if [[ "$#" == 1 && "${1:-}" == --dry-run ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2; exit 1
fi
for NANO_NAME in NANO_CLUSTER_ROOT NANO_REPO_DIR NANO_CONDA_ROOT NANO_ENV_PATH NANO_POOL_ROOT; do
    if [[ "${!NANO_NAME}" != /* ]]; then echo "${NANO_NAME} must be an absolute path" >&2; exit 1; fi
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
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_POOL_ROOT}")"
for NANO_SOURCE in "${NANO_REPO_DIR}" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-sampled-experiment" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-nano-large-r1000-f8" \
    "${NANO_EXPERIMENT_ROOT:-}" "${NANO_DATA_ROOT:-}" "${NANO_SEARCH_ROOT:-}" "${NANO_TEST_ROOT:-}"; do
    [[ -n "${NANO_SOURCE}" ]] || continue
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_POOL_ROOT must not overlap existing source roots: ${NANO_SOURCE}" >&2; exit 1
    fi
done
echo "Experiment=${NANO_POOL_ROOT}"
echo "Protocol: pool resampling; original support-row order; peak=75%; shift width=0.5n; mixed equal per-task choices=uniform,0.2n,0.5n,1n; no affine a,b."
echo "Five repetitions; batch=32 rows=150 features=5; train=80000 validation=3200 test=3200 tasks; one epoch."

NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
for NANO_INTEGER_SETTING in "CPUS=${NANO_CPUS}" "WORKERS=${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER_SETTING#*=}" =~ ^[1-9][0-9]{0,5}$ ]]; then
        echo "${NANO_INTEGER_SETTING%%=*} must be a positive integer" >&2; exit 1
    fi
done
if (( NANO_WORKERS > NANO_CPUS )); then
    echo "WORKERS must not exceed allocated SLURM_CPUS_PER_TASK=${NANO_CPUS}" >&2; exit 1
fi
NANO_CONDITIONS=(uniform shift mixed)
configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]{0,2}$ ]] || (( NANO_TASK_ID > 66 )); then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 66" >&2; exit 1
    fi
    NANO_LOCAL=$(( (NANO_TASK_ID - 1) % 33 ))
    if (( NANO_TASK_ID <= 33 )); then NANO_MODE=confounding; else NANO_MODE=covariate; fi
    if (( NANO_LOCAL < 30 )); then
        NANO_REPETITION=$(( NANO_LOCAL / 6 + 1 ))
        NANO_SUBDIR="rep_${NANO_REPETITION}"
        NANO_CONDITION="${NANO_CONDITIONS[NANO_LOCAL % 3]}"
        if (( NANO_LOCAL % 6 < 3 )); then
            NANO_SPLIT=train; NANO_STEPS=2500; NANO_SEED=$(( 1000 + NANO_REPETITION ))
        else
            NANO_SPLIT=test; NANO_STEPS=100; NANO_SEED=$(( 2000 + NANO_REPETITION ))
        fi
    else
        NANO_REPETITION=shared; NANO_SUBDIR=validation; NANO_SPLIT=validation
        NANO_CONDITION="${NANO_CONDITIONS[NANO_LOCAL - 30]}"; NANO_STEPS=100; NANO_SEED=3001
    fi
    NANO_DUMP="${NANO_POOL_ROOT}/data/${NANO_MODE}/${NANO_SUBDIR}/${NANO_CONDITION}_${NANO_SPLIT}_seed${NANO_SEED}_${NANO_STEPS}steps_b32_r150_f5.h5"
    NANO_ARGS=(generate --root "${NANO_POOL_ROOT}" --task-id "${NANO_TASK_ID}" --workers "${NANO_WORKERS}")
}
print_plan() {
    echo "bank=${NANO_TASK_ID} graph_u_mode=${NANO_MODE} repetition=${NANO_REPETITION} split=${NANO_SPLIT} condition=${NANO_CONDITION} seed=${NANO_SEED} batches=${NANO_STEPS} tables=$(( NANO_STEPS * 32 )) workers=${NANO_WORKERS}"
    echo "Dump=${NANO_DUMP}"
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"; printf '\n'
}
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then configure_task "${SLURM_ARRAY_TASK_ID}"; print_plan
    else for (( NANO_PREVIEW_ID=1; NANO_PREVIEW_ID<=66; NANO_PREVIEW_ID++ )); do configure_task "${NANO_PREVIEW_ID}"; print_plan; done; fi
    exit 0
fi
configure_task "${SLURM_ARRAY_TASK_ID:-}"
print_plan
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"
echo "POOL_GENERATE_COMPLETE task=${NANO_TASK_ID} root=${NANO_POOL_ROOT}"
