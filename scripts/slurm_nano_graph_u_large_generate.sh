#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-large-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-large-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-large-data-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-12%2
#SBATCH --export=ALL

# Run the GPU memory smoke first, then generate the twelve new banks.
# Preview: bash scripts/slurm_nano_graph_u_large_generate.sh --dry-run
set -euo pipefail
NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_LARGE_ROOT="${NANO_LARGE_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-nano-large-r1000-f8}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
NANO_DRY_RUN=false
if [[ "$#" == 1 && "${1:-}" == --dry-run ]]; then NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 1; fi
for NANO_NAME in NANO_CLUSTER_ROOT NANO_LARGE_ROOT NANO_REPO_DIR NANO_CONDA_ROOT NANO_ENV_PATH; do
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
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_LARGE_ROOT}")"
for NANO_SOURCE in "${NANO_REPO_DIR}" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-experiments-shuffled" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-test-shift-1p5"; do
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_LARGE_ROOT must not overlap existing source roots: ${NANO_SOURCE}" >&2; exit 1
    fi
done
for NANO_INTEGER in "${NANO_CPUS}" "${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER}" =~ ^[1-9][0-9]*$ ]]; then echo "CPUS and WORKERS must be positive integers" >&2; exit 1; fi
done
if (( NANO_WORKERS > NANO_CPUS )); then echo "WORKERS must not exceed allocated CPUs" >&2; exit 1; fi
configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^([1-9]|1[0-2])$ ]]; then echo "SLURM_ARRAY_TASK_ID must be from 1 to 12" >&2; exit 1; fi
    local NANO_SETTING=$(( (NANO_TASK_ID - 1) / 3 ))
    NANO_MODE=confounding; NANO_CONDITION=identity; NANO_LOCATION=0; NANO_SCALE=1
    if (( NANO_SETTING >= 2 )); then NANO_MODE=covariate; fi
    if (( NANO_SETTING % 2 == 1 )); then NANO_CONDITION=shift; NANO_LOCATION=2; NANO_SCALE=1.5; fi
    case "$(( (NANO_TASK_ID - 1) % 3 ))" in
        0) NANO_SPLIT=train; NANO_TASKS=500000; NANO_STEPS=15625; NANO_SEED=1001 ;;
        1) NANO_SPLIT=validation; NANO_TASKS=32000; NANO_STEPS=1000; NANO_SEED=3001 ;;
        2) NANO_SPLIT=test; NANO_TASKS=32000; NANO_STEPS=1000; NANO_SEED=2001 ;;
    esac
    NANO_ARGS=(generate --root "${NANO_LARGE_ROOT}" --task-id "${NANO_TASK_ID}" --workers "${NANO_WORKERS}")
}
print_plan() {
    echo "task=${NANO_TASK_ID} mode=${NANO_MODE} condition=${NANO_CONDITION} split=${NANO_SPLIT} tasks=${NANO_TASKS} batches=${NANO_STEPS} seed=${NANO_SEED} location=${NANO_LOCATION} scale=${NANO_SCALE}"
    echo "batch=32 rows=1000 features=8 hidden_U=true repetition=1; output=${NANO_LARGE_ROOT}"
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.large_experiment "${NANO_ARGS[@]}"; printf '\n'
}
echo "Run the GPU memory smoke before full generation. Twelve banks occupy approximately 90.24 GB with the current HDF5 format."
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then configure_task "${SLURM_ARRAY_TASK_ID}"; print_plan
    else for NANO_PREVIEW_ID in {1..12}; do configure_task "${NANO_PREVIEW_ID}"; print_plan; done; fi
    exit 0
fi
configure_task "${SLURM_ARRAY_TASK_ID:-}"
print_plan
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m tabicl.nano_graph_u.large_experiment "${NANO_ARGS[@]}"
echo "LARGE_GENERATION_JOB_COMPLETE task=${NANO_TASK_ID} root=${NANO_LARGE_ROOT}"
