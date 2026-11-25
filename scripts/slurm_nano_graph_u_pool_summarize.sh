#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-pool-summary
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-summary-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-summary-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=1

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

NANO_ARGS=(summarize --root "${NANO_POOL_ROOT}")
echo "summary=all winners=30 test_comparisons=180 test_cells=36 repetitions=5 training_rows=30 training_cells=6"
echo "Results=${NANO_POOL_ROOT}/results/{summary.csv,per_repetition.csv,training_summary.csv,training_per_repetition.csv,winners.csv,summary.json}"
echo "Requires all frozen-winner evaluations; reports mean and sample SD across five repetitions."
printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"; printf '\n'
if [[ "${NANO_DRY_RUN}" == true ]]; then exit 0; fi
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"
echo "POOL_SUMMARIZE_COMPLETE root=${NANO_POOL_ROOT}"
