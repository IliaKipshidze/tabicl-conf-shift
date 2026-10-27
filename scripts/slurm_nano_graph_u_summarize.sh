#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-summary
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-summary-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-summary-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=2

# Run after all 20 winner-test jobs complete. Summarize across five repetitions.
# Submit: sbatch scripts/slurm_nano_graph_u_summarize.sh
# Use the same GRAPH_U_MODE and TEST_GRAPH_U_MODE as the test array.

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
NANO_TEST_GRAPH_U_MODE="${TEST_GRAPH_U_MODE:-${NANO_GRAPH_U_MODE}}"
case "${NANO_TEST_GRAPH_U_MODE}" in
    confounding) NANO_TEST_MODE_SUFFIX="" ;;
    covariate) NANO_TEST_MODE_SUFFIX="_covariate" ;;
    *) echo "TEST_GRAPH_U_MODE must be confounding or covariate" >&2; exit 1 ;;
esac
NANO_TRAIN_DATA_ROOT="${NANO_TRAIN_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions${NANO_MODE_SUFFIX}}"
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions${NANO_TEST_MODE_SUFFIX}}"
NANO_SEARCH_ROOT="${NANO_SEARCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search${NANO_MODE_SUFFIX}}"
NANO_RESULT_SUFFIX="${NANO_MODE_SUFFIX}"
if [[ "${NANO_GRAPH_U_MODE}" != "${NANO_TEST_GRAPH_U_MODE}" ]]; then
    NANO_RESULT_SUFFIX="_${NANO_GRAPH_U_MODE}_on_${NANO_TEST_GRAPH_U_MODE}"
fi
NANO_TEST_ROOT="${NANO_TEST_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests${NANO_RESULT_SUFFIX}}"

if [[ "${NANO_DATA_ROOT}" != /* || "${NANO_SEARCH_ROOT}" != /* || "${NANO_TEST_ROOT}" != /* || "${NANO_TRAIN_DATA_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT, NANO_SEARCH_ROOT, NANO_TEST_ROOT and NANO_TRAIN_DATA_ROOT must be absolute paths" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p "${NANO_TEST_ROOT}"
exec 9>"${NANO_TEST_ROOT}/.summary.job.lock"
if ! flock -n 9; then
    echo "Another summary job is using ${NANO_TEST_ROOT}" >&2
    exit 1
fi
NANO_SUMMARY_ARGS=(
    --data-root "${NANO_DATA_ROOT}" --search-root "${NANO_SEARCH_ROOT}"
    --result-root "${NANO_TEST_ROOT}" --graph-u-mode "${NANO_GRAPH_U_MODE}"
)
if [[ "${NANO_GRAPH_U_MODE}" != "${NANO_TEST_GRAPH_U_MODE}" ]]; then
    NANO_SUMMARY_ARGS+=(
        --test-graph-u-mode "${NANO_TEST_GRAPH_U_MODE}"
        --training-data-root "${NANO_TRAIN_DATA_ROOT}"
    )
fi
python -u -m tabicl.nano_graph_u.summarize_tests "${NANO_SUMMARY_ARGS[@]}"
echo "SUMMARY_COMPLETE results=${NANO_TEST_ROOT}/summary.csv"
