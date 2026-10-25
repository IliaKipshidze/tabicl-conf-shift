#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-summary
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-summary-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-summary-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=2

# Run after all 20 winner-test jobs complete. Summarize across five repetitions.
# Submit: sbatch scripts/slurm_nano_graph_u_summarize.sh

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions}"
NANO_SEARCH_ROOT="${NANO_SEARCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search}"
NANO_TEST_ROOT="${NANO_TEST_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests}"

if [[ "${NANO_DATA_ROOT}" != /* || "${NANO_SEARCH_ROOT}" != /* || "${NANO_TEST_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT, NANO_SEARCH_ROOT and NANO_TEST_ROOT must be absolute paths" >&2
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
python -u -m tabicl.nano_graph_u.summarize_tests \
    --data-root "${NANO_DATA_ROOT}" --search-root "${NANO_SEARCH_ROOT}" \
    --result-root "${NANO_TEST_ROOT}"
echo "SUMMARY_COMPLETE results=${NANO_TEST_ROOT}/summary.csv"
