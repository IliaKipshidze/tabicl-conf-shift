#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-select
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-select-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-select-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=2

# Run after all 100 tuning jobs finish. Requires every candidate result and
# retains ten winning checkpoints, using matching-validation AUROC only.
# Submit: sbatch scripts/slurm_nano_graph_u_select.sh

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions}"
NANO_SEARCH_ROOT="${NANO_SEARCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search}"

if [[ "${NANO_DATA_ROOT}" != /* || "${NANO_SEARCH_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT and NANO_SEARCH_ROOT must be absolute paths" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Selection only reads completed candidate outputs; it never reads test data.
mkdir -p "${NANO_SEARCH_ROOT}"
exec 9>"${NANO_SEARCH_ROOT}/.selection.job.lock"
if ! flock -n 9; then
    echo "Another winner-selection job is using ${NANO_SEARCH_ROOT}" >&2
    exit 1
fi
python -u -m tabicl.nano_graph_u.select \
    --data-root "${NANO_DATA_ROOT}" --output-root "${NANO_SEARCH_ROOT}"
echo "SELECTION_COMPLETE winners=${NANO_SEARCH_ROOT}/winners/winners.json"
