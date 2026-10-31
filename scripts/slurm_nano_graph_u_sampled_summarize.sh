#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-sampled-summary
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-summary-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-summary-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=2

# Requires all 80 final tests. Writes 16 summary cells, each across five
# repetitions, plus four per-direction summaries and per-repetition results.
# Preview: bash scripts/slurm_nano_graph_u_sampled_summarize.sh --dry-run

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
NANO_ARGS=(summarize --root "${NANO_EXPERIMENT_ROOT}")
echo "Summary=${NANO_EXPERIMENT_ROOT}/results/summary.csv"
printf 'Command:'
printf ' %q' python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
printf '\n'
if [[ "${NANO_DRY_RUN}" == true ]]; then exit 0; fi

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
echo "SAMPLED_SUMMARIZE_COMPLETE task=${NANO_TASK_ID:-single} root=${NANO_EXPERIMENT_ROOT}"
