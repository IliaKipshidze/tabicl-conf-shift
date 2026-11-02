#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-location-sweep
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-location-sweep-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-location-sweep-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# One sequential job: generate new test banks, evaluate frozen winners, include
# existing IID reports, then write combined CSVs. No training or model selection.
# Submit: sbatch scripts/slurm_nano_graph_u_location_sweep.sh
# Preview without reading or writing files: bash <this-script> --dry-run
# Resubmit the same script to reuse complete outputs and resume partial banks.

set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_SOURCE_ROOT="${NANO_SOURCE_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-sampled-experiment}"
NANO_LOCATION_ROOT="${NANO_LOCATION_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-location-sweep}"
NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
NANO_DRY_RUN=false

if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi
if [[ "${NANO_SOURCE_ROOT}" != /* || "${NANO_LOCATION_ROOT}" != /* ]]; then
    echo "NANO_SOURCE_ROOT and NANO_LOCATION_ROOT must be absolute paths" >&2
    exit 1
fi
if [[ "${NANO_SOURCE_ROOT%/}" == "${NANO_LOCATION_ROOT%/}" ]]; then
    echo "NANO_LOCATION_ROOT must differ from NANO_SOURCE_ROOT" >&2
    exit 1
fi
for NANO_INTEGER_SETTING in "CPUS=${NANO_CPUS}" "WORKERS=${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER_SETTING#*=}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${NANO_INTEGER_SETTING%%=*} must be a positive integer" >&2
        exit 1
    fi
done
if (( NANO_WORKERS > NANO_CPUS )); then
    echo "WORKERS must not exceed allocated SLURM_CPUS_PER_TASK=${NANO_CPUS}" >&2
    exit 1
fi

NANO_PROFILE_TEXT="${SWEEP_LOCATION_PROFILES:-0.3:0.1:0.5 1.3:1.1:1.5 1.7:1.5:1.9 1.9:1.7:2.1 2.2:2.0:2.4 4.5:4.3:4.7 10:9.8:10.2 100:85:115}"
read -r -a NANO_LOCATION_PROFILES <<< "${NANO_PROFILE_TEXT//$'\n'/ }"
if (( ${#NANO_LOCATION_PROFILES[@]} == 0 )); then
    echo "SWEEP_LOCATION_PROFILES must contain at least one mean:lower:upper profile" >&2
    exit 1
fi
NANO_ARGS=(
    --source-root "${NANO_SOURCE_ROOT}" --root "${NANO_LOCATION_ROOT}"
    --location-profiles "${NANO_LOCATION_PROFILES[@]}"
    --workers "${NANO_WORKERS}" --device cuda
)
echo "Source experiment=${NANO_SOURCE_ROOT}"
echo "Sweep output=${NANO_LOCATION_ROOT}"
echo "Location profiles=${NANO_LOCATION_PROFILES[*]}"
echo "Query scale sampling and location SD inherited from source (defaults: scale mean=1.5 SD=0.1 bounds=[1.3,1.7]; location SD=0.15)"
echo "New test banks=$(( 10 * ${#NANO_LOCATION_PROFILES[@]} )); new evaluations=$(( 40 * ${#NANO_LOCATION_PROFILES[@]} )); reused IID evaluations=40; total records=$(( 40 * ${#NANO_LOCATION_PROFILES[@]} + 40 ))"
printf 'Command:'
printf ' %q' python -u -m tabicl.nano_graph_u.location_sweep "${NANO_ARGS[@]}"
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

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python -u -m tabicl.nano_graph_u.location_sweep "${NANO_ARGS[@]}"
echo "LOCATION_SWEEP_COMPLETE results=${NANO_LOCATION_ROOT}/results/summary.csv"
