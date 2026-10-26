#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-data-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-22%2

# Generate only: no model training or evaluation.
# Submit: sbatch scripts/slurm_nano_graph_u_generate.sh
# Feature-only U construction and its identity control:
# sbatch --export=ALL,GRAPH_U_MODE=covariate scripts/slurm_nano_graph_u_generate.sh
# Preview all banks: bash scripts/slurm_nano_graph_u_generate.sh --dry-run
# Array tasks 1--20 are four banks per repetition:
#   identity train, shift train, identity test, shift test.
# Tasks 21/22 are the shared identity/shift validation banks.
# Resubmit a subset with, e.g., sbatch --array=5-8%2 <this-script>.

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
NANO_DATA_ROOT="${NANO_DATA_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions${NANO_MODE_SUFFIX}}"
NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
NANO_BANK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false

if [[ "$#" == 1 && "${1:-}" == "--dry-run" ]]; then
    NANO_DRY_RUN=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi

for NANO_INTEGER_SETTING in "SLURM_CPUS_PER_TASK=${NANO_CPUS}" "WORKERS=${NANO_WORKERS}"; do
    if ! [[ "${NANO_INTEGER_SETTING#*=}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${NANO_INTEGER_SETTING%%=*} must be a positive integer" >&2
        exit 1
    fi
done
if (( NANO_WORKERS > NANO_CPUS )); then
    echo "WORKERS must not exceed the allocated SLURM_CPUS_PER_TASK=${NANO_CPUS}" >&2
    exit 1
fi
if [[ "${NANO_DATA_ROOT}" != /* ]]; then
    echo "NANO_DATA_ROOT must be an absolute path" >&2
    exit 1
fi

configure_bank() {
    NANO_BANK_ID="$1"
    if ! [[ "${NANO_BANK_ID}" =~ ^([1-9]|1[0-9]|2[0-2])$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 22" >&2
        exit 1
    fi
    if (( NANO_BANK_ID <= 20 )); then
        NANO_REPETITION=$(( (NANO_BANK_ID - 1) / 4 + 1 ))
        NANO_SUBDIR="rep_${NANO_REPETITION}"
        if (( (NANO_BANK_ID - 1) % 4 < 2 )); then
            NANO_SPLIT=train
            NANO_STEPS=2500
            NANO_SEED=$(( 1000 + NANO_REPETITION ))
        else
            NANO_SPLIT=test
            NANO_STEPS=100
            NANO_SEED=$(( 2000 + NANO_REPETITION ))
        fi
    else
        NANO_REPETITION=shared
        NANO_SUBDIR=validation
        NANO_SPLIT=validation
        NANO_STEPS=100
        NANO_SEED=3001
    fi
    if (( NANO_BANK_ID % 2 == 1 )); then
        NANO_CONDITION=identity
        NANO_LOCATION=0.0
        NANO_SCALE=1.0
    else
        NANO_CONDITION=shift
        NANO_LOCATION=2.0
        NANO_SCALE=1.5
    fi
    NANO_DUMP="${NANO_DATA_ROOT}/${NANO_SUBDIR}/${NANO_CONDITION}_${NANO_SPLIT}_seed${NANO_SEED}_${NANO_STEPS}steps_b32_r150_f5.h5"
    NANO_GENERATE_ARGS=(
        generate --dump "${NANO_DUMP}"
        --steps "${NANO_STEPS}" --batch-size 32
        --rows 150 --features 5 --seed "${NANO_SEED}"
        --n-jobs 1 --workers "${NANO_WORKERS}"
        --min-train-size 0.3 --max-train-size 0.9
        --structure-mode add_root --max-attempts 1000
        --graph-u-mode "${NANO_GRAPH_U_MODE}"
        --query-location "${NANO_LOCATION}" --query-scale "${NANO_SCALE}"
    )
}

print_plan() {
    echo "bank=${NANO_BANK_ID} repetition=${NANO_REPETITION} split=${NANO_SPLIT} condition=${NANO_CONDITION} graph_u_mode=${NANO_GRAPH_U_MODE} seed=${NANO_SEED} batches=${NANO_STEPS} tables=$(( NANO_STEPS * 32 )) workers=${NANO_WORKERS}"
    echo "Dump=${NANO_DUMP}"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u "${NANO_GENERATE_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_BANK_ID}" ]]; then
        configure_bank "${NANO_BANK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..22}; do
            configure_bank "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_bank "${NANO_BANK_ID}"
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

mkdir -p "${NANO_DUMP%/*}"
exec 9>"${NANO_DUMP}.job.lock"
if ! flock -n 9; then
    echo "Another generation job is using ${NANO_DUMP}" >&2
    exit 1
fi
if [[ -f "${NANO_DUMP}" ]]; then
    echo "Existing bank found; validating and resuming any uncommitted batches"
    NANO_GENERATE_ARGS+=(--resume)
fi
python -u -m tabicl.nano_graph_u "${NANO_GENERATE_ARGS[@]}"
echo "GENERATION_COMPLETE bank=${NANO_BANK_ID} tables=$(( NANO_STEPS * 32 )) dump=${NANO_DUMP}"
