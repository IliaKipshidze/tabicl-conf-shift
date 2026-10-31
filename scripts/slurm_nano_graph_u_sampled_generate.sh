#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-sampled-data
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-data-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-sampled-data-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=wdr
#SBATCH --cpus-per-task=8
#SBATCH --array=1-44%2

# All 44 banks: tasks 1--22 confounding, 23--44 covariate.
# Within each mode: four banks/repetition, then two shared validation banks.
# IID banks retain scale=1, location=0; shifted banks sample once per task.
# Preview: bash scripts/slurm_nano_graph_u_sampled_generate.sh --dry-run
# Retry only failed tasks: sbatch --array=<indices> <this-script>

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
NANO_CPUS="${SLURM_CPUS_PER_TASK:-8}"
NANO_WORKERS="${WORKERS:-${NANO_CPUS}}"
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

configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]{0,2}$ ]] || (( NANO_TASK_ID > 44 )); then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 44" >&2
        exit 1
    fi
    NANO_LOCAL_ID=$(( (NANO_TASK_ID - 1) % 22 + 1 ))
    if (( NANO_TASK_ID <= 22 )); then NANO_MODE=confounding; else NANO_MODE=covariate; fi
    if (( NANO_LOCAL_ID <= 20 )); then
        NANO_REPETITION=$(( (NANO_LOCAL_ID - 1) / 4 + 1 ))
        NANO_SUBDIR="rep_${NANO_REPETITION}"
        if (( (NANO_LOCAL_ID - 1) % 4 < 2 )); then
            NANO_SPLIT=train; NANO_STEPS=2500; NANO_SEED=$(( 1000 + NANO_REPETITION ))
        else
            NANO_SPLIT=test; NANO_STEPS=100; NANO_SEED=$(( 2000 + NANO_REPETITION ))
        fi
    else
        NANO_REPETITION=shared; NANO_SUBDIR=validation; NANO_SPLIT=validation
        NANO_STEPS=100; NANO_SEED=3001
    fi
    if (( NANO_LOCAL_ID % 2 == 1 )); then
        NANO_CONDITION=identity; NANO_SHIFT_MODE=fixed
    else
        NANO_CONDITION=shift; NANO_SHIFT_MODE=sampled
    fi
    NANO_DUMP="${NANO_EXPERIMENT_ROOT}/data/${NANO_MODE}/${NANO_SUBDIR}/${NANO_CONDITION}_${NANO_SPLIT}_seed${NANO_SEED}_${NANO_STEPS}steps_b32_r150_f5.h5"
    NANO_ARGS=(generate --root "${NANO_EXPERIMENT_ROOT}" --task-id "${NANO_TASK_ID}"
        --workers "${NANO_WORKERS}"
        --query-scale-mean "${SAMPLE_SCALE_MEAN:-1.5}"
        --query-scale-std "${SAMPLE_SCALE_STD:-0.1}"
        --query-scale-min "${SAMPLE_SCALE_MIN:-1.3}"
        --query-scale-max "${SAMPLE_SCALE_MAX:-1.7}"
        --query-location-mean "${SAMPLE_LOCATION_MEAN:-2.0}"
        --query-location-std "${SAMPLE_LOCATION_STD:-0.15}"
        --query-location-min "${SAMPLE_LOCATION_MIN:-1.7}"
        --query-location-max "${SAMPLE_LOCATION_MAX:-2.3}")
}

print_plan() {
    echo "bank=${NANO_TASK_ID} graph_u_mode=${NANO_MODE} repetition=${NANO_REPETITION} split=${NANO_SPLIT} condition=${NANO_CONDITION} shift_mode=${NANO_SHIFT_MODE} seed=${NANO_SEED} batches=${NANO_STEPS} tables=$(( NANO_STEPS * 32 )) workers=${NANO_WORKERS}"
    echo "Dump=${NANO_DUMP}"
    printf 'Command:'
    printf ' %q' python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
    printf '\n'
}

if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ -n "${NANO_TASK_ID}" ]]; then
        configure_task "${NANO_TASK_ID}"
        print_plan
    else
        for NANO_PREVIEW_ID in {1..44}; do
            configure_task "${NANO_PREVIEW_ID}"
            print_plan
        done
    fi
    exit 0
fi

configure_task "${NANO_TASK_ID}"
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

python -u -m tabicl.nano_graph_u.sampled "${NANO_ARGS[@]}"
echo "SAMPLED_GENERATE_COMPLETE task=${NANO_TASK_ID:-single} root=${NANO_EXPERIMENT_ROOT}"
