#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-epochs
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-epochs-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-epochs-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-30%2
#SBATCH --export=ALL

# Existing repetition-1 banks only; no data generation. Each task tunes three
# learning rates for one model and epoch budget, then tests its selected winner.
# IID models are retrained once per budget and reused across the four fixed pairs.
# Whole saved batches are shuffled reproducibly each epoch, using the model seed.
# --training-eval-only scores the frozen winners on their own training banks.
# Preview: bash scripts/slurm_nano_graph_u_epoch_experiments.sh --dry-run
# Submit:  sbatch scripts/slurm_nano_graph_u_epoch_experiments.sh
set -euo pipefail

NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_FIXED_ROOT="${NANO_FIXED_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments}"
NANO_EPOCH_ROOT="${NANO_EPOCH_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-experiments-shuffled}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
NANO_DRY_RUN=false
NANO_TRAINING_EVAL_ONLY=false

for NANO_OPTION in "$@"; do
    case "${NANO_OPTION}" in
        --dry-run) NANO_DRY_RUN=true ;;
        --training-eval-only) NANO_TRAINING_EVAL_ONLY=true ;;
        *) echo "Usage: $0 [--dry-run] [--training-eval-only]" >&2; exit 1 ;;
    esac
done
for NANO_NAME in NANO_CLUSTER_ROOT NANO_FIXED_ROOT NANO_EPOCH_ROOT NANO_REPO_DIR NANO_CONDA_ROOT NANO_ENV_PATH; do
    if [[ "${!NANO_NAME}" != /* ]]; then
        echo "${NANO_NAME} must be an absolute path" >&2; exit 1
    fi
done
if ! [[ "${SLURM_CPUS_PER_TASK:-2}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_CPUS_PER_TASK must be a positive integer" >&2; exit 1
fi

# Check lexical paths without filesystem access; Python also checks resolved paths.
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
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_EPOCH_ROOT}")"
for NANO_SOURCE in "${NANO_FIXED_ROOT}" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations"; do
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* \
        || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_EPOCH_ROOT must not overlap read-only sources: ${NANO_SOURCE}" >&2; exit 1
    fi
done

configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^([1-9]|[12][0-9]|30)$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 30" >&2; exit 1
    fi
    local NANO_SLOT=$(( (NANO_TASK_ID - 1) % 10 ))
    case "$(( (NANO_TASK_ID - 1) / 10 ))" in
        0) NANO_EPOCHS=3 ;; 1) NANO_EPOCHS=6 ;; 2) NANO_EPOCHS=10 ;;
    esac
    NANO_CONDITION=identity
    NANO_MODE=confounding
    NANO_PAIR_TAG=shared_iid
    NANO_PAIR_ID=
    NANO_LOCATION=0
    NANO_SCALE=1
    if (( NANO_SLOT >= 2 )); then
        NANO_CONDITION=shift
        NANO_PAIR_ID=$(( (NANO_SLOT - 2) / 2 + 1 ))
        case "${NANO_PAIR_ID}" in
            1) NANO_LOCATION=4; NANO_SCALE=1.5; NANO_PAIR_TAG=loc_4_scale_1p5 ;;
            2) NANO_LOCATION=20; NANO_SCALE=2; NANO_PAIR_TAG=loc_20_scale_2 ;;
            3) NANO_LOCATION=60; NANO_SCALE=3; NANO_PAIR_TAG=loc_60_scale_3 ;;
            4) NANO_LOCATION=100; NANO_SCALE=4; NANO_PAIR_TAG=loc_100_scale_4 ;;
        esac
    fi
    if (( NANO_SLOT % 2 == 1 )); then NANO_MODE=covariate; fi
    if [[ "${NANO_CONDITION}" == identity ]]; then
        local NANO_SUFFIX=
        if [[ "${NANO_MODE}" == covariate ]]; then NANO_SUFFIX=_covariate; fi
        NANO_DATA_ROOT="${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors/nano_graph_u/repetitions${NANO_SUFFIX}"
    else
        NANO_DATA_ROOT="${NANO_FIXED_ROOT}/${NANO_PAIR_TAG}/data/${NANO_MODE}"
    fi
    NANO_ARGS=(--cluster-root "${NANO_CLUSTER_ROOT}" --fixed-root "${NANO_FIXED_ROOT}"
        --root "${NANO_EPOCH_ROOT}" --task-id "${NANO_TASK_ID}" --device cuda)
    if [[ "${NANO_TRAINING_EVAL_ONLY}" == true ]]; then
        NANO_ARGS+=(--training-eval-only)
    fi
}
print_plan() {
    echo "task=${NANO_TASK_ID} epochs=${NANO_EPOCHS} updates=$(( 2500 * NANO_EPOCHS )) mode=${NANO_MODE} condition=${NANO_CONDITION} repetition=1 pair=${NANO_PAIR_TAG} location=${NANO_LOCATION} scale=${NANO_SCALE}"
    if [[ "${NANO_TRAINING_EVAL_ONLY}" == true ]]; then
        echo "Frozen selected winner; score all 80,000 own-training tasks; no training, tuning or selection"
    else
        echo "Learning rates=0.0007,0.001,0.002; selection=matching validation AUROC at the full epoch budget"
        echo "Batch order=seeded permutation of whole batches per epoch; model_seed=42; scheme=seedsequence_model_epoch_v1"
        echo "Shared validation=${NANO_DATA_ROOT}/validation/${NANO_CONDITION}_validation_seed3001_100steps_b32_r150_f5.h5"
    fi
    echo "Training=${NANO_DATA_ROOT}/rep_1/${NANO_CONDITION}_train_seed1001_2500steps_b32_r150_f5.h5"
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.epoch_experiments "${NANO_ARGS[@]}"; printf '\n'
}

echo "Read-only inputs: original IID banks plus fixed banks at ${NANO_FIXED_ROOT}"
if [[ "${NANO_TRAINING_EVAL_ONLY}" == true ]]; then
    echo "New outputs=${NANO_EPOCH_ROOT}; 30 training-bank evaluations of already selected models"
    echo "Training results collected automatically after all 30 tasks: 48 rows; repetition 1 only"
else
    echo "New outputs=${NANO_EPOCH_ROOT}; 90 candidates, 30 selected models, 156 unique test evaluations"
    echo "Results collected automatically after all 30 tasks: 192 rows; repetition 1 only"
fi
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then
        configure_task "${NANO_TASK_ID}"; print_plan
    else
        for NANO_PREVIEW_ID in {1..30}; do configure_task "${NANO_PREVIEW_ID}"; print_plan; done
    fi
    exit 0
fi
configure_task "${NANO_TASK_ID}"
print_plan
# shellcheck disable=SC1091
source "${NANO_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${NANO_ENV_PATH}"
cd "${NANO_REPO_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
python -u -m tabicl.nano_graph_u.epoch_experiments "${NANO_ARGS[@]}"
if [[ "${NANO_TRAINING_EVAL_ONLY}" == true ]]; then
    echo "EPOCH_TRAINING_EVALUATION_JOB_COMPLETE task=${NANO_TASK_ID} root=${NANO_EPOCH_ROOT}"
else
    echo "EPOCH_EXPERIMENT_JOB_COMPLETE task=${NANO_TASK_ID} root=${NANO_EPOCH_ROOT}"
fi
