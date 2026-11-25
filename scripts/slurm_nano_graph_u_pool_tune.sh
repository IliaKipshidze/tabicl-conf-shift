#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-pool-tune
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-tune-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-pool-tune-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-300%2

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

NANO_CONDITIONS=(uniform shift mixed)
NANO_RATES=(0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 0.002 0.003 0.004)
configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]{0,2}$ ]] || (( NANO_TASK_ID > 300 )); then
        echo "SLURM_ARRAY_TASK_ID must be an integer from 1 to 300" >&2; exit 1
    fi
    NANO_LOCAL=$(( (NANO_TASK_ID - 1) % 150 ))
    if (( NANO_TASK_ID <= 150 )); then NANO_MODE=confounding; else NANO_MODE=covariate; fi
    NANO_REPETITION=$(( NANO_LOCAL / 30 + 1 ))
    NANO_CONDITION="${NANO_CONDITIONS[(NANO_LOCAL % 30) / 10]}"
    NANO_LR="${NANO_RATES[NANO_LOCAL % 10]}"
    NANO_LR_TAG="${NANO_LR//./p}"
    NANO_MODEL_SEED=$(( 41 + NANO_REPETITION ))
    NANO_ARGS=(trial --root "${NANO_POOL_ROOT}" --task-id "${NANO_TASK_ID}" --device cuda)
}
print_plan() {
    echo "trial=${NANO_TASK_ID} graph_u_mode=${NANO_MODE} repetition=${NANO_REPETITION} condition=${NANO_CONDITION} learning_rate=${NANO_LR} model_seed=${NANO_MODEL_SEED} epochs=1 updates=2500"
    echo "Train=${NANO_POOL_ROOT}/data/${NANO_MODE}/rep_${NANO_REPETITION}/${NANO_CONDITION}_train_seed$(( 1000 + NANO_REPETITION ))_2500steps_b32_r150_f5.h5"
    echo "Validation=${NANO_POOL_ROOT}/data/${NANO_MODE}/validation/${NANO_CONDITION}_validation_seed3001_100steps_b32_r150_f5.h5"
    echo "Candidate=${NANO_POOL_ROOT}/checkpoints/${NANO_MODE}/rep_${NANO_REPETITION}/${NANO_CONDITION}/lr_${NANO_LR_TAG}"
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"; printf '\n'
}
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then configure_task "${SLURM_ARRAY_TASK_ID}"; print_plan
    else for (( NANO_PREVIEW_ID=1; NANO_PREVIEW_ID<=300; NANO_PREVIEW_ID++ )); do configure_task "${NANO_PREVIEW_ID}"; print_plan; done; fi
    exit 0
fi
configure_task "${SLURM_ARRAY_TASK_ID:-}"
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
python -u -m tabicl.nano_graph_u.pool_experiment "${NANO_ARGS[@]}"
echo "POOL_TRIAL_COMPLETE task=${NANO_TASK_ID} root=${NANO_POOL_ROOT}"
