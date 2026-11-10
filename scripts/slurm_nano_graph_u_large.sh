#!/usr/bin/env bash
#SBATCH --job-name=nano-gu-large
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-large-%A_%a.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-large-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1
#SBATCH --export=ALL

# First: sbatch scripts/slurm_nano_graph_u_large.sh smoke
# After generation: sbatch --array=1-36%2 scripts/slurm_nano_graph_u_large.sh trial
# After all trials: sbatch --array=1-12%2 scripts/slurm_nano_graph_u_large.sh evaluate
set -euo pipefail
NANO_CLUSTER_ROOT="${NANO_CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
NANO_LARGE_ROOT="${NANO_LARGE_ROOT:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift-nano-large-r1000-f8}"
NANO_REPO_DIR="${NANO_REPO_DIR:-${NANO_CLUSTER_ROOT}/tabicl-conf-shift}"
NANO_CONDA_ROOT="${NANO_CONDA_ROOT:-${NANO_CLUSTER_ROOT}/miniconda3}"
NANO_ENV_PATH="${NANO_ENV_PATH:-${NANO_CONDA_ROOT}/envs/tabicl-conf-shift}"
NANO_STAGE=smoke; NANO_STAGE_SET=false; NANO_DRY_RUN=false
for NANO_OPTION in "$@"; do
    case "${NANO_OPTION}" in
        --dry-run) NANO_DRY_RUN=true ;;
        smoke|trial|evaluate)
            if [[ "${NANO_STAGE_SET}" == true ]]; then echo "Specify only one GPU stage" >&2; exit 1; fi
            NANO_STAGE="${NANO_OPTION}"; NANO_STAGE_SET=true ;;
        *) echo "Usage: $0 [smoke|trial|evaluate] [--dry-run]" >&2; exit 1 ;;
    esac
done
for NANO_NAME in NANO_CLUSTER_ROOT NANO_LARGE_ROOT NANO_REPO_DIR NANO_CONDA_ROOT NANO_ENV_PATH; do
    if [[ "${!NANO_NAME}" != /* ]]; then echo "${NANO_NAME} must be an absolute path" >&2; exit 1; fi
done
if ! [[ "${SLURM_CPUS_PER_TASK:-2}" =~ ^[1-9][0-9]*$ ]]; then echo "SLURM_CPUS_PER_TASK must be positive" >&2; exit 1; fi
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
NANO_OUTPUT_CHECK="$(normalize_path "${NANO_LARGE_ROOT}")"
for NANO_SOURCE in "${NANO_REPO_DIR}" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-priors" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-checkpoints" "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-evaluations" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-fixed-experiments" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-experiments-shuffled" \
    "${NANO_CLUSTER_ROOT}/tabicl-conf-shift-epoch-test-shift-1p5"; do
    NANO_SOURCE_CHECK="$(normalize_path "${NANO_SOURCE}")"
    if [[ "${NANO_OUTPUT_CHECK}" == / || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}" \
        || "${NANO_OUTPUT_CHECK}" == "${NANO_SOURCE_CHECK}/"* || "${NANO_SOURCE_CHECK}" == "${NANO_OUTPUT_CHECK}/"* ]]; then
        echo "NANO_LARGE_ROOT must not overlap existing source roots: ${NANO_SOURCE}" >&2; exit 1
    fi
done
case "${NANO_STAGE}" in smoke) NANO_TASK_COUNT=1 ;; trial) NANO_TASK_COUNT=36 ;; evaluate) NANO_TASK_COUNT=12 ;; esac
configure_task() {
    NANO_TASK_ID="$1"
    if ! [[ "${NANO_TASK_ID}" =~ ^[1-9][0-9]*$ ]] || (( NANO_TASK_ID > NANO_TASK_COUNT )); then
        echo "SLURM_ARRAY_TASK_ID must be from 1 to ${NANO_TASK_COUNT} for ${NANO_STAGE}" >&2; exit 1
    fi
    NANO_ARGS=("${NANO_STAGE}" --root "${NANO_LARGE_ROOT}" --device cuda)
    if [[ "${NANO_STAGE}" == smoke ]]; then return; fi
    NANO_ARGS+=(--task-id "${NANO_TASK_ID}")
    local NANO_SLOT NANO_SETTING NANO_BUDGET
    if [[ "${NANO_STAGE}" == trial ]]; then
        NANO_BUDGET=$(( (NANO_TASK_ID - 1) / 12 )); NANO_SLOT=$(( (NANO_TASK_ID - 1) % 12 )); NANO_SETTING=$(( NANO_SLOT / 3 ))
        case "$(( NANO_SLOT % 3 ))" in 0) NANO_LR=0.0007 ;; 1) NANO_LR=0.001 ;; 2) NANO_LR=0.002 ;; esac
    else
        NANO_BUDGET=$(( (NANO_TASK_ID - 1) / 4 )); NANO_SETTING=$(( (NANO_TASK_ID - 1) % 4 ))
    fi
    case "${NANO_BUDGET}" in 0) NANO_EPOCHS=1 ;; 1) NANO_EPOCHS=3 ;; 2) NANO_EPOCHS=9 ;; esac
    NANO_MODE=confounding; NANO_CONDITION=identity
    if (( NANO_SETTING >= 2 )); then NANO_MODE=covariate; fi
    if (( NANO_SETTING % 2 == 1 )); then NANO_CONDITION=shift; fi
}
print_plan() {
    if [[ "${NANO_STAGE}" == smoke ]]; then
        echo "stage=smoke task=1 batch=32 rows=1000 features=8 support_rows=900; unchanged Nano forward/backward; output=${NANO_LARGE_ROOT}/smoke.json"
    else
        echo "stage=${NANO_STAGE} task=${NANO_TASK_ID} epochs=${NANO_EPOCHS} updates=$(( 15625 * NANO_EPOCHS )) mode=${NANO_MODE} condition=${NANO_CONDITION} batch=32 rows=1000 features=8 model_seed=42 seeded_shuffle=true"
        if [[ "${NANO_STAGE}" == trial ]]; then
            echo "learning_rate=${NANO_LR}; full-budget matching validation AUROC on 32,000 tasks; checkpoints resume automatically"
        else
            echo "Select one of three completed LR candidates; score all four 32,000-task tests and own 500,000-task training bank"
        fi
    fi
    printf 'Command:'; printf ' %q' python -u -m tabicl.nano_graph_u.large_experiment "${NANO_ARGS[@]}"; printf '\n'
}
echo "New experiment=${NANO_LARGE_ROOT}; fixed shifted location=2 scale=1.5 on train/validation/test; IID location=0 scale=1"
echo "Run smoke before full generation (~90.24 GB); 36 candidates -> 12 winners -> 48 test rows and 12 training rows; repetition=1"
if [[ "${NANO_DRY_RUN}" == true ]]; then
    if [[ "${SLURM_ARRAY_TASK_ID+x}" == x ]]; then configure_task "${SLURM_ARRAY_TASK_ID}"; print_plan
    else for (( NANO_PREVIEW_ID=1; NANO_PREVIEW_ID<=NANO_TASK_COUNT; NANO_PREVIEW_ID++ )); do configure_task "${NANO_PREVIEW_ID}"; print_plan; done; fi
    exit 0
fi
if [[ "${NANO_STAGE}" == smoke ]]; then configure_task "${SLURM_ARRAY_TASK_ID:-1}"
else configure_task "${SLURM_ARRAY_TASK_ID:-}"; fi
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
python -u -m tabicl.nano_graph_u.large_experiment "${NANO_ARGS[@]}"
echo "LARGE_GPU_JOB_COMPLETE stage=${NANO_STAGE} task=${NANO_TASK_ID} root=${NANO_LARGE_ROOT}"
