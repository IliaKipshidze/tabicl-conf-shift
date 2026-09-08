#!/usr/bin/env bash
#SBATCH --job-name=tabicl-gu-clf-s1
#SBATCH --output=/home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-%j.out
#SBATCH --error=/home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# One-GPU TabICLv2 classifier Stage 1 with the shifted Graph-U prior.
#
# Safe default: RUN_MODE=pilot trains for 100 steps in a pilot-only checkpoint
# directory. For the full recipe, submit with RUN_MODE=full. Full jobs use the
# original 500,000-step schedule and resume automatically from their latest
# checkpoint when the same job is submitted again after a 48-hour allocation.

set -euo pipefail

CLUSTER_ROOT="${CLUSTER_ROOT:-/home/bin/CISPA-scratch/c01ilki}"
REPO_DIR="${REPO_DIR:-${CLUSTER_ROOT}/tabicl-conf-shift}"
CONDA_ROOT="${CONDA_ROOT:-${CLUSTER_ROOT}/miniconda3}"
ENV_PATH="${ENV_PATH:-${CONDA_ROOT}/envs/tabicl-conf-shift}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CLUSTER_ROOT}/tabicl-conf-shift-checkpoints}"

RUN_MODE="${RUN_MODE:-pilot}"
GRAPH_U_QUERY_LOCATION_WAS_SET="${GRAPH_U_QUERY_LOCATION+x}"
GRAPH_U_QUERY_SCALE_WAS_SET="${GRAPH_U_QUERY_SCALE+x}"
GRAPH_U_QUERY_LOCATION="${GRAPH_U_QUERY_LOCATION:-2.0}"
GRAPH_U_QUERY_SCALE="${GRAPH_U_QUERY_SCALE:-1.5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK}}"

case "${RUN_MODE}" in
    pilot)
        MAX_STEPS="${MAX_STEPS:-100}"
        SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-25}"
        SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-100}"
        ;;
    full)
        if [[ -z "${GRAPH_U_QUERY_LOCATION_WAS_SET}" || -z "${GRAPH_U_QUERY_SCALE_WAS_SET}" ]]; then
            echo "Full runs require explicit GRAPH_U_QUERY_LOCATION and GRAPH_U_QUERY_SCALE." >&2
            echo "This prevents accidentally training the scientific run with pilot defaults." >&2
            exit 1
        fi
        MAX_STEPS="${MAX_STEPS:-500000}"
        SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-500}"
        SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-5000}"
        ;;
    *)
        echo "RUN_MODE must be 'pilot' or 'full', got: ${RUN_MODE}" >&2
        exit 1
        ;;
esac

if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer, got: ${MAX_STEPS}" >&2
    exit 1
fi

# Encode the condition in the path. This prevents a run with one shift from
# silently loading a checkpoint produced with different shift parameters.
SHIFT_TAG="loc_${GRAPH_U_QUERY_LOCATION}_scale_${GRAPH_U_QUERY_SCALE}"
SHIFT_TAG="${SHIFT_TAG//-/m}"
SHIFT_TAG="${SHIFT_TAG//./p}"
RUN_TAG="${RUN_MODE}_${MAX_STEPS}steps"
CKPT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/graph_u_clf_stage1/${SHIFT_TAG}/${RUN_TAG}}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${REPO_DIR}"
mkdir -p "${CKPT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

hostname
nvidia-smi
python --version
python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "Run mode: ${RUN_MODE}"
echo "Maximum steps: ${MAX_STEPS}"
echo "Graph-U query location: ${GRAPH_U_QUERY_LOCATION}"
echo "Graph-U query scale: ${GRAPH_U_QUERY_SCALE}"
echo "Graph-U source family: random (not forced Gaussian)"
echo "Checkpoint directory: ${CKPT_DIR}"

# A direct Python launch is intentional: this job requests one GPU, so DDP and
# torchrun are unnecessary. Gradient accumulation preserves global batch 64.
python -u -m tabicl.train \
    --wandb_log False \
    --wandb_project TabICLv2-Graph-U \
    --wandb_name "graph_u_clf_stage1_${SHIFT_TAG}_${RUN_TAG}" \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --lr 8e-4 \
    --muon True \
    --beta1 0.9 \
    --weight_decay 0.01 \
    --use_cautious_wd False \
    --scheduler cosine_with_restarts \
    --warmup_proportion 0.01 \
    --cosine_num_cycles 1 \
    --cosine_amplitude_decay 1 \
    --cosine_lr_end 1e-7 \
    --gradient_clipping 10.0 \
    --prior_type graph_scm \
    --prior_device cpu \
    --n_jobs "${N_JOBS}" \
    --batch_size_per_gp 4 \
    --min_features 1 \
    --max_features 100 \
    --max_classes 10 \
    --max_seq_len 1024 \
    --min_train_size 0.3 \
    --max_train_size 0.9 \
    --seq_len_per_gp True \
    --graph_noise False \
    --filter_unpredictable_graphs True \
    --filter_unpredictable_datasets True \
    --allow_act_warping False \
    --min_n_nodes 3 \
    --max_n_nodes 32 \
    --cauchy_dag_offset 0.0 \
    --graph_u_enabled True \
    --graph_u_query_location "${GRAPH_U_QUERY_LOCATION}" \
    --graph_u_query_scale "${GRAPH_U_QUERY_SCALE}" \
    --graph_u_force_gaussian False \
    --graph_u_max_attempts 1000 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 8 \
    --col_num_inds 128 \
    --col_affine False \
    --col_feature_group same \
    --col_feature_group_size 3 \
    --col_target_aware True \
    --col_ssmax True \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --row_rope_interleaved False \
    --icl_num_blocks 12 \
    --icl_nhead 8 \
    --icl_ssmax True \
    --ssmax_type qassmax-mlp-elementwise \
    --ff_factor 2 \
    --norm_first True \
    --zero_init False \
    --use_flash_attn3 False \
    --checkpoint_dir "${CKPT_DIR}" \
    --save_temp_every "${SAVE_TEMP_EVERY}" \
    --save_perm_every "${SAVE_PERM_EVERY}"
