#!/usr/bin/env bash

set -euo pipefail

GPU_DEVICES="${GPU_DEVICES:-0}"
DATASET="${1:-${DATASET:-}}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_2}"
TOKENIZER="${TOKENIZER:-model/videochat}"
VIDEOCHAT_CHECKPOINT="${VIDEOCHAT_CHECKPOINT:-checkpoints/videochat}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-checkpoints/sam2.1_hiera_large.pt}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-checkpoints/virst_checkpoint.pt}"
RVOS_ROOT="${RVOS_ROOT:-}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-./eval_results}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-./eval_results}"
WANDB_TRAIN_NAME="${WANDB_TRAIN_NAME:-eval_mevis}"

if [[ -z "${DATASET}" ]]; then
    echo "Usage: bash scripts/eval_mevis.sh <dataset>"
    echo "Supported datasets: mevis_valid, mevis_test"
    exit 1
fi

case "${DATASET}" in
    mevis_valid)
        DATASET_SUBDIR="mevis/valid_u"
        ;;
    mevis_test)
        DATASET_SUBDIR="mevis/valid"
        ;;
    *)
        echo "Unsupported dataset: ${DATASET}"
        echo "Supported datasets: mevis_valid, mevis_test"
        exit 1
        ;;
esac

export DS_ZERO_STAGE=3
export DS_OFFLOAD_OPTIMIZER_DEVICE=none
export DS_OFFLOAD_PARAM_DEVICE=none
export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "Using GPUs: ${GPU_DEVICES}"
echo "Dataset: ${DATASET}"
echo "Eval output root: ${EVAL_OUTPUT_ROOT}"

for required_file in "${MODEL_CHECKPOINT}" "${SAM2_CHECKPOINT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required checkpoint not found: ${required_file}" >&2
        echo "Run: bash scripts/setup_assets.sh" >&2
        exit 1
    fi
done

if [[ ! -d "${VIDEOCHAT_CHECKPOINT}" ]]; then
    echo "VideoChat checkpoint directory not found: ${VIDEOCHAT_CHECKPOINT}" >&2
    echo "Run: bash scripts/setup_assets.sh" >&2
    exit 1
fi

RESOLVED_RVOS_ROOT="${RVOS_ROOT:-dataset/RVOS_ROOT}"
if [[ ! -f "${RESOLVED_RVOS_ROOT}/${DATASET_SUBDIR}/meta_expressions.json" ]]; then
    echo "Dataset metadata not found: ${RESOLVED_RVOS_ROOT}/${DATASET_SUBDIR}/meta_expressions.json" >&2
    echo "Set RVOS_ROOT to the directory containing the mevis folder." >&2
    exit 1
fi

CMD=(
    deepspeed
    --master_port "${MASTER_PORT}"
    eval.py
    --tokenizer "${TOKENIZER}"
    --videochat_checkpoint "${VIDEOCHAT_CHECKPOINT}"
    --sam2_checkpoint "${SAM2_CHECKPOINT}"
    --dataset "${DATASET}"
    --eval_output_root "${EVAL_OUTPUT_ROOT}"
    --eval_log_root "${EVAL_LOG_ROOT}"
    --version "${PROMPT_VERSION}"
    --output_dir ./output/
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --group_by_modality_length True
    --mm_patch_merge_type spatial_nopad
    --mm_newline_position nothing
    --bf16 True
    --local_num_frames 4
    --mm_local_num_frames 4
    --vision_encode_type video_image
    --image_aspect_ratio anyres_nopad
    --image_grid_pinpoints "(1x1),...,(6x6)"
    --num_train_epochs 100
    --learning_rate 1e-5
    --weight_decay 0.
    --batch_size_per_device 1
    --bce_loss_weight 1
    --dice_loss_weight 1
    --steps_per_epoch 1000
    --num_classes_per_sample 3
    --seg_image_length 32
    --num_seg_keyframes 3
    --seg_image_size 1024
    --logging_steps 5
    --wandb False
    --wandb_train_name "${WANDB_TRAIN_NAME}"
    --keyframe_scheme uniform
)

CMD+=(--model_checkpoint "${MODEL_CHECKPOINT}")

if [[ -n "${RVOS_ROOT}" ]]; then
    CMD+=(--rvos_root "${RVOS_ROOT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${CMD[@]}"
