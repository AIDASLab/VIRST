GPU_DEVICES="0"

PROMPT_VERSION="qwen_2"
TOKENIZER="model/videochat"
RVOS_ROOT="your/path/to/dataset/RVOS_ROOT"
EVAL_OUTPUT_ROOT="./eval_results"
EVAL_LOG_ROOT="./eval_results"

export DS_ZERO_STAGE=3
export DS_OFFLOAD_OPTIMIZER_DEVICE=none
export DS_OFFLOAD_PARAM_DEVICE=none

export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=$((10000 + RANDOM % 50000))

echo "Using GPUs: $GPU_DEVICES"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} deepspeed \
    --master_port $MASTER_PORT \
    eval.py \
    --tokenizer ${TOKENIZER} \
    --rvos_root ${RVOS_ROOT} \
    --eval_output_root ${EVAL_OUTPUT_ROOT} \
    --eval_log_root ${EVAL_LOG_ROOT} \
    --version ${PROMPT_VERSION} \
    --output_dir ./output/ \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing \
    --bf16 True \
    --local_num_frames 4 \
    --mm_local_num_frames 4 \
    --vision_encode_type video_image \
    --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --batch_size_per_device 1 \
    --bce_loss_weight 1 \
    --dice_loss_weight 1 \
    --steps_per_epoch 1000 \
    --num_classes_per_sample 3 \
    --seg_image_length 32 \
    --num_seg_keyframes 3 \
    --seg_image_size 1024 \
    --logging_steps 5 \
    --wandb False \
    --wandb_train_name "your_wandb_train_name" \
    --keyframe_scheme "uniform"
