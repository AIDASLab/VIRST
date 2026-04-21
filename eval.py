import os
import logging
import glob
from functools import partial, reduce
import deepspeed
from tqdm import tqdm 
import wandb
import traceback
import json
from datetime import datetime
from itertools import islice
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp 
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
import random

from model.builder import build_virst, load_checkpoint_virst

from data.rvos_dataset import RVOSDataset
from data.base_dataset import collate_fn
from data.dataset_config import RVOS_DATA_INFO as _DATA_INFO 
from data.dataset_config import RVOS_ROOT 
from utils import conversation as conversation_lib
from utils.argument import ModelArguments, DataArguments, TrainingArguments
from utils.mm_utils import KeywordsStoppingCriteria

logger = logging.getLogger(__name__)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_dataset_builder(
        tokenizer, 
        data_args,
        training_args,
        rank=0,
        world_size=1,
        dataset = 'refytvos_valid',
    ):
    val_dataset = RVOSDataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        samples_per_epoch=training_args.steps_per_epoch,
        num_frames_sample_range="8,8",
        rvos_seg_data = dataset, 
        rvos_sample_policy = "flex",
        train = False,
        rvos_root=data_args.rvos_root,
    )
    
    total_len = len(val_dataset)
    start = total_len * rank // world_size 
    end = total_len * (rank + 1) // world_size
    subset = torch.utils.data.Subset(val_dataset, list(range(start,end)))
    
    val_loader = DataLoader(
        subset,
        batch_size=1,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
        ),
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    return val_dataset, val_loader

def move_to_device(batch, device):
    """Recursively move tensors in batch to the given device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        return batch

def sync_timestamp(shared_path="logs/shared_timestamp.txt"):
    if is_main_process():
        timestamp_str = datetime.now().strftime("%y%m%d%H%M")
        with open(shared_path, "w") as f:
            f.write(timestamp_str)
    else:
        while not os.path.exists(shared_path):
            time.sleep(0.1)
        with open(shared_path, "r") as f:
            timestamp_str = f.read().strip()
    return timestamp_str

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def main(timestamp=None):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    set_seed(42)
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if timestamp is None:
        timestamp = training_args.wandb_train_name

    dataset = training_args.dataset
    print(f"Evaluating {dataset}")
    
    run_name = f"eval_{dataset}_{timestamp}"
    SAVE_ROOT = os.path.join(training_args.eval_output_root, dataset)
    LOG_ROOT = os.path.join(training_args.eval_log_root or training_args.eval_output_root, dataset)
    log_dir = f"{LOG_ROOT}/logs/{run_name}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/{local_rank}.txt",
        filemode="w",
        level=logging.INFO,
        format="%(message)s"
    )
    
    model_path = "model"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")

    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    logging.info('config and tokenizer have been loaded.')
    
    ckpt = training_args.model_checkpoint
    tokenizer.add_tokens(["[SEG]"])
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = build_virst(config, model_args= model_args, checkpoint=ckpt, seg_token_idx=seg_token_idx)
    model.resize_token_embeddings(len(tokenizer))

    logging.info(f"[SEG] token is added {seg_token_idx} {tokenizer.convert_ids_to_tokens(seg_token_idx)}")

    model.config.max_num_pixels = data_args.max_num_pixels
    model.config.frame_grid_pinpoints = data_args.frame_grid_pinpoints
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.frame_aspect_ratio = data_args.frame_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True 

    eval_dataset, dataloader = eval_dataset_builder(
        tokenizer=tokenizer, 
        data_args=data_args,
        training_args=training_args,
        rank=local_rank,
        world_size=world_size,
        dataset = dataset,
    )
    
    _, exp_root = _DATA_INFO[dataset]
    rvos_root = data_args.rvos_root or RVOS_ROOT
    exp_path = os.path.join(rvos_root, exp_root)
    
    exp_dict = json.load(open(exp_path))['videos']

    target_modules = []
    target_modules_suffix = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in target_modules_suffix) and ("layers" in name) and ("seg_model" not in name):
            target_modules.append(name)
    
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if ckpt is not None:   
        model = load_checkpoint_virst(model,ckpt)

    model.eval()
    
    model = deepspeed.init_inference(
        model=model,
        mp_size = 1,  # torch.cuda.device_count(),
        dtype=torch.bfloat16,
        replace_method = 'auto',
        replace_with_kernel_inject=True,
    )
    
    modules_to_float = [
        'module.base_model.model.model.seg_model.sam_prompt_encoder'
    ]
    for path in modules_to_float:
        reduce(getattr, path.split('.'), model).to(torch.float32)

    for step, batch in enumerate(tqdm(dataloader)):
        try:
            device = next(model.parameters()).device
            batch = move_to_device(batch, device)

            video_name = batch['video_path'][0].split('JPEGImages' + os.sep, 1)[1]
            filename_save = f"{SAVE_ROOT}/{run_name}/{video_name}/{batch['exp_id'][0]}"

            if os.path.exists(filename_save):
                tqdm.write(f"video {video_name} already inferenced")
                continue

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = batch["input_ids"]

                video_segments = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_masks"],
                    images_clip    = batch["images_clip"],   #BS : T * 3 * H * W
                    images_sam     = batch["images_sam"], # [T,B,C,H,W]
                    image_ids      = batch["image_ids"],
                    labels         = batch["labels"],
                    modalities     = batch["modalities"],
                    gt_masks       = batch["masks_list"], #[conv,T,H,W]
                    video_path     = batch["video_path"],
                    frame_ids      = batch["frame_ids"],
                    generation     = False,
                    seg_evaluate   = True,
                    filename_save=f"{video_name}/{batch['exp_id'][0]}",
                )
            
            query_sentence = batch["input_ids"][0]
            query_sentence = query_sentence[query_sentence != -200] 
            input_str = tokenizer.decode(query_sentence, skip_special_tokens=False)

            os.makedirs(filename_save, exist_ok=True)
            logging.info(f"{video_name} {input_str}")
            
            all_frame_ids = exp_dict[video_name]['frames']

            save_binary_mask(
                video_segments=video_segments,
                output_dir = filename_save,
                frame_ids = all_frame_ids
            )
        
        except Exception as e:
            video_name = os.path.basename(batch['video_path'][0])
            logging.error(f"[ERROR] Exception at {video_name}: {traceback.format_exc()}") 
            raise e
        
import numpy as np 
import matplotlib.pyplot as plt 
import cv2    
import imageio
    
def save_binary_mask(video_segments, output_dir, frame_ids):
    os.makedirs(output_dir, exist_ok=True)
    sorted_frames = sorted(video_segments.keys())
    
    for i, frame_id in enumerate(sorted_frames):
        frame_dict = video_segments[frame_id]
        
        h, w = next(iter(frame_dict.values())).shape[-2:]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for mask in frame_dict.values():
            combined_mask |= mask.squeeze().astype(np.uint8)
        
        filename = f"{frame_ids[i]}.png"
        save_path = os.path.join(output_dir, filename)

        imageio.imwrite(save_path, combined_mask * 255) 
        
    print(f"Saved {len(sorted_frames)} masks to {output_dir}")

def images_to_video(image_paths, output_path, fps=10):
    """
    Create a video from a list of image paths.

    Args:
        image_paths (List[str]): List of image file paths (must be same size).
        output_path (str): Path to output video file (e.g., 'output.mp4').
        fps (int): Frames per second.

    Raises:
        ValueError: If image list is empty or image sizes are inconsistent.
    """
    print("image paths: ", image_paths)
    
    if not image_paths:
        raise ValueError("Image path list is empty.")

    # Read first image to get size
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise ValueError(f"Cannot read image: {image_paths[0]}")
    height, width, _ = first_img.shape

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: skipping unreadable image: {img_path}")
            continue

        # Optional: resize to match first image
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        writer.write(img)

    writer.release()
    print(f"Video saved to: {output_path}")

    
def show_mask_and_save(mask, save_path, obj_id=None, random_color=False):
    """
    mask: np.ndarray of shape (H, W), dtype=bool or binary
    save_path: str, path to save image (e.g., 'mask.png')
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])  # RGBA

    _, h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    fig, ax = plt.subplots()
    ax.imshow(mask_image)
    ax.axis("off") 
    
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def show_mask_as_rgba(mask, obj_id=None, random_color=False):
    """Convert a binary mask to an RGBA image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id % 10  # tab10 has only 10 colors
        color = np.array([*cmap(cmap_idx)[:3], 0.6])  # RGBA

    h, w = mask.shape[-2:]
    rgba_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    rgb_mask = (rgba_mask[:, :, :3] * 255).astype(np.uint8)
    return rgb_mask

def save_video_from_masks(video_segments, output_path, frame_size, fps=5):
    """
    Saves a video by overlaying masks per frame.
    :param video_segments: Dict[int frame_idx][int obj_idx] = binary mask
    :param output_path: Path to save video
    :param frame_size: (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    sorted_frames = sorted(video_segments.keys())
    for frame in sorted_frames:
        combined_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        for obj_idx, obj_mask in video_segments[frame].items():
            obj_mask = obj_mask.squeeze()
            colored_mask = show_mask_as_rgba(obj_mask, obj_id=obj_idx)
            mask_bool = obj_mask.astype(bool)
            combined_frame[mask_bool] = colored_mask[mask_bool]

        writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved video: {output_path}")


if __name__=='__main__':

    main()
