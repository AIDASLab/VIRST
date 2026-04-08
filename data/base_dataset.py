from abc import ABC, abstractmethod
import time 

from PIL import Image
from typing import Dict

import logging
import torch
import transformers
import traceback

from utils.mm_utils import process_anyres_image_nopad, process_anyres_video_nopad
from utils.video_utils import VIDEO_READER_FUNCS
from utils.preprocess import preprocess_multimodal, preprocess
from utils.argument import DataArguments
from utils.constants import IGNORE_INDEX

def collate_fn(
    batch, tokenizer,
):
    image_path_list = []
    images_sam_list = []
    images_clip_list = []
    masks_list = []             # (num_conv, O, T', h, w )
    modalities_list = []
    input_ids_list = []
    label_list = []
    resize_list = []
    image_ids_list = []
    video_path_list = []
    frame_ids_list = []
    questions_list = []
    dataset_name_list = []
    exp_id_list = []
    
    for item in batch:
        image_path_list.append(item["image_path"])
        images_sam = item.get("images_sam", None)
        if images_sam is not None:
            images_sam = images_sam.to(torch.bfloat16)
        images_sam_list.append(images_sam)
        images_clip = item.get("images_clip", None)
        if images_clip is not None:
            images_clip = images_clip.to(torch.bfloat16)
        masks = item.get("masks", None)
        if masks is not None:
            masks = [m.to(torch.bfloat16).transpose(0,1) for m in masks] # (O, T_seg, H, W) -> (T_seg, O, H, W)
            masks_list.extend(masks)
        else:
            masks_list.append(None)
        
        images_clip_list.append(images_clip)    
        input_ids_list.extend(item.get("input_ids", None)) # NOTE: extend these since there can be more than one conv
        image_ids_list.extend([len(images_clip_list)-1] * len(item.get("input_ids", None))) #  the number of conversations per sample = num_classes
        label_list.extend(item.get("labels", None))
        modalities_list.extend(item.get("modalities", None))

        resize_list.append(item.get("resize", None))
        questions = item.get("questions", None)
        if questions is not None:
            questions_list.extend(questions)
            
        video_paths = item.get("video_paths", None)
        if video_paths is not None:
            video_path_list.extend(video_paths) # video paths is supposed to be given only one
        frame_ids_list.append(item.get("frame_ids", None))
        dataset_name_list.append(item.get("dataset_name", None))
        exp_ids = item.get("exp_ids", None)
        if exp_ids is not None:
            exp_id_list.extend(exp_ids) # only for inference 
        
    # pad sequences 
    input_ids  = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, 
        batch_first = True, 
        padding_value = tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        label_list,
        batch_first= True,
        padding_value= IGNORE_INDEX
    )
    
    images_clip_tensor = torch.stack(images_clip_list, dim=0).to(torch.bfloat16)
    if images_sam_list[0] is not None:
        images_sam_tensor = torch.stack(images_sam_list, dim=0).to(torch.bfloat16)
    else:
        images_sam_tensor = None
    
    return {
        "image_paths": image_path_list,
        "images_sam": images_sam_tensor, #BS : T * 3 * H * W
        "images_clip": images_clip_tensor, #BS : T * 3 * H * W
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention_masks,
        "masks_list": masks_list, # [Conv * B, T_seg, O, H, W, ...]
        "resize_list": resize_list,
        "image_ids": image_ids_list,
        "modalities": modalities_list, # BS: 1 
        "questions": questions_list, # only for inference
        "video_path": video_path_list, # only for inference
        "frame_ids": frame_ids_list, # only for inference
        "exp_id": exp_id_list, # only for inference 
        "dataset_name": dataset_name_list,
    }

class BaseVirstDataset(torch.utils.data.Dataset, ABC):
    #_get_item() method should be implemented
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args 
        self.num_video_tokens = max(8, data_args.frames_upbound) * 128 // 8 # Estimate the number of video tokens 
        self.client = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if "image" in sample:
                img_tokens = 128
            elif "video" in sample:
                img_tokens = self.num_video_tokens
            else:
                img_tokens = 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list
    
    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list
    
    def process_image_vlm(self, image_file, overwrite_image_aspect_ratio=None):
        processor = self.data_args.image_processor
        try: 
            image = Image.open(image_file).convert('RGB')
        except Exception as exn:
            logging.error("Failed to open image %s: %s", image_file, exn)
            raise exn
        
        image_size = image.size
        image = process_anyres_image_nopad(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)

        return image, image_size, "image"

    def process_video_vlm(self, video_file, data_anno, data_args):
        """
        video_file: video path (.mp4)
        data_anno: dict["start":, "end":, "video_read_type", "decord",]
        """
        local_num_frames = data_args.local_num_frames
        max_num_frames = data_args.frames_upbound
        min_num_frames = data_args.frames_lowbound
        sample_type = data_args.sample_type
        video_reader_type = data_anno.get("video_read_type", "decord")
        if "start" in data_anno and "end" in data_anno:
            clip = [float(data_anno["start"]), float(data_anno["end"])]
        else:
            clip = None
        
        if video_reader_type == 'rvos':
            video_reader = VIDEO_READER_FUNCS[video_reader_type]

            frames, frame_indices, fps, duration = video_reader(
                video_file, max_num_frames, sample_type,
                min_num_frames=min_num_frames, 
                max_num_frames=max_num_frames, client=self.client, clip=clip,
                local_num_frames=local_num_frames
            )
                
        elif clip is None or video_reader_type == "img":
            video_reader = VIDEO_READER_FUNCS[video_reader_type]
            frames, frame_indices, fps, duration = video_reader(
                video_file, max_num_frames, sample_type,
                min_num_frames=min_num_frames, 
                max_num_frames=max_num_frames, client=self.client, clip=clip,
                local_num_frames=local_num_frames
            )
            
        else:
            video_reader = VIDEO_READER_FUNCS['lazy']
            start, end = clip
            duration = end - start
            if min_num_frames > duration:
                min_num_frames = (duration // local_num_frames) * local_num_frames
                
            if sample_type == 'dynamic_fps1':
                num_segments = int(duration // local_num_frames)
                if num_segments == 0:
                    num_frames = local_num_frames
                else:
                    num_frames = local_num_frames * num_segments

                num_frames = min(num_frames, max_num_frames)
                num_frames = max(num_frames, min_num_frames)
            else:
                num_frames = max_num_frames

            frames, frame_indices, fps, duration = video_reader(video_file, num_frames=num_frames, video_start=start, video_end=end, client=self.client)
        
        if sample_type == 'dynamic_fps1' and len(frames) % local_num_frames != 0:
            raise ValueError(f"min_num_frames={min_num_frames}, max_num_frames={max_num_frames},  local_num_frames={local_num_frames}, len(frames)={len(frames)}, is wrong!!!")

        sec = [str(round(f / fps, 1)) for f in frame_indices]

        if data_args.time_msg is not None and sec is not None:
            if data_args.time_msg == 'short':
                msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled from it. "
            else:
                msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled at {', '.join(sec)} seconds. "
        else:
            msg = ""

        return frames, frame_indices, msg

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 2
        
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                logging.error(f"[ERROR] [Try #{attempt_idx}] Failed to fetch sample {i} with {e} : {traceback.format_exc()}") 
                if attempt_idx != (num_base_retries -1):
                    time.sleep(1)

        retry_step = 5
        for attempt_idx in range(num_base_retries+3):
            try:
                next_index = min(i + retry_step, len(self) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                logging.error(f"[ERROR] [Try #{attempt_idx}] Failed to fetch sample {next_index} with {e} : {traceback.format_exc()}") 
                retry_step *= 2

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    @abstractmethod
    def _get_item(self, idx):
        return dict()

    def _get_dict(self, i, sources) -> Dict[str, torch.Tensor]:
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]

            if type(image_file) is list:
                if len(image_file) > 1:
                    image = [self.process_image_vlm(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
                else:
                    image = [self.process_image_vlm(f) for f in image_file]
            else:
                image = [self.process_image_vlm(image_file)]
            sources = preprocess_multimodal([e["conversations"] for e in sources], self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]

            try:
                video, time_msg = self.process_video_vlm(video_file, data_anno=self.list_data_dict[i], data_args=self.data_args)

                processor = self.data_args.image_processor
                frame_aspect_ratio = self.data_args.frame_aspect_ratio
                if "anyres" in frame_aspect_ratio:
                    if 'nopad' in frame_aspect_ratio:
                        image = process_anyres_video_nopad(video, self.data_args.image_processor, self.data_args.frame_grid_pinpoints, max_resolutions=self.data_args.max_num_pixels // len(video))
                    else:
                        raise NotImplementedError
                else:
                    image = processor.preprocess(video, return_tensors="pt")["pixel_values"]

                image = [(image, video[0].shape[0:2], "video")]
                sources = preprocess_multimodal([e["conversations"] for e in sources], self.data_args, msg=time_msg)

            except Exception as e:
                logging.error("Failed to read video file %s: %s", video_file, e)
                raise e
        else:
            sources = [e["conversations"] for e in sources]

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        
        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict
