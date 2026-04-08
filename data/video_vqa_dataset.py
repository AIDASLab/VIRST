import glob
import json
import logging
import os.path as osp
import random
from pathlib import Path

import numpy as np

from data.base_dataset import BaseVirstDataset
from data.dataset_config import VQA_VIDEO_ROOT
from utils.preprocess import preprocess_virst
from utils.utils import TASK_VIDEO_TEXT_ONLY

logger = logging.getLogger(__name__)

class VideoVQADataset(BaseVirstDataset):

    def __init__(
        self, 
        tokenizer, 
        data_args, 
        num_frames_sample_range : str   = "8,12",
        sample_policy      : str   = "uniform", # all, uniform, random
        
        samples_per_epoch : int = 500 * 8 * 2 * 10,
    ):
        super().__init__(tokenizer, data_args)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.samples_per_epoch = samples_per_epoch
        self.video_vqa_ROOT = VQA_VIDEO_ROOT
        self.image_ROOT = osp.join(self.video_vqa_ROOT, "JPEGImages")
        
        self.num_frames_sample_range = [int(x) for x in num_frames_sample_range.split(",")]
        assert len(self.num_frames_sample_range) == 2 and self.num_frames_sample_range[0] <= self.num_frames_sample_range[1], f"invalid num_frames_sample_range {num_frames_sample_range}"
        self.sample_policy = sample_policy
        assert self.sample_policy in ["all", "uniform", "random", "flex"], f"invalid sample_policy {self.sample_policy}"
        
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        json_file = osp.join(self.video_vqa_ROOT, "0_30_s_academic_v0_1_cap_processed.json")
        with open(json_file, "r") as f:
            dataset = json.load(f)
        
        logger.info("Loaded %d video VQA samples", len(dataset))
        return dataset
        
            
    def __len__(self):
        return self.samples_per_epoch
    
    def _get_item(self, idx):
        # NOTE: LLaVA academic dataset has only one conversation per video
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[idx]
        
        vid_id = item["id"]
        conv = item["conversations"]
        video = item["video"]
        
        # text tokenization
        input_ids = []
        labels = []
        modalities = []
        questions = []
        
        question = conv[0]["value"] # human: 
        question = question.replace("<image>", "") # remove existing <image> token
        answer = conv[1]["value"] # assistant: 
        
        video_path = Path(osp.join(self.image_ROOT, f"{video}"))
        video_path = video_path.with_suffix("")
        video_frame_path_list = glob.glob(osp.join(video_path, "*.jpg"))
        
        data_anno = {
            "video_read_type": "rvos",
        }
        
        
        num_frames_per_sample = np.random.randint(self.num_frames_sample_range[0], self.num_frames_sample_range[1] + 1)
        
        
        frame_ids = []
        if len(video_frame_path_list) > num_frames_per_sample:
            if self.sample_policy == "random":
                frame_ids = np.random.choice(len(video_frame_path_list), num_frames_per_sample, replace=False).tolist()
                frame_ids = sorted(frame_ids)
            elif self.sample_policy == "uniform":
                num_length = len(video_frame_path_list)
                split_point = np.linspace(0, num_length, num=num_frames_per_sample+1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i+1]) for i in range(num_frames_per_sample)]
            elif self.sample_policy == "all":
                frame_ids = list(range(len(video_frame_path_list)))
            elif self.sample_policy == "flex":
                num_length = len(video_frame_path_list)

                # sample the number of frames to be the closest multiple of 4
                target_frames = min(32, num_length)
                target_frames = (target_frames // 4) * 4

                # sample the frames uniformly
                split_point = np.linspace(0, num_length, num=target_frames + 1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(target_frames)]
        else:
            frame_ids = list(range(len(video_frame_path_list)))
        video_frame_path_list = [video_frame_path_list[i] for i in frame_ids]
        
        processor = self.data_args.image_processor
        
        frames, frame_indices, time_msg = self.process_video_vlm(
            video_file = video_frame_path_list,
            data_anno = data_anno,
            data_args = self.data_args, 
        )
        frames_clip = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            
        out = preprocess_virst(
            question, 
            self.tokenizer, 
            has_image=True,
            seg_token_num=0,
            add_explanation=answer,
            modality = "video",
            only_text=True,
            task_prompt=TASK_VIDEO_TEXT_ONLY
        )
        
        input_ids.append(out["input_ids"])
        labels.append(out["labels"])
        modalities.append("video")
        questions.append(f"video_vqa_{question}")
        
        resize=None
        
        return {
            "image_path": ','.join(video_frame_path_list),
            "images_sam": None,
            "images_clip": frames_clip,
            "masks": None,
            "input_ids": input_ids,
            "labels": labels,
            "resize": resize,
            "modalities": modalities,
            "questions": questions,
        }
