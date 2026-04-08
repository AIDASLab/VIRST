import glob
import json
import logging
import os
import random

import cv2
import numpy as np

from data.base_dataset import BaseVirstDataset
from data.data_processing import get_mask_from_json
from data.dataset_config import LISA_ROOT
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.image_utils import read_image_sam2
from utils.preprocess import preprocess_virst
from utils.utils import ANSWER_LIST, TASK_IMAGE_SINGLE_SEG, TASK_IMAGE_TEXT_ONLY

logger = logging.getLogger(__name__)

class ReasonSegDataset(BaseVirstDataset):

    def __init__(
        self,
        tokenizer,
        data_args,
        
        num_classes_per_sample: int = 3,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        reason_seg_data: str = "ReasonSeg|train",
        explanatory: float = 0.1,
        train: bool = True,
    ):
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.explanatory = explanatory

        self.data_args = data_args
        self.base_image_dir = LISA_ROOT
        self.tokenizer = tokenizer
        self.answer_list = ANSWER_LIST
        
        self.seg_image_size = data_args.seg_image_size
        self.seg_token_num = 1 # NOTE: the number of objects is ALWAYS 1 for reason_seg

        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size, square=True)

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    self.base_image_dir, "reason_seg", split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        logger.info("Loaded %d reason_seg samples", len(images))

        if explanatory != -1:
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    self.base_image_dir,
                    "reason_seg",
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            logger.info("Loaded %d explanation entries", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch
    
    def _get_item(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]
        processor = self.data_args.image_processor

        # image is used to get the shape 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        
        # preprocess image for clip
        image = image[np.newaxis, :, :] # (H, W, C) -> (1, H, W, C)
        images_clip = processor.preprocess(image, return_tensors="pt")["pixel_values"]
        images_sam = read_image_sam2(image_path)
        images_sam = self.sam2_transform(images_sam)

        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2 # use vanilla text explanation (no segmentation mask)
            else:
                choice = random.randint(0, 1)

        input_ids = []
        labels = []
        questions = []
        modalities = []
        answers = []
        masks = []
        
        for idx in sampled_inds:
            expression = sents[idx]
            
            # NOTE: the number of objects is ALWAYS 1 for reason_seg
            mask_np = np.any(mask, axis=0) # (O=1, h', w') -> (h', w')
            mask_tensor = self.sam2_mask_transform(mask_np)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (O=1, T=1, h', w') temporal dimension is 1
            
            masks.append(mask_tensor)
            modalities.append("image")
            
            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            
            if self.explanatory != -1 and img_name in self.img_to_explanation:    
                if choice == 0:  # [SEG] token
                    add_explanation = None
                    only_text = False
                    task_prompt = TASK_IMAGE_SINGLE_SEG
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    add_explanation = self.img_to_explanation[image_name]["outputs"]
                    only_text = False 
                    task_prompt = TASK_IMAGE_SINGLE_SEG
                elif choice == 2:
                    image_name = image_path.split("/")[-1]
                    add_explanation = self.img_to_explanation[image_name]["outputs"]
                    only_text = True
                    task_prompt = TASK_IMAGE_TEXT_ONLY
                else:
                    raise ValueError("Not implemented yet.")
                    
                out = preprocess_virst(
                    expression, 
                    self.tokenizer, 
                    has_image=True, 
                    seg_token_num=self.seg_token_num, 
                    add_explanation=add_explanation, 
                    only_text=only_text,
                    task_prompt=task_prompt
                )  
                input_ids.append(out["input_ids"])
                labels.append(out["labels"])
                questions.append("reason_seg_" +out["question"])
            else:
                raise ValueError("Not implemented yet.")

        return {
            "image_path": image_path,
            "images_sam": images_sam,
            "images_clip": images_clip,
            "masks": masks,
            "input_ids": input_ids,
            "labels": labels,
            "resize": None,
            "modalities": modalities,
            "questions": questions,
            "answers": answers,
        }
