import os
import logging
import cv2
from tqdm import tqdm

from data.base_dataset import BaseVirstDataset
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.utils import TASK_IMAGE_SINGLE_SEG
from utils.preprocess import preprocess_virst
from utils.image_utils import read_image_sam2
from data.data_processing import get_mask_from_json
from data.dataset_config import LISA_ROOT

logger = logging.getLogger(__name__)


class ValReasonSegDataset(BaseVirstDataset):
    def __init__(
        self,
        tokenizer,
        data_args,
        split: str = "val",  # "train", "val", "test" 
        samples_per_epoch: int = None,  # not used in eval
    ):
        super().__init__(tokenizer, data_args)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.split = split
        self.base_image_dir = LISA_ROOT

        self.seg_image_size = data_args.seg_image_size
        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size, square=True)

        self.load_data_eval()

    def load_data_eval(self):
        split_dir = os.path.join(self.base_image_dir, "reason_seg", self.split)
        images = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".jpg")]
        jsons = [img.replace(".jpg", ".json") for img in images]

        self.records = []
        for img_path, json_path in tqdm(zip(images, jsons), total=len(images), desc=f"Loading reason_seg {self.split} split..."):
            image_name = os.path.basename(img_path)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, sents, _ = get_mask_from_json(json_path, image)

            for sid, sent_text in enumerate(sents):
                self.records.append({
                    "image_path": img_path,
                    "json_path": json_path,
                    "sentence": " ".join(sent_text.lower().split()),
                    "sent_id": sid,
                    "mask": None,  # numpy array
                })

        logger.info(
            "Loaded %d samples for reason_seg %s split",
            len(self.records),
            self.split,
        )

    def __len__(self):
        return len(self.records)

    def _get_item(self, idx):
        record = self.records[idx]
        image_path = record["image_path"]
        sentence = record["sentence"]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processor = self.data_args.image_processor
        images_clip = processor.preprocess(image[np.newaxis, :, :], return_tensors="pt")["pixel_values"]

        images_sam = read_image_sam2(image_path)
        images_sam = self.sam2_transform(images_sam)

        out = preprocess_virst(
            sentence,
            self.tokenizer,
            has_image=True,
            seg_token_num=1,
            task_prompt=TASK_IMAGE_SINGLE_SEG,
        )

        return {
            "image_path": image_path,
            "images_sam": images_sam,
            "images_clip": images_clip,
            "masks": None,
            "input_ids": [out["input_ids"]],
            "labels": [out["labels"]],
            "resize": None,
            "modalities": ["image"],
            "questions": [record["sentence"]],
            "video_paths": [image_path],
            "frame_ids": [[0]],
            "exp_ids": [record["sent_id"]],
        }
