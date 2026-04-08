import os
import logging
import cv2
from tqdm import tqdm

from data.base_dataset import BaseVirstDataset
from data.dataset_config import LISA_ROOT
from data.refer import REFER
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.image_utils import read_image_sam2
from utils.preprocess import preprocess_virst
from utils.utils import TASK_IMAGE_SINGLE_SEG

logger = logging.getLogger(__name__)


class ValReferSegDataset(BaseVirstDataset):
    def __init__(
        self,
        tokenizer,
        data_args,
        dataset: str = "refcoco",      # only refcoco validation
        split: str = "val",            # val / testA / testB
        samples_per_epoch: int = None  # not used in eval
    ):
        super().__init__(tokenizer, data_args)
        if dataset in ["refcoco", "refcoco+"]:
            assert split in ["val", "testA", "testB"], f"Invalid split {split} for {dataset}"
        else:  # refcocog
            assert split in ["val", "test"], f"Invalid split {split} for {dataset}"

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.dataset = dataset
        self.split = split
        self.base_image_dir = os.path.join(LISA_ROOT, "refer_seg")

        self.seg_image_size = data_args.seg_image_size
        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size, square=True)

        # load refcoco validation split
        self.load_data_eval()

    def load_data_eval(self):
        splitBy_map = {"refcoco": "unc", "refcoco+": "unc", "refcocog": "umd"}
        refer_api = REFER(self.base_image_dir, self.dataset, splitBy_map[self.dataset])
        ref_ids_eval = refer_api.getRefIds(split=self.split)
        refs_eval = refer_api.loadRefs(ref_ids=ref_ids_eval)
        images_ids_eval = refer_api.getImgIds(ref_ids=ref_ids_eval)
        loaded_images = refer_api.loadImgs(image_ids=images_ids_eval)

        self.annotations = refer_api.Anns
        self.records = []

        for ref in tqdm(refs_eval, desc=f"Loading {self.dataset} {self.split} split..."):
            image_info = next(img for img in loaded_images if img["id"] == ref["image_id"])

            fname = image_info["file_name"]  # e.g., COCO_train2014_xxx.jpg / COCO_val2014_xxx.jpg
            subdir = "train2014" if "train2014" in fname else "val2014"

            file_name = os.path.join(
                self.base_image_dir,
                "images/mscoco/images",
                subdir,
                fname
            )

            for s in ref["sentences"]:
                sent_text = " ".join(s["sent"].lower().split())
                sent_id   = s["sent_id"]         
                self.records.append({
                    "image_path": file_name,
                    "image_info": image_info,
                    "ref_id": ref["ref_id"],
                    "ann_id": ref["ann_id"],
                    "sent_id": sent_id,
                    "sentence": sent_text,
                })

        logger.info(
            "Loaded %d samples for %s %s split",
            len(self.records),
            self.dataset,
            self.split,
        )

    def __len__(self):
        return len(self.records)

    def _get_item(self, idx):
        record = self.records[idx]
        image_path = record["image_path"]
        image_info = record["image_info"]
        ann_id = record["ann_id"]
        sentence = record["sentence"]

        # read and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processor = self.data_args.image_processor
        images_clip = processor.preprocess(image[np.newaxis, :, :], return_tensors="pt")["pixel_values"]

        images_sam = read_image_sam2(image_path)
        images_sam = self.sam2_transform(images_sam)

        # tokenize
        out = preprocess_virst(
            sentence,
            self.tokenizer,
            has_image=True,
            seg_token_num=1,
            task_prompt=TASK_IMAGE_SINGLE_SEG
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
            "questions": [record["sent_id"]],
            "video_paths": [image_path],
            "frame_ids": [[0]],
            "exp_ids": [record["ref_id"]],
        }
