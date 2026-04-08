import os
import os.path as osp
import random
import logging

import cv2
import numpy as np
from pycocotools import mask

from data.base_dataset import BaseVirstDataset
from data.dataset_config import LISA_ROOT
from data.grefer import G_REFER
from data.refer import REFER
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.image_utils import read_image_sam2
from utils.preprocess import preprocess_virst
from utils.utils import TASK_IMAGE_SINGLE_SEG

logger = logging.getLogger(__name__)

class ReferSegDataset(BaseVirstDataset):

    def __init__(
        self,
        tokenizer,
        data_args,
        
        num_classes_per_sample  : int   = 3,                    # only for train 
        samples_per_epoch=500 * 8 * 2 * 10,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        train=True,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_classes_per_sample = num_classes_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.base_image_dir = LISA_ROOT
        self.refer_seg_data = refer_seg_data
        
        self.seg_image_size = data_args.seg_image_size
    
        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size)
        self.train= train
        if self.train:
            self.load_data_train()
        else:
            self.load_data_eval()

    def load_data_train(self):
        
        DATA_DIR = os.path.join(self.base_image_dir, "refer_seg")
        self.refer_seg_ds_list = self.refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            logger.info(
                "Loaded %s (%s train split): %d images, %d annotations",
                ds,
                splitBy,
                len(refer_seg_ds["images"]),
                len(refer_seg_ds["annotations"]),
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds
        
    def load_data_eval(self):
        raise NotImplementedError("Evaluation is not implemented for ReferSegDataset")

    def __len__(self):
        return self.samples_per_epoch

    def _get_item(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self._get_item(0)

        sents = []
        ann_ids = []
        
        # randomly sample sentences 
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        
        sampled_classes = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        
        # read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip and sam 
        image = image[np.newaxis, :, :] # (H, W, C) -> (1, H, W, C)
        processor = self.data_args.image_processor
        images_clip = processor.preprocess(image, return_tensors="pt")["pixel_values"]
        images_sam = read_image_sam2(image_path)
        images_sam = self.sam2_transform(images_sam)
        
        input_ids = []
        labels = []
        modalities = []
        questions = []
            
        masks = []
        
        for index in sampled_inds:
            expression = sents[index]
            ann_id = ann_ids[index]
            
            expression = expression.strip()
            assert len(expression.split("||")) == 1
            out = preprocess_virst(expression, self.tokenizer, has_image=True, seg_token_num= 1, task_prompt=TASK_IMAGE_SINGLE_SEG) # refcoco has only one object
            input_ids.append(out["input_ids"])
            labels.append(out["labels"])
            questions.append(f"{ds}_{out['question']}_{image_path}")
            modalities.append("image")
        
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue
            
            else:
                ann = annotations[ann_id]

                if len(ann["segmentation"]) == 0:
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                    masks.append(m)
                    continue

                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"], image_info["height"], image_info["width"]
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = self.sam2_mask_transform(masks) # (num_conv, h', w')
        masks = masks.unsqueeze(1).unsqueeze(1) # (num_conv, O=1, 1, h', w') temporal dimension is 1

        resize=None

        return {
            "image_path": image_path,
            "images_sam": images_sam,
            "images_clip": images_clip,
            "masks": masks,
            "input_ids": input_ids,
            "labels": labels,
            "resize": resize,
            "exp_pair": sampled_classes,
            "modalities": modalities,
            "questions": questions,
        }
