import glob
import json
import logging
import os
import os.path as osp
import random

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from data.dataset_config import LISA_ROOT
from data.base_dataset import BaseVirstDataset
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.image_utils import read_image_sam2
from utils.preprocess import preprocess_virst
from utils.utils import TASK_IMAGE_MULTI_SEG, TASK_IMAGE_SINGLE_SEG

logger = logging.getLogger(__name__)

def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    logger.info("Loaded %d Mapillary images", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    logger.info("Loaded %d ADE20K images", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    logger.info("Loaded %d COCOStuff images", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    logger.info("Loaded %d PACO-LVIS images", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    logger.info("Loaded %d Pascal-Part images", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(BaseVirstDataset):

    def __init__(
        self,
        tokenizer,
        data_args,
        
        num_classes_per_sample  : int   = 3,                    # only for train 
        samples_per_epoch       : int   = 500 * 8 * 2 * 10,     # only for train
        sem_seg_data           : str   = "ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        train                   : bool  = True,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_classes_per_sample = num_classes_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.sem_seg_data = sem_seg_data
        self.train = train
        self.base_image_dir = osp.join(LISA_ROOT, "sem_seg")
        self.seg_image_size = data_args.seg_image_size
        
        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size, square=True)
        
        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        self.init_fn_map = {
            "ade20k": init_ade20k,
            "cocostuff": init_cocostuff,
            "mapillary": init_mapillary,
            "pascal_part": init_pascal_part,
            "paco_lvis": init_paco_lvis,
        }
        for ds in self.sem_seg_datas:
            init_fn = self.init_fn_map[ds]
            classes, images, labels = init_fn(self.base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

    def __len__(self):
        return self.samples_per_epoch

    def _get_item(self, idx):
        
        def _get_expression(sampled_cls):
            if isinstance(sampled_cls, tuple):
                obj, part = sampled_cls
                if random.random() < 0.5:
                    expression = obj + " " + part
                else:
                    expression = "the {} of the {}".format(part, obj)
            else:
                expression = sampled_cls
                
            return expression
        
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]
        
        sampled_classes = []
        
        input_ids = []
        labels = []
        questions = []
        modalities = []
        masks = []

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            else:
                raise ValueError(f"Invalid dataset: {ds}")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self._get_item(0)
            
            
            unique_categories = list(set([class_map[ann["category_id"]] for ann in anns]))
            
            if len(unique_categories) > self.num_classes_per_sample:
                sampled_category_idx = np.random.choice(
                    len(unique_categories), size=self.num_classes_per_sample, replace=False
                ).tolist()
                sampled_classes = [unique_categories[i] for i in sampled_category_idx]
            else:
                sampled_classes = unique_categories
                
            # get mask for each sampled class
            for sampled_cls in sampled_classes:
                
                mask = [] # mask for multi-object segmentation
                for ann in anns:
                    if class_map[ann["category_id"]] == sampled_cls:
                        mask.append(coco_api.annToMask(ann))
                
                # NOTE: the number of objects may vary for pascal_part, paco_lvis
                mask_tensor = self.sam2_mask_transform(mask) # (O, h', w')
                # NOTE: the number of objects is always 1 for now!!!!!!!!!!!!!!!!!!
                mask_tensor = mask_tensor.any(dim=0).unsqueeze(0) # (O, h', w') -> (1, h', w') 
                mask_tensor = mask_tensor.unsqueeze(1) # (O, T=1, h', w') temporal dimension is 1
                masks.append(mask_tensor)
                
                expresssion = _get_expression(sampled_cls)
                out = preprocess_virst(
                    expresssion, 
                    self.tokenizer, 
                    has_image=True, 
                    seg_token_num= mask_tensor.shape[0],
                    task_prompt=TASK_IMAGE_MULTI_SEG
                )
                input_ids.append(out["input_ids"])
                labels.append(out["labels"])
                questions.append(f"{ds}_{out['question']}")
                modalities.append("image")

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, classes = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            class_path = classes[idx]
            cur_class = Image.open(class_path)
            cur_class = np.array(cur_class)
            if ds == "ade20k":
                cur_class[cur_class == 0] = 255
                cur_class -= 1
                cur_class[cur_class == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        cur_class[cur_class == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            resize = image.shape[:2]
            unique_classes = np.unique(cur_class).tolist()
            if 255 in unique_classes:
                unique_classes.remove(255)
            if len(unique_classes) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_classes]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes
            
            # get mask for each sampled class
            for sampled_cls in sampled_classes:
                
                class_id = self.data2classes[ds].tolist().index(sampled_cls)
                class_mask = (cur_class == class_id).astype(np.uint8)
                
                # NOTE: the number of objects is ALWAYS 1 for sem_seg
                mask_tensor = self.sam2_mask_transform(class_mask) # (h', w')
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (O=1, T=1, h', w') temporal dimension is 1
                masks.append(mask_tensor)
                
                expresssion = _get_expression(sampled_cls)
                # NOTE: the number of objects is ALWAYS 1 for ade20k, cocostuff, mapillary
                out = preprocess_virst(
                    expresssion, 
                    self.tokenizer, 
                    has_image=True, 
                    seg_token_num= 1,
                    task_prompt=TASK_IMAGE_SINGLE_SEG
                ) 
                input_ids.append(out["input_ids"])
                labels.append(out["labels"])
                questions.append(f"{ds}_{out['question']}")
                modalities.append("image")
                
        else:
            raise ValueError(f"Invalid dataset: {ds}")

        image = image[np.newaxis, :, :] # (H, W, C) -> (1, H, W, C)
        images_clip = self.data_args.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        images_sam = read_image_sam2(image_path)
        images_sam = self.sam2_transform(images_sam)
        resize = None
            
        return {
            "image_path": image_path,
            "images_sam": images_sam,
            "images_clip": images_clip,
            "masks": masks, # list of (O=1, 1, h', w') with length of num_conv
            "input_ids": input_ids,
            "labels": labels,
            "resize": resize,
            "exp_pair": sampled_classes,
            "modalities": modalities,
            "questions": questions,
        }
