import glob
import json
import os

import cv2
import numpy as np
from pycocotools import mask as maskUtils

def get_mask_from_json(json_path, img) -> tuple[list[np.ndarray], list[str], bool]:
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_items = []   # (label_id, rle)

    for s in inform:
        label_id = s["label"]
        if label_id.lower() == "flag":      # deprecated annotations
            continue

        pts = np.asarray(s["points"], dtype=np.float64).reshape(-1).tolist()
        # frPyObjects expects a list of polygons (even if single); then merge parts
        rle_list = maskUtils.frPyObjects([pts], height, width)
        rle = maskUtils.merge(rle_list)

        area = float(maskUtils.area(rle))   # scalar
        area_list.append(area)
        valid_items.append((label_id, rle))
    
    # --- sort by area desc
    sort_index = np.argsort(area_list)[::-1].astype(np.int32).tolist()
    sorted_items = [valid_items[i] for i in sort_index]

    # --- per-polygon masks with your label values (1 or 255) ---
    masks: list[np.ndarray] = []
    for label_id, rle in sorted_items:
        m = maskUtils.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.uint8)

        label_value = np.uint8(255) # if ("ignore" in label_id.lower()) else 1
        if "ignore" in label_id.lower():
            label_value = 0
        # keep 0 for background; set foreground pixels to label_value
        m = (m > 0).astype(np.uint8) * label_value
        masks.append(m)

    return masks, comments, is_sentence
