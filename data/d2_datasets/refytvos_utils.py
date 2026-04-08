###########################################################################
# Created by: Xiaohongshu
# Email: clyanhh@gmail.com
# Copyright (c) 2023
###########################################################################


import json
import logging
import numpy as np
import os
import os.path as osp
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import pickle

import pycocotools.mask as maskUtils
from data.dataset_config import RVOS_DATA_INFO as _DATA_INFO 
from data.dataset_config import RVOS_ROOT 

try:
    from .categories import ytvos_category_dict
    from .categories import davis_category_dict
except:
    from categories import ytvos_category_dict
    from categories import davis_category_dict
"""
This file contains functions to parse Refer-Youtube-VOS dataset of COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

def find_existing_mask_path(base_path):
    """
    Try all common image extensions and return the first existing path.
    """
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No mask file found for base: {base_path}")

def encode_anno_mask(frames, vid_len, img_folder, video, exp_id, obj_id, anno_id, meta, ):
    anno_id_segm_list = list()
    for frame_idx in range(vid_len):
        frame_name = frames[frame_idx]
        base_path  = os.path.join(str(img_folder), 'Annotations', video,exp_id,  frame_name) 
        mask_path  = find_existing_mask_path(base_path)
        mask       = Image.open(mask_path).convert('P')
        mask       = np.array(mask, )
        mask       = (mask == 255) # (mask == obj_id) # 0, 1 binary
        segm       = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))) if mask.any() else None
        anno_id_segm_list.append(segm)
    assert len(anno_id_segm_list) == meta['length']
    assert len(anno_id_segm_list) == len(meta['frames'])
    return {str(anno_id): anno_id_segm_list}


def load_refytvos_json(img_folder: str, ann_file: str, dataset_name: str, mask_dict_path: str = None, is_train: bool = False):
    """
    img_folder (str)    : path to the folder where 'Annotations' && 'JPEGImages' && 'meta.json' are stored.
    ann_file (str)      : path to the json file.
    """

    def prepare_metas():
        if ('train' in dataset_name) or is_train:
            # read object information
            with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
                subset_metas_by_video = json.load(f)['videos']
            
            # read expression data
            with open(str(ann_file), 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            videos = sorted(list(subset_expressions_by_video.keys()))

            metas = []
            anno_count = 0  # serve as anno_id
            vid2metaid = defaultdict(list)
            for vid in videos:
                vid_meta   = subset_metas_by_video[vid]
                vid_data   = subset_expressions_by_video[vid]
                vid_frames = sorted(vid_data['frames'])
                vid_len    = len(vid_frames)

                exp_id_list = sorted(list(vid_data['expressions'].keys()))
                for exp_id in exp_id_list:
                    exp_dict            = vid_data['expressions'][exp_id]
                    meta                = {}
                    meta['video']       = vid
                    meta['exp']         = exp_dict['exp']
                    meta['obj_id']      = [0, ]  # Ref-Youtube-VOS only has one object per expression
                    meta['anno_id']     = [str(anno_count), ]
                    anno_count         += 1
                    meta['frames']      = vid_frames
                    meta['exp_id']      = exp_id
                    obj_id              = exp_dict['obj_id']
                    meta['obj_id_ori']  = int(obj_id)
                    meta['category']    = vid_meta['objects'][obj_id]['category']
                    meta['length']      = vid_len
                    metas.append(meta)
                    vid2metaid[vid].append(len(metas) - 1)
        else:
            # for some reasons the competition's validation expressions dict contains both the validation (202) & 
            # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
            # the validation expressions dict:
            assert 'valid' in dataset_name
            data = json.load(open(ann_file, 'r'))["videos"]
            valid_test_videos = set(data.keys())

            if "davis" in dataset_name.lower():
                # DAVIS17: validation set already clean
                valid_videos = valid_test_videos
            else: 
                test_meta_file = ann_file.replace('valid/meta_expressions.json', 'test/meta_expressions.json')
                test_data = json.load(open(test_meta_file, 'r'))["videos"]
                test_videos = set(test_data.keys())
                valid_videos = valid_test_videos - test_videos
            video_list = sorted([video for video in valid_videos])
            if "davis" in dataset_name.lower():
                assert len(video_list) == 30, f"error: got {len(video_list)} davis valid videos"
            else:
                assert len(video_list) == 202, 'error: incorrect number of validation videos'
            metas = [] # list[dict], length is number of expressions
            vid2metaid = defaultdict(list)
            anno_count = 0  # serve as anno_id

            for video in video_list:
                expressions = data[video]["expressions"]
                expression_list = list(expressions.keys()) 
                num_expressions = len(expression_list)
                video_len = len(data[video]["frames"])

                # read all the anno meta
                for i in range(num_expressions):
                    meta = {}
                    meta["video"]    = video
                    meta["exp"]      = expressions[expression_list[i]]["exp"]
                    meta['obj_id']   = [0, ]  # Ref-Youtube-VOS only has one object per expression
                    meta['anno_id']  = [str(anno_count), ]
                    anno_count         += 1
                    meta["frames"]   = data[video]["frames"]
                    meta["exp_id"]   = expression_list[i]
                    meta['category'] = 0
                    meta['length']   = video_len
                    metas.append(meta)
                    vid2metaid[video].append(len(metas) - 1)

        return metas, vid2metaid

    if dataset_name.startswith('refytvos'):
        category_dict = ytvos_category_dict
    elif dataset_name.startswith('davis'):
        category_dict = davis_category_dict
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))

    mask_json           = os.path.join(img_folder, 'mask_dict.pkl') if mask_dict_path is None else mask_dict_path
    read_mask_from_json = osp.exists(mask_json)
    is_train            = img_folder.split('/')[-1] == 'train'
    if read_mask_from_json:
        with open(mask_json, 'rb') as f:
            mask_dict = pickle.load(f)
    else:
        mask_dict = dict()  # need to be filled later, anno_id -> frame_id
    
    metas, vid2metaid = prepare_metas()
    return metas, mask_dict, vid2metaid, is_train



def generate_pkl(img_folder: str, ann_file: str, dataset_name: str, mask_dict_path: str = None, is_train: bool = False):
    """
    img_folder (str)    : path to the folder where 'Annotations' && 'JPEGImages' && 'meta.json' are stored.
    ann_file (str)      : path to the json file.
    """

    def prepare_metas():
        if ('train' in dataset_name) or is_train:
            # read object information
            with open(os.path.join(str(img_folder), 'meta.json'), 'r') as f:
                subset_metas_by_video = json.load(f)['videos']
            
            # read expression data
            with open(str(ann_file), 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            videos = sorted(list(subset_expressions_by_video.keys()))

            metas = []
            anno_count = 0  # serve as anno_id
            vid2metaid = defaultdict(list)
            for vid in videos:
                vid_meta   = subset_metas_by_video[vid]
                vid_data   = subset_expressions_by_video[vid]
                vid_frames = sorted(vid_data['frames'])
                vid_len    = len(vid_frames)

                exp_id_list = sorted(list(vid_data['expressions'].keys()))
                for exp_id in exp_id_list:
                    exp_dict            = vid_data['expressions'][exp_id]
                    meta                = {}
                    meta['video']       = vid
                    meta['exp']         = exp_dict['exp']
                    meta['obj_id']      = [0, ]  # Ref-Youtube-VOS only has one object per expression
                    meta['anno_id']     = [str(anno_count), ]
                    anno_count         += 1
                    meta['frames']      = vid_frames
                    meta['exp_id']      = exp_id
                    obj_id              = exp_dict['obj_id']
                    meta['obj_id_ori']  = int(obj_id)
                    meta['category']    = vid_meta['objects'][obj_id]['category']
                    meta['length']      = vid_len
                    metas.append(meta)
                    vid2metaid[vid].append(len(metas) - 1)
        else:
            
            mask_dict = {} 
            # for some reasons the competition's validation expressions dict contains both the validation (202) & 
            # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
            # the validation expressions dict:
            assert 'valid' in dataset_name
            data = json.load(open(ann_file, 'r'))["videos"]
            valid_test_videos = set(data.keys())
            test_meta_file = ann_file.replace('valid/meta_expressions.json', 'test/meta_expressions.json')
            test_data = json.load(open(test_meta_file, 'r'))["videos"]
            test_videos = set(test_data.keys())
            valid_videos = valid_test_videos - test_videos
            video_list = sorted([video for video in valid_videos])
            assert len(video_list) == 202, 'error: incorrect number of validation videos'
            metas = [] # list[dict], length is number of expressions
            vid2metaid = defaultdict(list)
            anno_count = 0  # serve as anno_id

            for video in tqdm(video_list, desc="Processing videos"):
                expressions = data[video]["expressions"]
                expression_list = list(expressions.keys()) 
                num_expressions = len(expression_list)
                video_len = len(data[video]["frames"])

                # read all the anno meta
                for i in range(num_expressions):
                    meta = {}
                    meta["video"]    = video
                    meta["exp"]      = expressions[expression_list[i]]["exp"]
                    meta['obj_id']   = [0, ]  # Ref-Youtube-VOS only has one object per expression
                    meta['anno_id']  = [str(anno_count), ]
                    meta["frames"]   = data[video]["frames"]
                    meta["exp_id"]   = expression_list[i]
                    meta['category'] = 0
                    meta['length']   = video_len
                    metas.append(meta)
                    vid2metaid[video].append(len(metas) - 1)
                    
                    mask = encode_anno_mask(
                        frames = meta["frames"],
                        vid_len = len(meta["frames"]),
                        img_folder = img_folder, 
                        exp_id=expression_list[i],
                        video = video,
                        obj_id = meta['obj_id'],
                        anno_id = str(anno_count),
                        meta = meta
                    )
                    
                    anno_count         += 1
                        
                    mask_dict.update(mask)

        return metas, vid2metaid, mask_dict 

    if dataset_name.startswith('refytvos'):
        category_dict = ytvos_category_dict
    elif dataset_name.startswith('davis'):
        category_dict = davis_category_dict
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))

    metas, vid2metaid, mask_dict  = prepare_metas()
    save_path  = osp.join(img_folder, 'mask_dict_valid.pkl')
    with open(save_path, "wb") as f:
        pickle.dump(mask_dict, f)
        
    print(f"saved at {save_path}")
    
    return 

if __name__ == '__main__':
    dataset = 'refytvos_valid'
    image_root, json_file = _DATA_INFO[dataset]
    image_root = osp.join(RVOS_ROOT, image_root)
    json_file = osp.join(RVOS_ROOT, json_file)
    generate_pkl(
        img_folder=image_root,
        ann_file=json_file,
        dataset_name = dataset,
        is_train=False
    )