###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import time
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as cocomask
import multiprocessing as mp
import logging 
import pickle
from datetime import datetime

from utils.evaluation.metrics import db_eval_iou, db_eval_boundary
from data.dataset_config import RVOS_ROOT
from data.dataset_config import RVOS_DATA_INFO as _DATA_INFO
from data.d2_datasets.refytvos_val_videos import REFYTVOS_VAL_VIDEOS
from data.d2_datasets.refytvos_utils import load_refytvos_json

NUM_WOEKERS = 48

def eval_queue(q, rank, out_dict, mevis_pred_path,dataset, log_dir):
    logger = logging.getLogger()
    logger.handlers.clear()
    handler = logging.FileHandler(osp.join(log_dir, f'worker_{rank}.log'))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    while not q.empty():
        # print(q.qsize())
        
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'
        
        if 'revos' in dataset:
            vid_name = vid_name.split("/")[-1]
        if not os.path.exists(f'{mevis_pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue
        
        pred_0_path = f'{mevis_pred_path}/{vid_name}/{exp}/00000.png'
        logging.info(pred_0_path)
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)
            
            pred_mask_path = f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png'
            pred_masks[frame_idx] = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            if pred_masks[frame_idx] is None:
                logging.info(f"Failed to load mask {pred_mask_path}")
        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]
        logging.info(f"{j},{f},{vid_name},{exp}")

def check_pred_vs_gt(exp_dict, pred_path, dataset):
    """
    exp_dict: GT annotation dict (video -> expressions)
    pred_path: prediction root dir
    dataset: dataset name (for handling revos/restvos naming)
    """
    print("\n[Check GT vs Prediction Directory Structure]")

    for vid_name, vid in exp_dict.items():
        vid_dir = vid_name.split("/")[-1] if 'revos' in dataset else vid_name
        gt_exprs = set(map(int, vid['expressions'].keys()))  # GT expression indices
        pred_dir = osp.join(pred_path, vid_dir)

        if not osp.exists(pred_dir):
            print(f"[MISSING VIDEO] {vid_dir} not found in pred_dir")
            continue

        pred_exprs = set()
        for d in os.listdir(pred_dir):
            if d.isdigit():
                pred_exprs.add(int(d))

        missing = gt_exprs - pred_exprs
        extra   = pred_exprs - gt_exprs

        print(f"Video={vid_dir} | GT exprs={len(gt_exprs)} | Pred exprs={len(pred_exprs)}")
        if missing:
            print(f"  -> Missing expr: {sorted(missing)}")
        if extra:
            print(f"  -> Extra expr: {sorted(extra)}")
    print("[Done checking]\n")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str, )
    parser.add_argument("--dataset", type=str, default='refytvos_valid')
    parser.add_argument("--save_name", type=str, default="mevis_val_u.json")
    args = parser.parse_args()
    queue = mp.Queue()
    
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    log_dir = f'logs/{args.dataset}_eval_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=f'{log_dir}/main_log.txt', level=logging.INFO)
    
    dataset = args.dataset
    assert dataset in _DATA_INFO.keys(), f"dataset {dataset} not found!"
    
    root  = RVOS_ROOT
    image_root, json_file = _DATA_INFO[dataset]
    if 'mevis' in dataset or 'revos' in dataset or 'restvos' in dataset:
        mask_path = osp.join(root, image_root, 'mask_dict.json') 
        exp_path = osp.join(root, json_file)
        mask_dict = json.load(open(mask_path))
        exp_dict = json.load(open(exp_path))['videos']
    elif 'refytvos' in dataset:
        mask_path = osp.join(root, image_root, '../train/mask_dict.pkl') 
        exp_path = osp.join(root, json_file)
        with open(mask_path, 'rb') as f:
            mask_dict= pickle.load(f)
        # mask_dict = json.load(open(mask_path))
        exp_dict, mask_dict, _, _ = load_refytvos_json(
            img_folder = osp.join(root, image_root), 
            ann_file = exp_path, 
            dataset_name = dataset,
            mask_dict_path = mask_path,
        )
        # exp_dict = json.load(open(exp_path))['videos']

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    #check_pred_vs_gt(exp_dict, args.pred_path, dataset)

    for vid_name in exp_dict:
        if 'refytvos' in dataset and vid_name not in REFYTVOS_VAL_VIDEOS:
            print(vid_name, "continued..")
            continue
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.pred_path, dataset, log_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict, args.pred_path, dataset, log_dir)


    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    output_path = osp.join(log_dir, args.save_name)
    results = {
        'J'  : round(100 * float(np.mean(j)), 2),
        'F'  : round(100 * float(np.mean(f)), 2),
        'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(json.dumps(results, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))
