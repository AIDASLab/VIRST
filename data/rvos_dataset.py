import os 
import os.path as osp
import random
import logging

import cv2
import numpy as np 
import pycocotools.mask as maskUtils
import torch
from tqdm import tqdm 

from data.base_dataset import BaseVirstDataset
from data.dataset_config import RVOS_DATA_INFO as _DATA_INFO 
from data.dataset_config import RVOS_ROOT 
from data.d2_datasets.refytvos_utils import load_refytvos_json
from data.d2_datasets.refytvos_val_videos import REFYTVOS_VAL_VIDEOS
from data.d2_datasets.mevis_utils import load_mevis_json
from data.transforms import SAM2Transform, SAM2MaskTransform
from utils.utils import TASK_VIDEO_MULTI_SEG
from utils.preprocess import preprocess_virst
from utils.video_utils import read_frames_sam2, sample_index_masks

logger = logging.getLogger(__name__)

class RVOSDataset(BaseVirstDataset):
    def __init__(
        self, 
        tokenizer,
        data_args,
        
        num_classes_per_sample  : int   = 3,                    # only for train 
        samples_per_epoch       : int   = 500 * 8 * 2 * 10,     # only for train
        num_frames_sample_range : str   = "8,12",
        rvos_sample_ratio       : str   = '4000||15000||400||6000||6000', # only for train
        rvos_seg_data           : str   = "mevis_train||refytvos_train||davis17_train||revos_train||restvos_train",
        rvos_sample_policy      : str   = "uniform", # all, uniform, random
        rvos_sample_list        : list  = [],
        output_dir              : str   = "/home/csjihwanh/virst/test/mask_results", # only for inference
        train                   : bool  = True,
        only_question           : bool  = False,
    ):
        super().__init__(tokenizer, data_args)
        self.train = train 
        self.root = RVOS_ROOT
        self.output_dir = output_dir
        self.num_classes_per_sample = num_classes_per_sample
        self.seg_image_size = data_args.seg_image_size
        self.seg_token_num = data_args.seg_token_num
        self.seg_image_length = data_args.seg_image_length
        self.only_question = only_question # in the inference time 
        self.data_anno = {
            "video_read_type": 'rvos'
        }
        self.samples_per_epoch = samples_per_epoch
        self.rvos_seg_ds_list = rvos_seg_data.split("||")
        rvos_sample_ratio = np.array([float(x) for x in rvos_sample_ratio.split("||")])
        self.rvos_sample_ratio = rvos_sample_ratio / rvos_sample_ratio.sum()
        assert rvos_sample_policy in ["random", "uniform", "all", "flex"], f"invalid rvos_sample_policy {rvos_sample_policy}"
        
        self.rvos_sample_policy = rvos_sample_policy 
        self.rvos_sample_list = rvos_sample_list
        self.num_frames_sample_range = [int(x) for x in num_frames_sample_range.split(",")]
        assert len(self.num_frames_sample_range) == 2 and self.num_frames_sample_range[0] <= self.num_frames_sample_range[1], f"invalid num_frames_sample_range {num_frames_sample_range}"
        
        self.num_be_called = 0
        self.rvos_seg_data = {}

        self.sam2_transform = SAM2Transform(size=self.seg_image_size)
        self.sam2_mask_transform = SAM2MaskTransform(size=self.seg_image_size, square=True)

        if train: 
            self.load_data_train()
        else: 
            self.load_data_eval()

    def _get_mask(self, mask_dict, anno_ids, frame_ids, image_shape):
        obj_mask_seqs = []
        for anno_id in anno_ids:
            mask_seq = []
            for frame_id in frame_ids:
                seg_mask = mask_dict[anno_id][frame_id]
                if seg_mask is not None:
                    m = maskUtils.decode(seg_mask)
                    if m.ndim == 3:
                        m = m.sum(axis=2).astype(np.uint8)
                    else:
                        m = m.astype(np.int8)
                else:
                    m = np.zeros(image_shape, dtype=np.uint8)
                mask_seq.append(m)
            obj_mask_seqs.append(np.stack(mask_seq,axis=0)) # (T, H, W) append 
            
        if not obj_mask_seqs: # no mask 
            T = len(frame_ids)
            H, W = image_shape 
            obj_mask_seqs.append(np.zeros((T, H, W), dtype=np.uint8))
        
        obj_mask_array = np.stack(obj_mask_seqs, axis=0) # (O, T, H, W)

        return obj_mask_array 

    def load_data_train(self):
        for dataset in self.rvos_seg_ds_list:
            assert dataset in _DATA_INFO.keys(), f"dataset {dataset} not found!"
            logger.info("Loading %s into memory", dataset)

            image_root, json_file = _DATA_INFO[dataset]
            image_root = osp.join(self.root, image_root)
            json_file = osp.join(self.root, json_file)
            if 'mevis' in dataset or 'revos' in dataset or 'lvvis' in dataset or 'restvos' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_mevis_json(image_root, json_file, dataset, is_train = True)
            elif 'refytvos' in dataset or 'davis' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_refytvos_json(image_root, json_file, dataset)
            else:
                raise ValueError(f"Unknown dataset name: {dataset}")
            logger.info(
                "Loaded %s: %d expressions, %d videos, %d masks",
                dataset,
                len(metas),
                len(vid2metaid),
                len(mask_dict),
            )
            
            self.rvos_seg_data[dataset] = {
                'image_root': image_root,
                'json_file' : json_file,
                'metas'     : metas,
                'mask_dict' : mask_dict,
                'is_train'  : is_train,
                'vid2metaid': vid2metaid,
            }

    def load_data_eval(self):
        assert len(self.rvos_seg_ds_list) == 1, "Only a single dataset is allowed during eval"
        dataset = self.rvos_seg_ds_list[0]

        self.d2_dataset_dicts = []
        image_root, json_file = _DATA_INFO[dataset]
        image_root = osp.join(self.root, image_root)
        json_file = osp.join(self.root, json_file)        

        
        if 'mevis' in dataset or 'revos' in dataset or 'lvvis' in dataset or 'restvos' in dataset:
            if 'test' in dataset:
                is_test = True
            else:
                is_test = False
            metas, mask_dict, vid2metaid, is_train = load_mevis_json(image_root, json_file, dataset, is_train=False, is_test=is_test)
        elif 'refytvos' in dataset or 'davis' in dataset:
            if 'refytvos' in dataset:
                mask_path = os.path.join(image_root, 'mask_dict_valid.pkl')
            else:
                mask_path = None 
            metas, mask_dict, vid2metaid, is_train = load_refytvos_json(image_root, json_file, dataset, mask_dict_path=mask_path)
        
        if dataset == 'davis17_valid':
            mask_dict=None
            
        self.mask_dict_eval = mask_dict 
        
        for vid_idx, vid_dict in tqdm(enumerate(metas), desc=f'Loading {dataset} ... '):
            record = {}
            if (dataset == "refytvos_valid") and (vid_dict['video'] not in REFYTVOS_VAL_VIDEOS):
                continue
            record["video_path"] = os.path.join(image_root, 'JPEGImages', vid_dict['video'])
            record["file_names"] = [
                os.path.join(image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') 
                    for i in range(vid_dict["length"])
                ]
            record["length"] = vid_dict["length"]
            video_name, exp, anno_ids, obj_ids, category, exp_id = \
                vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']

            exp = " ".join(exp.lower().split())
            if "eval_idx" in vid_dict:
                record["eval_idx"] = vid_dict["eval_idx"]

            record["sentence"]    = exp
            record["exp_id"]      = exp_id
            record["video_name"]  = video_name
            record["anno_ids"]    = anno_ids
            self.d2_dataset_dicts.append(record)
        
        logger.info("Loaded %d evaluation samples", len(self.d2_dataset_dicts))

    def __len__(self):
        if self.train:
            return self.samples_per_epoch
        else:
            return len(self.d2_dataset_dicts)

    def _get_item(self, idx):
        if self.train:
            data = self.sample_data()
        else:
            data = self.sample_data_eval(idx)
        
        frame_path_list = data['video_frame_path_list']

        frames, frame_indices, msg = self.process_video_vlm(frame_path_list, data_anno=self.data_anno, data_args=self.data_args)


        processor = self.data_args.image_processor

        frames_clip = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        
        seg_image_length = min(self.seg_image_length, len(frame_indices))
        frame_indices_for_sam = sorted(random.sample(frame_indices, seg_image_length), key=frame_indices.index)
        frames_sam = read_frames_sam2(frame_path_list, frame_indices_for_sam)
        frames_sam = self.sam2_transform(frames_sam)

        masks = []
        input_ids = []
        labels = []
        modalities = []
        questions = []
        video_paths = [] # for inference
        frame_ids = [] # for inference
        exp_ids = [] # for inference
        num_conv = 0

        for expression, mask in data['exp_mask_pairs']:
            text = expression.strip()
            assert len(text.split('||')) == 1

            mask_per_exp = [] 
            if mask is not None:
                for mask_per_obj in mask: # (O, T, H, W) -> (T, H, W)
                    mask_per_obj_indexed = sample_index_masks(mask_per_obj, frame_indices_for_sam) # (T, h', w')
                    mask_per_obj_indexed = self.sam2_mask_transform(mask_per_obj_indexed)
                    mask_per_exp.append(mask_per_obj_indexed)
                
                mask_per_exp = torch.stack(mask_per_exp, dim=0) # (O, T, h', w')
                mask_per_exp = mask_per_exp.any(dim=0).unsqueeze(0) # (O, T, h', w') -> (1, T, h', w')
            else:
                mask_per_exp = torch.zeros((1, len(frame_indices_for_sam), self.seg_image_size, self.seg_image_size))
            
            out = preprocess_virst(
                expression, 
                self.tokenizer, 
                has_image=True, 
                seg_token_num=10,
                task_prompt=TASK_VIDEO_MULTI_SEG,
                only_question = self.only_question
            )
            input_ids.append(out["input_ids"])
            labels.append(out["labels"])
            
            masks.append(mask_per_exp)
            modalities.append("video")
            num_conv += 1
            questions.append(f"{data['ds']}_{text}")
            
            video_paths.append(data.get("video_path", None))
            if "frame_ids" in data and data["frame_ids"] is not None:
                frame_ids_for_sam = [data["frame_ids"][i] for i in frame_indices_for_sam]
                frame_ids.append(frame_ids_for_sam)
            else:
                frame_ids.append(None)
            exp_ids.append(data.get("exp_id", None))

        resize=None
        
        return {
            "image_path": ','.join(frame_path_list),
            "images_sam": frames_sam,
            "images_clip": frames_clip,
            "masks": masks,
            "input_ids": input_ids,
            "labels": labels,
            "resize": resize,
            "modalities": modalities,
            "questions": questions,
            "video_paths": video_paths, # for inference
            "frame_ids": frame_ids, # for inference
            "exp_ids": exp_ids, # for inference
        }
        
    def sample_data_eval(self,idx):
        num_frames_per_sample = np.random.randint(self.num_frames_sample_range[0], self.num_frames_sample_range[1] + 1)
        
        data_d2     = self.d2_dataset_dicts[idx]
        frame_path  = data_d2['file_names']
        anno_ids    = data_d2['anno_ids']
        image_shape = cv2.imread(frame_path[0]).shape[:2]
        length = len(data_d2['file_names'])
        
        if length > num_frames_per_sample:
            if self.rvos_sample_policy == "random":
                frame_ids = np.random.choice(length, num_frames_per_sample, replace=False).tolist()
                frame_ids = sorted(frame_ids)
            elif self.rvos_sample_policy == "uniform":
                num_length = length
                split_point = np.linspace(0, num_length, num=num_frames_per_sample+1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i+1]) for i in range(num_frames_per_sample)]
            elif self.rvos_sample_policy == "all":
                frame_ids = list(range(length))
            elif self.rvos_sample_policy == "flex":
                num_length = length

                target_frames = min(64, num_length)
                target_frames = (target_frames // 4) * 4

                split_point = np.linspace(0, num_length, num=target_frames + 1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(target_frames)]
        elif length == num_frames_per_sample:
            frame_ids = list(range(length))
        else:
            frame_ids = list(range(length))
            frame_ids += [frame_ids[-1]] * (num_frames_per_sample - len(frame_ids))
        
        video_frame_path_list = [data_d2["file_names"][i] for i in frame_ids]
        
        if self.mask_dict_eval is not None:
            obj_mask_array = self._get_mask(
                mask_dict= self.mask_dict_eval, 
                anno_ids = anno_ids,
                frame_ids=frame_ids ,
                image_shape=image_shape
            )
        else :
            obj_mask_array = None
        
        data = {
            "video_name"           : data_d2['video_name'],
            "video_frame_path_list": video_frame_path_list,
            "exp_mask_pairs"       : [(data_d2['sentence'], obj_mask_array)],
            "length"               : len(data_d2['file_names']),
            "video_path"           : data_d2['video_path'],
            "frame_ids"            : frame_ids,
            "exp_id"               : data_d2['exp_id'],
            "ds"                   : self.rvos_seg_ds_list[0],
        }
        
        return data 

    def sample_data(self):
        ds         = np.random.choice(list(range(len(self.rvos_seg_ds_list))), p=self.rvos_sample_ratio)
        ds         = self.rvos_seg_ds_list[ds]
 
        metas      = self.rvos_seg_data[ds]['metas']
        mask_dict  = self.rvos_seg_data[ds]['mask_dict']
        image_root = self.rvos_seg_data[ds]['image_root']
        vid2metaid = self.rvos_seg_data[ds]['vid2metaid']
        
        vid = np.random.choice(list(vid2metaid.keys()))
        meta_ids = vid2metaid[vid]
        meta_ids = np.random.choice(meta_ids, min(self.num_classes_per_sample, len(meta_ids)), replace=False)
        video_name = metas[meta_ids[0]]['video']
        assert all([metas[meta_id]['video'] == video_name for meta_id in meta_ids]), "video name not match"

        record = {}
        vid_dict_first = metas[meta_ids[0]]
        record["file_names"] = [
            os.path.join(image_root, 'JPEGImages', vid_dict_first['video'], vid_dict_first["frames"][i]+ '.jpg') 
            for i in range(vid_dict_first["length"])
        ]
        record["length"] = vid_dict_first["length"]

        if len(self.rvos_sample_list) > 0:
            num_frames_per_sample = self.rvos_sample_list[self.num_be_called % len(self.rvos_sample_list)]
            self.num_be_called += 1
        else:
            num_frames_per_sample = np.random.randint(self.num_frames_sample_range[0], self.num_frames_sample_range[1] + 1)

        if vid_dict_first["length"] > num_frames_per_sample:
            if self.rvos_sample_policy == "random":
                frame_ids = np.random.choice(vid_dict_first["length"], num_frames_per_sample, replace=False).tolist()
                frame_ids = sorted(frame_ids)
            elif self.rvos_sample_policy == "uniform":
                num_length = vid_dict_first["length"]
                split_point = np.linspace(0, num_length, num=num_frames_per_sample+1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i+1]) for i in range(num_frames_per_sample)]
            elif self.rvos_sample_policy == "all":
                frame_ids = list(range(vid_dict_first["length"]))

            elif self.rvos_sample_policy == "flex":
                num_length = vid_dict_first["length"]

                target_frames = min(64, num_length)
                target_frames = (target_frames // 4) * 4

                split_point = np.linspace(0, num_length, num=target_frames + 1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(target_frames)]
        else:
            frame_ids = list(range(vid_dict_first["length"]))
        video_frame_path_list = [record["file_names"][i] for i in frame_ids]
        image_shape = cv2.imread(record["file_names"][0]).shape[:2]

        exp_mask_pairs = []
        for meta_id in meta_ids:
            vid_dict = metas[meta_id]
            assert vid_dict['video'] == video_name, "video name not match"
            assert vid_dict['length'] == vid_dict_first['length'], "video length not match"
            anno_ids = vid_dict['anno_id']
            obj_ids = vid_dict['obj_id']
            exp = vid_dict['exp']
            if 'lvvis' in ds:
                exp = exp.replace('_', ' ')     

            obj_mask_array = self._get_mask(
                mask_dict= mask_dict, 
                anno_ids= anno_ids,
                frame_ids=frame_ids,
                image_shape=image_shape
            )
            
            exp_mask_pairs.append((exp, obj_mask_array))


        data = {
            "video_name"           : video_name,
            "video_frame_path_list": video_frame_path_list,
            "exp_mask_pairs"       : exp_mask_pairs, # (O, T, H, W)
            "video_path"           : None,
            "ds"                   : ds,
        }

        return data
    
