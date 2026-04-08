import numpy as np
import random

from data.dataset_config import LISA_ROOT, RVOS_ROOT
from data.sem_seg_dataset import SemSegDataset
from data.refer_seg_dataset import ReferSegDataset
from data.reason_seg_dataset import ReasonSegDataset
from data.rvos_dataset import RVOSDataset
from data.base_dataset import BaseVirstDataset
from data.random_list import get_random_list
from data.video_vqa_dataset import VideoVQADataset
import logging

class HybridDataset(BaseVirstDataset):

    def __init__(
        self,
        tokenizer,
        data_args,
        
        samples_per_epoch=500 * 8 * 2 * 10,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg||video_vqa||rvos" ,
        sample_rate=[9, 3, 3, 1, 4, 12],
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        reason_seg_data="ReasonSeg|train",
        rvos_seg_data="mevis_train||refytvos_train||davis17_train||revos_train||lvvis_train",
        rvos_sample_ratio="4000||15000||400||3000||3000",
        num_frames_sample_range="8,12",
        video_sample_policy="uniform",
        explanatory=0.1,
        train=True,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = LISA_ROOT
        self.tokenizer = tokenizer

        self.datasets = dataset.split("||")
        self.num_datasets = len(self.datasets)

        self.num_be_called = 0
        
        self.dataset_sample_list = get_random_list(probabilities=self.sample_rate.tolist(), values=list(range(self.num_datasets)), length=samples_per_epoch)
        rvos_sample_range = [int(i) for i in num_frames_sample_range.split(',')]
        rvos_range_length = rvos_sample_range[-1] - rvos_sample_range[0] + 1
        self.rvos_sample_list = get_random_list(probabilities=[float(1/rvos_range_length) for _ in range(rvos_range_length)], values=list(range(rvos_sample_range[0],rvos_sample_range[-1]+1)), length=10000)

        self.all_datasets = []
        self.dataset_name = [] 
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        num_classes_per_sample=num_classes_per_sample,
                        tokenizer=tokenizer, 
                        data_args=data_args,
                        samples_per_epoch=samples_per_epoch,
                        sem_seg_data=sem_seg_data,
                        train=train,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        num_classes_per_sample=num_classes_per_sample,
                        tokenizer=tokenizer, 
                        data_args=data_args,
                        samples_per_epoch=samples_per_epoch,
                        refer_seg_data=refer_seg_data,
                        train=train,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        num_classes_per_sample=num_classes_per_sample,
                        tokenizer=tokenizer, 
                        data_args=data_args,
                        samples_per_epoch=samples_per_epoch,
                        reason_seg_data=reason_seg_data,
                        train=train,
                    )
                )
            elif dataset == "rvos":            
                self.all_datasets.append(
                    RVOSDataset(
                        num_classes_per_sample    = num_classes_per_sample,
                        tokenizer                 = tokenizer,
                        data_args                 = data_args,
                        samples_per_epoch         = samples_per_epoch,
                        rvos_seg_data             = rvos_seg_data,
                        num_frames_sample_range   = num_frames_sample_range,
                        rvos_sample_ratio         = rvos_sample_ratio,
                        rvos_sample_policy        = video_sample_policy,
                        train                     = train,
                    )
                )
            elif dataset == "video_vqa":
                self.all_datasets.append(
                    VideoVQADataset(
                        tokenizer=tokenizer, 
                        data_args=data_args,
                        samples_per_epoch=samples_per_epoch,
                        num_frames_sample_range=num_frames_sample_range,
                        sample_policy=video_sample_policy,
                    )
                )
            self.dataset_name.append(dataset)
    def __len__(self):
        return self.samples_per_epoch

    def _get_item(self, idx):
        self.num_be_called += 1
        ind = self.dataset_sample_list[self.num_be_called % self.samples_per_epoch]
        data = self.all_datasets[ind]
        dict_data = data._get_item(idx)
        # NOTE: This tagging is designed for 1-batch training only 
        dict_data["dataset_name"] = self.dataset_name[ind]
        return dict_data

