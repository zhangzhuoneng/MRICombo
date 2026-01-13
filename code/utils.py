import logging
import sys
import datetime
import pytz  
from MOE_dataset_seg import position_seg_dict
from MOE_dataset_cls import position_cls_dict
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from typing import Sequence
from medpy.metric import binary

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from scipy import ndimage

class ChinaTimeFormatter(logging.Formatter):
    def converter(self, timestamp):
      
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc).astimezone(pytz.timezone('Asia/Shanghai'))
        return dt

    def formatTime(self, record, datefmt=None):
        
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime("%Y-%m-%d %H:%M:%S")
        return s

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr
    
def adjust_learning_all_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    
    # 获取实际的优化器
    actual_optimizer = optimizer._optim if hasattr(optimizer, '_optim') else optimizer
    actual_optimizer.param_groups[0]['lr'] = lr
    
    return lr
def weight_base_init(nn_dataset,task):
   
    position_num_dict = {}
    if task=="seg":
        position_prompt_dict = position_seg_dict
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径列表
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
        
        position_weight_dict = {}
        for position in position_num_dict:
            position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])

       
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]
    elif task=="cls":
        position_prompt_dict = position_cls_dict
        
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径列表
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
       
        position_weight_dict = {}
        for position in position_num_dict:
           
            # position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])
            if  position!=1 or position!=2: 
                position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])
            else:
                position_weight_dict[position] = 0.2
            # else:
      
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]

    else:
        print("no seg or cls task")
    return all_sample_weight_list

def weight_base_init_new(nn_dataset,task):
   
    position_num_dict = {}
    if task=="seg":
        position_prompt_dict = position_seg_dict
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径列表
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
       
        position_weight_dict = {}
        for position in position_num_dict:
            
            if  position!=6:
                position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])
            else:
                position_weight_dict[position] = 1/3

      
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]
    elif task=="cls":
        position_prompt_dict = position_cls_dict
        
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径列表
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
           
        position_weight_dict = {}
        for position in position_num_dict:
            if  position!=1 or position!=2: 
                position_weight_dict[position] = 1 / np.sqrt(position_num_dict[position])
            else:
                position_weight_dict[position] = 1/5

       
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            # print(dataset_index, dataset_name)
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]

    else:
        print("no seg or cls task")
    return all_sample_weight_list
class WeightedRandomSamplerDDP(DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, data_set, weights: Sequence[float], num_replicas: int, rank: int, num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


TEMPLATE={
    '01': [1,2,3],
    '02': [4,5],
    '03': [6],
    '04': [7],
    '05': [8,9],
    '06': [10],
    '07': [11, 12, 13, 14, 15,16,17,18,19,20,21,22,23],
    '08': [24],
    '09': [25,26],
    '10': [27],
    }

ORGAN_NAME = [
            "brain enhanced tumor",
            "brain tumor core",
            "brain whole tumor",
            "gross tumor",
            "metastatic tumor",
            "nasopharyngeal tumor",
            "breast tumor",
            "liver1",
            "liver tumor",
            "colorectal tumor",
            "spleen",
            "right kidney", 
            "left kidney", 
            "gallbladder",
            "esophagus",
            "liver2",
            "stomach",
            "aorta",
            "inferiorvena cava",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
            "duodenum",
            "bladder tumor",
            "prostate",
            "prostate tumor1",
            "prostate tumor2"
            ]
# MpuA800.1156

def get_key_task(name):
    ## input: name
    ## output: the corresponding key
    sequence_index = name[-7:-4]
    part_index = name[:3]
    template_key = None
    # print(sequence_index,part_index)
    if part_index ==  "Bra":
        template_key = '01'
    elif part_index == "HNT":
        template_key = '02'
    elif part_index == "NPC":
        template_key = '03'
    elif part_index == "ISP":
        template_key = '04'
    elif part_index == "Liv":
        template_key = '05'
    elif part_index == "Col":
        template_key = '06'
    elif part_index == "amo": 
        template_key = '07'
    elif part_index == "cen": 
        template_key = '08'
    elif part_index == "Pro":
        template_key = '09'
    elif part_index == "csP":
        template_key = '10'
    elif part_index == "liv":
        template_key = '05'
    else:
        print("no task template_key")
    
    return template_key

def new_dice(pred, label):
    intersection = 2. * np.logical_and(pred, label).sum()
    union = pred.sum() + label.sum()
    return intersection / union

def iou_score(pred, label):
    """
    IoU (Intersection over Union) / Jaccard指数
    """
    intersection = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def asd_score(pred, gt, voxelspacing=None):
    """
    ASD (Average Surface Distance)
    """
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            asd = binary.asd(pred, gt, voxelspacing=voxelspacing)
            return asd
        except Exception as e:
            print(f"Error calculating ASD: {e}")
            return 0.0
    else:
        return 0.0

def assd_score(pred, gt, voxelspacing=None):
    """
    计算ASSD (Average Symmetric Surface Distance)
    """
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            assd = binary.assd(pred, gt, voxelspacing=voxelspacing)
            return assd
        except Exception as e:
            print(f"Error calculating ASSD: {e}")
            return 0.0
    else:
        return 0.0

def Hd_95(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0
if __name__ == "__main__":
    template_key = get_key_task('Liver-im0-t1c.nii')
    organ_list = TEMPLATE[template_key]
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, 29)) # 1st row for dice, 2nd row for count
    print(organ_list)
    print(dice_list)
    