import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
#from pyod.models.knn import KNN
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from scipy import ndimage

TEMPLATE={
    '01': [1,2,3],
    '02': [4,5],
    '03': [6],
    '04': [7],
    # '05': [8,9],
    '06': [10],
    # '07': [11, 12, 13, 14, 15,16,17,18,19,20,21,22,23],
    '08': [24],
    '09': [25,26],
    '10': [27],
    '05': [8]
        
    }

ORGAN_NAME = ["brain ET",
            "brain EC",
            "brain WT",
            "GTVp",
            "GTVn",
            "NPC",
            "Breast cancer",
            "Liver",
            "Liver cancer",
            "Colorectal cancer",
            "spleen",
            "right kidney", 
            "left kidney", 
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferiorvena cava",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
            "duodenum",
            "bladder cancer",
            "prostate",
            "prostate cancer1",
            "prostate cancer2"
            ]


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
    elif part_index == "CHA":
        template_key = '05'
    else:
        print("no task template_key")
    
    return template_key


if __name__ == "__main__":
    template_key = get_key_task('Liver-im0-t1c.nii')
    organ_list = TEMPLATE[template_key]
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, 29)) # 1st row for dice, 2nd row for count
    print(organ_list)
    print(dice_list)