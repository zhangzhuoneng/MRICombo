import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
# import cv2
import sys
sys.path.append("..")
sys.path.append("../../")
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from typing import Sequence

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
from dataset.aug_cls import *


position_cls_dict= {
    "11FedBca":0,
    "12NPC":1,
    "13LLD":2,
    "14BraTS":3,
    "15BreaDM":4
    
}

def list_add_prefix(txt_path, prefix_1):
    '''
    prefix_1: 
    prefix_2: 
    '''
    with open(txt_path, 'r') as f:
        lines = f.readlines()
      
    if prefix_1 is not None:
        filtered_lines = [line for line in lines if line.split('/')[1].startswith(prefix_1)]
        # print(filtered_lines)
        
        unique_names = set()
        for line in filtered_lines:

           
            filename = line.split('/')[-1]  
            unique_names.add(filename)
        
        return list(unique_names)
    
    else:
        return print('dataset error')
def get_subset_len(seg_dataset, text_path, split):
    subset_len = []
    for dataset_name in seg_dataset:
        subset_len.append(len(list_add_prefix(text_path, dataset_name)))
        # print(dataset_name,len(list_add_prefix(text_path, dataset_name)))
    return subset_len



class UniclsDataset(data.Dataset):
    def __init__(self, root, text_path, split,code = None, crop_size=(96, 128, 128), scale=True,
                 mirror=True):
        self.root = root
        self.text_path = text_path
        self.split = split
        self.code = code
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.is_mirror = mirror
        self.seg_use_dataset = []
        self.img_ids = [i_id.strip().split() for i_id in open(self.text_path)]
        # self.img_ids = [i_id.strip() for i_id in open(self.text_path)]
        # print(self.img_ids)
        self.files = []
        print("Start preprocessing....")
        self.cls_use_dataset = [
        "11FedBca",
        "12NPC",
        "13LLD",
        "14BraTS",
        "15BreaDM"
        ]
       
        self.sequences_AMB  =  ["_pre-dce","_pos-dce"]    
        self.sequences_BCa  =  ["_t2"]   
        self.sequences_NPC  =  ["_t1","_t1ce","_t2"]  
        self.sequences_LLD  =  ["_C+A","_C+V"] 
        self.sequences_Bra =   ["_t1","_t1ce","_t2","_flair"]
        # self.sequences_Bra =   ["-t1n","-t1c","-t2w","-t2f","-seg"]
        self.sequences_BrE =   ["_pre-dce","_pos-dce"]

        for item in self.img_ids:
            # print(item)
            patient_id, cls_label = item
            # print( patient_id, cls_label)
            cls_label = int(cls_label)
            part_id = int(patient_id[11:13])
            # print(part_id)
            if  part_id==10 or part_id==15:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_AMB]
                patient = dict(id=patient_id, pre_dce=paths[0], pos_dce=paths[1],label = cls_label)
                
            elif part_id==11:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_BCa]
                patient = dict(id=patient_id,t2=paths[0],label = cls_label)
                
            elif part_id==12:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_NPC]
                patient = dict(id=patient_id,t1=paths[0], t1ce=paths[1],t2=paths[2],label = cls_label)
                
            elif part_id==13:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_LLD]
                # print(paths)
                patient = dict(id=patient_id,c_a = paths[0],c_v = paths[1],label = cls_label)
                
            elif part_id==14:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_Bra]
                patient = dict(id=patient_id, t1=paths[0], t1ce=paths[1],t2=paths[2], flair=paths[3],label = cls_label)
                
            else:
                raise ValueError(f"Unknown part_id: {part_id}")
                
            # print(patient)
            self.files.append(patient)
        # print( self.files) 
        self.subset_len = get_subset_len(self.cls_use_dataset,self.text_path, split)
        print('{} cls {} images are loaded!'.format(len(self.files),self.split))

       
       
    def __len__(self):
        return len(self.files)

    def truncate(self, MR):
        mean_value = MR.mean()
        std_dev = MR.std()
        # normalize
        MR = (MR - mean_value) / (std_dev+1e-8)
        return MR

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        patient = self.files[index]

        patient_id = patient["id"]
        part_id = int(patient_id[11:13])  # 获取部位ID
        name =  patient_id.split('/')[-1]
        
        images = torch.stack([images[key].to(torch.float32) for key in images])
        label = patient["label"]
        
        for i in range(images.shape[0]):
             images[i] = self.truncate(images[i])#归一化
        # print(patient_id,images.shape,label)
        
        if self.split == 'train':
            images = images.unsqueeze(0)
            # if np.random.rand(1) <= 0.5:
            #     images, label = random_scale_rotate_translate_3d(images, label)
            #     images, label =  crop_3d(images, label, [self.crop_h, self.crop_w, self.crop_d],mode='center')
            if np.random.rand(1) <= 0.2:
                images = brightness_multiply(images, multiply_range=[0.7, 1.3])
            if np.random.rand(1) <= 0.2:
                images = brightness_additive(images, std=0.1)
            if np.random.rand(1) <= 0.2:
                images = gamma(images, gamma_range=[0.7, 1.5])
            if np.random.rand(1) <= 0.2:
                images = contrast(images, contrast_range=[0.7, 1.3])
            if np.random.rand(1) <= 0.2:
                images = gaussian_blur(images, sigma_range=[0.5, 1.0])
            if np.random.rand(1) <= 0.2:
                images = gaussian_noise(images, std= np.random.random() * 0.1)
                  
            images = images.squeeze(0)
            if np.random.rand(1) <= 0.5:
               
                images = rotate_3d_image_and_label(images, angle_spectrum=90)
           
            if np.random.rand(1) <= 0.5:  # mirror_flip W
                images = np.array(images)
                images = images[:, :, :, ::-1]  # h flip
                images = torch.from_numpy(images.copy())  
            if np.random.rand(1) <= 0.5:
                images = np.array(images)
                images = images[:, :, ::-1, :]  # v flip
                images = torch.from_numpy(images.copy())  
            if np.random.rand(1) <= 0.5:
                images = np.array(images)
                images = images[:, ::-1, :, :]  
                images = torch.from_numpy(images.copy())  
           
            images = np.array(images)
            images = resize(images, (images.shape[0], self.crop_h, self.crop_w, self.crop_d), order=1, mode='constant', cval=0,
                        clip=True, preserve_range=True)
            images = torch.from_numpy(images)
            
     

        elif self.split == 'val' or self.split == 'test':
            images = np.array(images)
          
            images = resize(images, (images.shape[0], self.crop_h, self.crop_w, self.crop_d), order=1, mode='constant', cval=0,
                        clip=True, preserve_range=True)
           
            images = torch.from_numpy(images)
                
        if name[:3] in ['Bra']:
            x1, x2, x3, x4 = images[0, ...], images[1, ...], images[2, ...], images[3, ...]  # (t1, t1ce, t2, f)(128,128,96)
            x5, x6,x7,x8=  [torch.zeros_like(images[0, ...]) for _ in range(4)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [0, 1, 2, 3]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),), dtype=sequence_code.dtype)
                else:
                    sequence_code[0:4] = 1
            elif self.split == 'val':
                sequence_code[0:4] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x1, x2, x3, x4 = x1 * sequence_code[0], x2 * sequence_code[1], x3 * sequence_code[2], x4 * sequence_code[3]
        elif name[:3] in ["cen"]:
            x3 = images[0, ...]  # (t2w)(128,128,96)
            x1, x2, x4, x5, x6,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(7)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            sequence_code[2] = 1 
        elif name[:3] in ['NPC']:
            x1, x2, x3 = images[0, ...], images[1, ...], images[2, ...]  # (t1, t1c, t2)(128,128,96)
            x4, x5, x6, x7, x8 = [torch.zeros_like(images[0, ...]) for _ in range(5)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [0, 1, 2]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
                else:
                    sequence_code[[0, 1, 2]] = 1
            elif self.split == 'val':
                sequence_code[[0, 1, 2]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x1, x2, x3 = x1 * sequence_code[0], x2 * sequence_code[1], x3 * sequence_code[2]
        elif name[:3] in ['amb','Bre']:
            x5, x6 = images[0, ...], images[1, ...] # (t1, t1ce)(128,128,96)
            x1, x2,x3,x4,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(6)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [4,5]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
                else:
                    sequence_code[[4,5]] = 1
            elif self.split == 'val':
                sequence_code[[4,5]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x5,x6 = x5 * sequence_code[4], x6 * sequence_code[5]
        elif name[:3] in ['LLD']:
            x7, x8 = images[0, ...], images[1, ...]  # (t1, t1c, t2)(128,128,96)
            x1,x2,x3,x4, x5, x6 = [torch.zeros_like(images[0, ...]) for _ in range(6)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [6,7]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
                else:
                    sequence_code[[6,7]] = 1
            elif self.split == 'val':
                sequence_code[[6,7]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x7,x8 = x7 * sequence_code[6], x8* sequence_code[7]
        
     
        x1,x2,x3,x4,x5,x6,x7,x8= x1[np.newaxis, :],x2[np.newaxis, :],x3[np.newaxis, :],x4[np.newaxis, :],x5[np.newaxis, :],x6[np.newaxis, :],x7[np.newaxis, :],x8[np.newaxis, :]
        
        region_ids = torch.zeros(10, dtype=torch.float32)  #
        
      
        if part_id == 11:  # FedBca
            region_ids[7] = 1.0
        elif part_id == 12:  # NPC
            region_ids[2] = 1.0
        elif part_id == 13:  # LLD
            region_ids[4] = 1.0
        elif part_id == 14:  # BraTS
            region_ids[0] = 1.0
        elif part_id == 15 or part_id == 10:  # BreaDM
            region_ids[3] = 1.0
        else:
            raise ValueError(f"Unknown part_id: {part_id}")
        
        # task encode：cls is 1
        task_ids = torch.tensor(1, dtype=torch.long)  #
        
        return x1.to(torch.float32),x2.to(torch.float32),x3.to(torch.float32),x4.to(torch.float32),\
        x5.to(torch.float32),x6.to(torch.float32),x7.to(torch.float32),x8.to(torch.float32),name,label,sequence_code,region_ids,task_ids
        #

def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms
def tr_cls_collate(batch):
    x1,x2,x3,x4,x5,x6,x7,x8,name,label,sequence_code,region_ids,task_ids = zip(*batch)
    x1 = np.stack(x1, 0)
    x2 = np.stack(x2, 0)
    x3 = np.stack(x3, 0)
    x4 = np.stack(x4, 0)
    x5 = np.stack(x5, 0)
    x6 = np.stack(x6, 0)
    x7 = np.stack(x7, 0)
    x8 = np.stack(x8, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    sequence_code = np.stack(sequence_code, 0)
    region_ids = np.stack(region_ids, 0) 
    task_ids = np.stack(task_ids, 0)
    data_dict = {
        'x1': x1,'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6,'x7': x7, 'x8': x8, 
        'label': label, 'name': name, 'sequence_code': sequence_code,
        'region_ids': region_ids, 'task_ids': task_ids
    }
    return data_dict

def val_cls_collate(batch):
    image, label, name, sequence_id,prompt_id= zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0).astype(np.float64)
    name = np.stack(name, 0)
    sequence_id = np.stack(sequence_id, 0)
    prompt_id = np.stack(prompt_id, 0)
    # task_id = np.stack(task_id, 0)
    data_dict = {'image': image, 'label': label, 'name': name, 'sequence_id': sequence_id,'prompt_id': prompt_id}
    return data_dict