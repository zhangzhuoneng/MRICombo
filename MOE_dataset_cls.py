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
    # "10AMBL":0,
    "11FedBca":0,
    "12NPC":1,
    "13LLD":2,
    "14BraTS":3,
    "15BreaDM":4
    
}

def list_add_prefix(txt_path, prefix_1):
    '''
    prefix_1: 用于训练的dataset的路径前缀
    prefix_2: imgs, masks等
    '''
    with open(txt_path, 'r') as f:
        lines = f.readlines()
      
    if prefix_1 is not None:
        filtered_lines = [line for line in lines if line.split('/')[1].startswith(prefix_1)]
        # print(filtered_lines)
        # 提取唯一名称（假设名称位于文件路径的某部分，例如 "ProstateX_0110"）
        unique_names = set()
        for line in filtered_lines:

            # Get/Obtain文件名部分，并通过规则提取唯一名称
            filename = line.split('/')[-1]  # Get/Obtain文件名
            unique_names.add(filename)
        
        return list(unique_names)
    
    else:
        return print('dataset错误')
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
        # "10AMBL",
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

        # for item in self.img_ids:
           
        #     image_path, cls_label = item
        #     sequence = image_path.split('.nii.gz')[0][-3:]
        #     sequence_id=None
        #     # print( image_path)
        #     part_id = int(image_path[11])
        #     if  sequence=="t1n":
        #         sequence_id= 0
        #     elif sequence=="t1c":
        #         sequence_id= 1
        #     elif sequence=="t2w" or sequence=="T2w":
        #         sequence_id= 2
        #     elif sequence=="t2f":
        #         sequence_id= 3
        #     elif sequence=="adc":
        #         sequence_id= 4
        #     elif sequence=="dce" or sequence=="DCE":
        #         sequence_id= 5
        #     name = osp.splitext(osp.basename(image_path))[0]
        #     img_file = osp.join(self.root, image_path)
        #     cls_label = float(cls_label)  # 先转换为浮点数
        #     label = int(cls_label)  # 再转换为整数并减去1       
        #     # print(img_file,label)
        #     self.files.append({
        #         "image": img_file,
        #         "label": label,
        #         "name": name,
        #         "part_id": part_id,
        #         "sequence_id":sequence_id
        #     })
        # print('{} cls_{} images are loaded!'.format(len(self.img_ids),self.split))
        # self.subset_len = get_subset_len(self.cls_use_dataset,self.text_path,split)
       
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
        part_id = int(patient_id[11:13])  # Get/Obtain部位ID
        name =  patient_id.split('/')[-1]
        # print(patient_id)
        images = {key:torch.tensor(load_nii(f"{self.root}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "label"]}
        # print(images[0].shape)
        # images = torch.stack([images[key].to(torch.float32) for key in images])
        
        # external test lld
        # images = {key: torch.tensor(
        #     resize(load_nii(f"{self.root}/{patient_id}/{patient[key]}"), (128, 128, 128),
        #            order=1, mode='constant', cval=0, clip=True, preserve_range=True)
        # ) for key in patient if key not in ["id", "label"]}
        #
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
                #  1. 生成统一的旋转parameter
                images = rotate_3d_image_and_label(images, angle_spectrum=90)
           
            if np.random.rand(1) <= 0.5:  # mirror_flip W
                images = np.array(images)
                images = images[:, :, :, ::-1]  # 水平翻转
                images = torch.from_numpy(images.copy())  # 添加 .copy()
            if np.random.rand(1) <= 0.5:
                images = np.array(images)
                images = images[:, :, ::-1, :]  # 垂直翻转
                images = torch.from_numpy(images.copy())  # 添加 .copy()
            if np.random.rand(1) <= 0.5:
                images = np.array(images)
                images = images[:, ::-1, :, :]  # 深度翻转
                images = torch.from_numpy(images.copy())  # 添加 .copy()
            # 如果是需要采样到非96 96 96， 需要修改这里
            images = np.array(images)
            images = resize(images, (images.shape[0], self.crop_h, self.crop_w, self.crop_d), order=1, mode='constant', cval=0,
                        clip=True, preserve_range=True)
            images = torch.from_numpy(images)
            
     

        elif self.split == 'val' or self.split == 'test':
            images = np.array(images)
            # 如果是需要采样到非96 96 96， 需要修改这里
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
        
        # image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
        #                 clip=True, preserve_range=True)
        x1,x2,x3,x4,x5,x6,x7,x8= x1[np.newaxis, :],x2[np.newaxis, :],x3[np.newaxis, :],x4[np.newaxis, :],x5[np.newaxis, :],x6[np.newaxis, :],x7[np.newaxis, :],x8[np.newaxis, :]
        # images = images.astype(np.float32)
        # return image.copy(), label, name,sequence_id,part_id
        # 添加部位编码和任务编码
        # 根据part_id映射到部位编码（0-5），转换为one-hot形式
        region_ids = torch.zeros(10, dtype=torch.float32)  # 6个部位，Initialize为全0
        
        # if part_id == 10:  # AMBL
        #     region_ids[0] = 1.0
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
        
        # 任务编码：classification任务为1
        task_ids = torch.tensor(1, dtype=torch.long)  # 1表示classification任务
        
        return x1.to(torch.float32),x2.to(torch.float32),x3.to(torch.float32),x4.to(torch.float32),\
        x5.to(torch.float32),x6.to(torch.float32),x7.to(torch.float32),x8.to(torch.float32),name,label,sequence_code,region_ids,task_ids
        # if name[:3] in ['Bra']:
        #     x1, x2, x3, x4 = images[0, ...], images[1, ...], images[2, ...], images[3, ...]  # (t1, t1ce, t2, f)(128,128,96)
        #     x5, x6 =  [torch.zeros_like(images[0, ...]) for _ in range(2)]
        #     sequence_code = torch.zeros(6, dtype=torch.int32)
        #     if self.split == 'train':
        #         random_indices = [0, 1, 2, 3]
        #         if torch.rand(1) <= 0.5:
        #             while sequence_code[random_indices].sum() == 0:
        #                 sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),), dtype=sequence_code.dtype)
        #         else:
        #             sequence_code[0:4] = 1
        #     elif self.split == 'val':
        #         sequence_code[0:4] = 1
        #     else:
        #         sequence_code = torch.tensor(self.code, dtype=torch.int32)
        #     x1, x2, x3, x4 = x1 * sequence_code[0], x2 * sequence_code[1], x3 * sequence_code[2], x4 * sequence_code[3]
        # elif name[:3] in ["cen"]:
        #     x3 = images[0, ...]  # (t2w)(128,128,96)
        #     x1, x2, x4, x5, x6 = [torch.zeros_like(images[0, ...]) for _ in range(5)]
        #     sequence_code = torch.zeros(6, dtype=torch.int32)
        #     sequence_code[2] = 1 
        # elif name[:3] in ['NPC']:
        #     x1, x2, x3 = images[0, ...], images[1, ...], images[2, ...]  # (t1, t1c, t2)(128,128,96)
        #     x4, x5, x6 = [torch.zeros_like(images[0, ...]) for _ in range(3)]
        #     sequence_code = torch.zeros(6, dtype=torch.int32)
        #     if self.split == 'train':
        #         random_indices = [0, 1, 2]
        #         if torch.rand(1) <= 0.5:
        #             while sequence_code[random_indices].sum() == 0:
        #                 sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
        #         else:
        #             sequence_code[[0, 1, 2]] = 1
        #     elif self.split == 'val':
        #         sequence_code[[0, 1, 2]] = 1
        #     else:
        #         sequence_code = torch.tensor(self.code, dtype=torch.int32)
        #     x1, x2, x3 = x1 * sequence_code[0], x2 * sequence_code[1], x3 * sequence_code[2]
        # elif name[:3] in ['amb','Bre']:
        #     x1, x2 = images[0, ...], images[1, ...] # (t1, t1ce)(128,128,96)
        #     x3,x4,x5,x6 = [torch.zeros_like(images[0, ...]) for _ in range(4)]
        #     sequence_code = torch.zeros(6, dtype=torch.int32)
        #     if self.split == 'train':
        #         random_indices = [0,1]
        #         if torch.rand(1) <= 0.5:
        #             while sequence_code[random_indices].sum() == 0:
        #                 sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
        #         else:
        #             sequence_code[[0,1]] = 1
        #     elif self.split == 'val':
        #         sequence_code[[0,1]] = 1
        #     else:
        #         sequence_code = torch.tensor(self.code, dtype=torch.int32)
        #     x1,x2 = x1 * sequence_code[0], x2 * sequence_code[1]
        # elif name[:3] in ['LLD']:
        #     x5, x6 = images[0, ...], images[1, ...]  # (t1, t1c, t2)(128,128,96)
        #     x1,x2,x3, x4 = [torch.zeros_like(images[0, ...]) for _ in range(4)]
        #     sequence_code = torch.zeros(6, dtype=torch.int32)
        #     if self.split == 'train':
        #         random_indices = [4,5]
        #         if torch.rand(1) <= 0.5:
        #             while sequence_code[random_indices].sum() == 0:
        #                 sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
        #         else:
        #             sequence_code[[4,5]] = 1
        #     elif self.split == 'val':
        #         sequence_code[[4,5]] = 1
        #     else:
        #         sequence_code = torch.tensor(self.code, dtype=torch.int32)
        #     x5,x6 = x5 * sequence_code[4], x6* sequence_code[5]
        
        # # image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
        # #                 clip=True, preserve_range=True)
        # x1,x2,x3,x4,x5,x6= x1[np.newaxis, :],x2[np.newaxis, :],x3[np.newaxis, :],x4[np.newaxis, :],x5[np.newaxis, :],x6[np.newaxis, :]
        # # images = images.astype(np.float32)
        # # return image.copy(), label, name,sequence_id,part_id
        # return x1.to(torch.float32),x2.to(torch.float32),x3.to(torch.float32),x4.to(torch.float32),\
        # x5.to(torch.float32),x6.to(torch.float32),name,label,sequence_code


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
    region_ids = np.stack(region_ids, 0)  # 现在是 [B, 6] 的one-hot向量
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
if __name__=="__main__":
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    
    # Set your data root and text file
    data_dir = "/data/zzn/UniMRINet/dataset/"
    train_list = "/data/zzn/UniMRINet/dataset/classification/cls_val.txt"  # 根据你的实际文件名替换
    max_iters = None  # 或者Set/Setup为你想要的训练迭代次数
    batch_size = 1  # 根据你的需要Set/Setup
    crop_size = (96,96, 96)  # 根据你的需要Set/Setup
    random_scale = True  # 是否使用随机缩放
    random_mirror = True  # 是否使用随机镜像
    position_prompt_dict= {
    "10AMBL":0,
    "11FedBca":1,
    "12NPC":2,
    "13LLD":3,
    "14BraTS":4
}
    nn_dataset =  UniclsDataset(data_dir, train_list, split="train",  crop_size=crop_size,
                                scale=random_scale, mirror=random_mirror)
    
    def weight_base_init(nn_dataset):
        # Calculate/Compute每个器官对应的数量，以生成权重
        position_num_dict = {}
       
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径list
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
        # Calculate/Compute权重 1/sqrt(n)
        position_weight_dict = {}
        for position in position_num_dict:
            position_weight_dict[position] = 1 / (position_num_dict[position])

        # 生成权重序列
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.cls_use_dataset):
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]
            
        return all_sample_weight_list
    
    samples_weight=weight_base_init(nn_dataset)
    # print(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Create DataLoader
    dataloader = DataLoader(nn_dataset, batch_size=6, sampler=sampler)

    # 遍历 DataLoader 并Printsample
    for batch_idx, batch in enumerate(dataloader):
        x1,x2,x3,x4,x5,x6,x7,x8,name,label,sequence_code,region_ids,task_ids= batch
        print(f"Batch {batch_idx}")
        print(f"Region IDs (one-hot): {region_ids}")  # 现在显示one-hot向量
        print(f"Task IDs: {task_ids}")
        print(f"Name: {name}")
        print(f"Label: {label}")