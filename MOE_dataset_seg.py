import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import sys
sys.path.append("..")
sys.path.append("../../")
# import cv2
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
from dataset.aug_seg import *
import torch.nn.functional as F
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import matplotlib.pyplot as plt
position_seg_dict= {
    '0BraTS':0, 
    '1HNTS':1,
    '2NPC':2,
    '3ISPY':3,
    '4ATLAS':4, 
    '5Colorectal':5,
    '6AMOS':6,
    '7BCa_seg':7,
    '8ProstateX':8,
    '9csPCa_seg':9, 
}

position_sequence_dict= {
    '0BraTS':4, 
    '1HNTS':1,
    '2NPC':3,
    '3ISPY':1,
    '4ATLAS':1, 
    '5Colorectal':1,
    '6AMOS':1,
    '7BCa_seg':1,
    '8ProstateX':2,
    '9csPCa_seg':3, 
} 

def list_add_prefix(txt_path, prefix_1):
    '''
    prefix_1: 用于训练的数据集的路径前缀
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

            # 获取文件名部分，并通过规则提取唯一名称
            filename = line.split('/')[-1]  # 获取文件名
            unique_names.add(filename)
        
        return list(unique_names)
    
    else:
        return print('数据集错误')
def get_subset_len(seg_dataset, text_path, split):
    subset_len = []
    for dataset_name in seg_dataset:
        subset_len.append(len(list_add_prefix(text_path, dataset_name)))
        # print(dataset_name,len(list_add_prefix(text_path, dataset_name, "imagesTr")))
    return subset_len


class UnisegDataset(data.Dataset):
    def __init__(self, root, text_path, split,code = None, crop_size=(96, 128, 128),scale=True,max_iters=None,
                 mirror=True):
        self.root = root
        self.text_path = text_path
        self.split = split
        self.code =  code
        self.crop_h, self.crop_w, self.crop_d = crop_size
      
        
        self.scale = scale
    
        self.is_mirror = mirror
        self.seg_use_dataset = []
        self.img_ids = [i_id.strip() for i_id in open(self.text_path)]
        # print( self.img_ids)
        # if not max_iters == None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        # print(len(self.img_ids))

        self.files = []
        grouped_files = {}
        # label_files = {}
        print("Start preprocessing....")
        self.seg_use_dataset = [
            '0BraTS', 
            '1HNTS',
            '2NPC',
            '3ISPY',
            # '3NACT',
            '4ATLAS', 
            '5Colorectal',
            '6AMOS',
            '7BCa_seg',
            '8ProstateX',
            '9csPCa_seg',
            '4liver_seg'
            
        ]
        
        
        # self.sequences_Bra =   ["_t1","_t1ce","_t2","_flair","_seg"]
        self.sequences_Bra =   ["-t1n","-t1c","-t2w","-t2f","-seg"]
        self.sequences_HNT  =  ["_t2","_seg"]      
        self.sequences_NPC  =  ["_t1","_t1ce","_t2","_seg"]      
        self.sequences_ISP  =  ["_pos-dce","_pre-dce","_seg"]        
        self.sequences_ATL  =  ["_t1ce","_seg"]
        self.sequences_Col  =  ["_t2","_seg"]      
        self.sequences_AMO  =  ["","_seg"]     
        self.sequences_BCa  =  ["_t2","_seg"]     
        self.sequences_Pro  =  ["_adc","_t2","_seg"]    
        self.sequences_csP  =  ["_adc","_dwi","_t2","_seg"]
        self.sequences_CHAOS  =  ["_t2","_t2_seg"]
        self.sequences_liver_seg  =  ["_HBP","_seg"]
        self.sequences_NACT  =   ["_pos-dce","_pre-dce","_seg"]    
        # paths, names = [], []
        # print(self.img_ids)
        for patient_id in self.img_ids:
            part_id = int(patient_id[11])
            if  part_id==0:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_Bra]
                patient = dict(id=patient_id, t1=paths[0], t1ce=paths[1],t2=paths[2], flair=paths[3], seg=paths[4])
                
            elif part_id==1:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_HNT]
                patient = dict(id=patient_id,t2=paths[0],seg=paths[1])
                
            elif part_id==2:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_NPC]
                patient = dict(id=patient_id,t1=paths[0], t1ce=paths[1],t2=paths[2],seg=paths[3])
                
            elif part_id==3:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_ISP]
                patient = dict(id=patient_id,pos_dce=paths[0],pre_dce=paths[1],seg=paths[2])
                
            # elif part_id==4:
            #     paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_ATL]
            #     patient = dict(id=patient_id, t1ce=paths[0],seg=paths[1])
                
            elif part_id==5:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_Col]
                patient = dict(id=patient_id,t2=paths[0],seg=paths[1])
                
            elif part_id==6:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_AMO]
                patient = dict(id=patient_id,t2=paths[0],seg=paths[1])
                
            elif part_id==7:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_BCa]
                patient = dict(id=patient_id,t2=paths[0],seg=paths[1])
                
            elif part_id==8:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_Pro]
                patient = dict(id=patient_id,adc=paths[0], t2=paths[1],seg=paths[2])
                
            elif part_id==9:
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_csP]
                # print( paths)
                patient = dict(id=patient_id,adc=paths[0], dwi=paths[1],t2=paths[2],seg=paths[3])
            elif part_id==4:
                
                paths = [f"{patient_id.split('/')[-1]}{sequence}.nii.gz" for sequence in self.sequences_liver_seg]
                # print(patient_id,paths)
                patient = dict(id=patient_id,t2=paths[0],seg=paths[1])
            else:
                raise ValueError(f"Unknown part_id: {part_id}")
            self.files.append(patient)
           
        self.subset_len = get_subset_len(self.seg_use_dataset,self.text_path, split)
        print('{} seg {} images are loaded!'.format(len(self.files),self.split))
       

    def __len__(self):
        return len(self.files)



    # def truncate(self, MR):
    #     # 计算分位数
    #     MR = MR.to(torch.float32)
    #     lower = torch.quantile(MR, 0.002)  # 等价于 0.2% 分位数
    #     upper = torch.quantile(MR, 0.998)  # 等价于 99.8% 分位数

    #     # 截断操作
    #     MR = torch.clamp(MR, min=lower, max=upper)

    #     # 计算均值和标准差
    #     mean_value = MR.mean()
    #     std_dev = MR.std()

    #     # 归一化
    #     MR = (MR - mean_value) / (std_dev + 1e-8)
    #     return MR

    def truncate(self, MR):
        mean_value = MR.mean()
        std_dev = MR.std()
        # normalize
        MR = (MR - mean_value) / (std_dev+1e-8)
        return MR

    def id2trainId(self, label, part_id):
        # 初始化结果映射，所有值设为0（表示不存在该类别）
        shape = label.shape
        results_map = torch.zeros((27, shape[0], shape[1], shape[2]), dtype=torch.float32, device=label.device)
        
        # 根据 part_id 设置不同类别的映射
        if part_id == 0:
            # results_map[0] = (label == 4).float()
            results_map[0] = (label == 3).float()
            results_map[1] = ((label >= 1) & (label != 2)).float()
            results_map[2] = (label >= 1).float()
        elif part_id == 1:
            results_map[3] = (label == 1).float()
            results_map[4] = (label == 2).float()
        elif part_id == 2:
            results_map[5] = (label == 1).float()
        elif part_id == 3:
            results_map[6] = (label == 1).float()
        # elif part_id == 4:
        #     results_map[7] = (label >= 1).float()
        #     results_map[8] = (label == 2).float()
        elif part_id == 5:
            results_map[9] = (label == 1).float()
        elif part_id == 6:
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
            results_map[10] = (label == 1).float()
            results_map[11] = (label == 2).float()
            results_map[12] = (label == 3).float()
            results_map[13] = (label == 4).float()
            results_map[14] = (label == 5).float()
            results_map[15] = (label == 6).float()
            results_map[16] = (label == 7).float()
            results_map[17] = (label == 8).float()
            results_map[18] = (label == 9).float()
            results_map[19] = (label == 10).float()
            results_map[20] = (label == 11).float()
            results_map[21] = (label == 12).float()
            results_map[22] = (label == 13).float()
        elif part_id == 7:
            results_map[23] = (label == 1).float()
        elif part_id == 8:
            results_map[24] = (label >= 1).float()
            results_map[25] = (label == 2).float()
        elif part_id == 9:
            results_map[26] = (label == 1).float()
        elif part_id == 4:
            results_map[7] = (label >= 1).float()
            results_map[8] = (label == 2).float()
            # results_map[11] = (label == 2).float()
            # results_map[12] = (label == 3).float()
            # results_map[10] = (label == 4).float()
            
        else:
            print("Error, No such part!")
        
        return results_map
    
    def pad_image(self, img, target_size,mode):
        """
        Pad a PyTorch tensor up to the target size.
        
        Args:
            img (torch.Tensor): Input tensor of shape [D, H, W].
            target_size (tuple): Target size (D, H, W).
        
        Returns:
            torch.Tensor: Padded tensor of shape [target_D, target_H, target_W].
        """
        # 计算需要填充的大小
        if mode =='image':
            rows_missing = math.ceil(target_size[0] - img.shape[1])
            cols_missing = math.ceil(target_size[1] - img.shape[2])
            dept_missing = math.ceil(target_size[2] - img.shape[3])
        else:
            rows_missing = math.ceil(target_size[0] - img.shape[0])
            cols_missing = math.ceil(target_size[1] - img.shape[1])
            dept_missing = math.ceil(target_size[2] - img.shape[2])
        # 如果缺失值小于 0，则设置为 0
        rows_missing = max(rows_missing, 0)
        cols_missing = max(cols_missing, 0)
        dept_missing = max(dept_missing, 0)

        # 使用 torch.nn.functional.pad 进行填充
        padded_img = torch.nn.functional.pad(
            img, 
            (0, dept_missing, 0, cols_missing, 0, rows_missing),  # 填充顺序为 (深度后, 深度前, 宽度后, 宽度前, 高度后, 高度前)
            mode='constant',
            value=0  # 填充值为 0
        )
        return padded_img
    def __getitem__(self, index):
        patient = self.files[index]

        patient_id = patient["id"]
        part_id =  int(patient_id[11])
        name =  patient_id.split('/')[-1]
       
        patient_image = {key:torch.tensor(load_nii(f"{self.root}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "seg"]}
        images = torch.stack([patient_image[key].to(torch.float32) for key in patient_image])
        # print( name,patient_image.shape)
        # labelNII = load_nii(f"{self.root}/{patient_id}/{patient['seg']}")
        
        labelNII = nib.load(f"{self.root}/{patient_id}/{patient['seg']}")
        
        label =  torch.tensor(load_nii(f"{self.root}/{patient_id}/{patient['seg']}").astype("int8"))
       
        # print(1,name,label.shape)
        # if self.split == 'train':
        if name[:3] in ['Bra']:
            images,label = Crop_brain_Foreground(images,label,name)
        if name[:3] in ['Liv', "amo","liv"]:
            images,label = Crop_Foreground(images,label,context_size=[10,30,30])
        
        # print(name,label.shape)
            # print(1,name,label_.shape,label.shape)
            # print(2,name,patient_label.shape)
        # if self.split == 'train':
        #     # if name[:3] not in ['Bra']:
        #     patient_image ,patient_label = Crop_Foreground(patient_image, patient_label,context_size=[0,0,0])
            # else:
            #     patient_image ,patient_label = Crop_Foreground(patient_image, patient_label,context_size=[0,0,0])
            # print(3,name,patient_label.shape)
        # print(3,name,patient_label.shape,patient_label.shape)
     
        # for i in range(patient_image.shape[0]):
        #      patient_image[i] = self.truncate(patient_image[i])#归一化
        
        if self.split == 'train':
            # patient_image,patient_label = crop_3d( patient_image ,patient_label,[self.crop_h+30, self.crop_w+30, self.crop_d+30],mode='random')
            # print(name,patient_image.shape,patient_label.shape)
            if np.random.uniform() < 0.2:
                scaler = np.random.uniform(0.7, 1.4)
            else:
                scaler = 1
            images = self.pad_image(images, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler],'image')
            label  = self.pad_image(label, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler],'label')
            # print(name,patient_image.shape,patient_label.shape)
            [h0, h1, w0, w1, d0, d1] = locate_bbx(label,scaler,self.crop_h, self.crop_w, self.crop_d)
            
            images = images[:,h0: h1, w0: w1, d0: d1]
            label  = label[h0: h1, w0: w1, d0: d1]
            for i in range(images.shape[0]):
                images[i] = self.truncate(images[i])#归一化
            # print(name,images.shape,label.shape)
            # if np.random.rand(1) <= 0.2:
            #     patient_label = patient_label.unsqueeze(dim=0)  
            #     images, label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label, target_size=(self.crop_h, self.crop_w, self.crop_d),mode="random")
            #     label = label.squeeze(dim=0)
            #     # images, label, pad_list, crop_list =  crop_3d (patient_image, patient_label, target_size=(self.crop_h, self.crop_w, self.crop_d),mode="random")
            # else:
            #     patient_label = patient_label.unsqueeze(dim=0)  
            #     images, label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label,target_size=(self.crop_h, self.crop_w, self.crop_d),mode="center")
            #     label = label.squeeze(dim=0)
                
            # for i in range(patient_image.shape[0]):
            #     patient_image[i] = self.truncate(patient_image[i])#归一化
             
            if np.random.rand(1) <= 0.5:
                #  1. 生成统一的旋转参数
                images, label = rotate_3d_image_and_label(images, label, angle_spectrum=45)
                
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
            
            images,label = np.array(images),np.array(label)
            if scaler != 1:
                images = resize(images, (images.shape[0], self.crop_h, self.crop_w, self.crop_d), order=1, mode='constant', cval=0,
                            clip=True, preserve_range=True)
                label = resize(label, (self.crop_h, self.crop_w, self.crop_d), order=0, mode='edge', cval=0, clip=True,
                            preserve_range=True)
            images,label = torch.from_numpy(images),torch.from_numpy(label)
            # print(2,images.shape,label.shape)
            # images, label = np.array(images),np.array(label)
           
            # if np.random.rand(1) <= 0.5:
            #     images, label = random_scale_rotate_translate_3d(images, label, rotate=15)
            
            # plt.figure(figsize=(64, 64))
            # plt.imshow(label[2][:,:,64], cmap='gray')
            # plt.title(f'Slice {10}')
            # plt.axis('off')
            # plt.savefig('./32.jpg')
            # print(name[:3], patient_image.shape,patient_label.shape)
            
        elif self.split == 'val' or self.split == 'test':
            # h, w, d= patient_image.shape[1:]
            # pad_d = (self.crop_d-d) if self.crop_d-d > 0 else 0
            # pad_h = (self.crop_h-h) if self.crop_h-h > 0 else 0
            # pad_w = (self.crop_w-w) if self.crop_w-w > 0 else 0
            # images, label, pad_list = pad_image_and_label(patient_image, patient_label, target_size=(d+pad_d, h+pad_h, w+pad_w))
            # for i in range(patient_image.shape[0]):
            #     patient_image[i] = self.truncate(patient_image[i])#归一化
            # images, label = np.array(patient_image),np.array(patient_label)
            images = self.pad_image(images, [self.crop_h, self.crop_w, self.crop_d],'image')
            label  = self.pad_image(label, [self.crop_h, self.crop_w, self.crop_d],'label')
            for i in range(images.shape[0]):
                images[i] = self.truncate(images[i])#归一化
            images, label = images,label
        else:
            print('error')

        if name[:3] in ['Bra']:
            x1, x2, x3, x4 = images[0, ...], images[1, ...], images[2, ...], images[3, ...]  # (t1, t1ce, t2, flair)(128,128,96)
            x5, x6,x7,x8 =  [torch.zeros_like(images[0, ...]) for _ in range(4)]
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

        elif name[:3] in ['HNT', 'Col', "amo", "cen",]:
            # print(name)
            x3 = images[0, ...]  # (t2w)(128,128,96)
            x1, x2, x4, x5, x6,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(7)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            sequence_code[2] = 1

        elif name[:3] in ['NPC']:
            x1, x2, x3 = images[0, ...], images[1, ...], images[2, ...]  # (t1, t1c, t2)(128,128,96)
            x4, x5, x6,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(5)]
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

        elif name[:3] in ['ISP','NAC']:
            x5,x6 = images[0, ...],images[1, ...]  # (dce)(128,128,96)
            x1, x2,x3, x4,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(6)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [4, 5]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),),dtype=sequence_code.dtype)
                else:
                    sequence_code[[4,5]] = 1
            elif self.split == 'val':
                sequence_code[[4,5]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x5, x6 = x5*sequence_code[4], x6*sequence_code[5]

        elif name[:3] in ['Liv',"liv"]:
            x2 = images[0, ...]  # (t1ce)(128,128,96)
            x1, x3, x4, x5, x6,x7,x8 = [torch.zeros_like(images[0, ...]) for _ in range(7)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            sequence_code[1] = 1

        elif name[:3] in ['Pro']:
            x7, x3 = images[0, ...], images[1, ...]  # (adc, t2)(128,128,96)
            x1, x2, x4,x5, x6,x8 = [torch.zeros_like(images[0, ...]) for _ in range(6)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [6, 2]
                while sequence_code[random_indices].sum() == 0:
                    sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),), dtype=sequence_code.dtype)
            elif self.split == 'val':
                sequence_code[[6, 2]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x7, x3 = x7 * sequence_code[6], x3 * sequence_code[2]

        elif name[:3] in ['csP']:
            x7, x8, x3 = images[0, ...], images[1, ...], images[2, ...]  # (adc, dwi, t2w)(128,128,96)
            x1, x2, x4,x5,x6 = [torch.zeros_like(images[0, ...]) for _ in range(5)]
            sequence_code = torch.zeros(8, dtype=torch.int32)
            if self.split == 'train':
                random_indices = [6,7,2]
                if torch.rand(1) <= 0.5:
                    while sequence_code[random_indices].sum() == 0:
                        sequence_code[random_indices] = torch.randint(0, 2, (len(random_indices),), dtype=sequence_code.dtype)
                else:
                    sequence_code[[6,7,2]] = 1
            elif self.split == 'val':
                sequence_code[[6,7,2]] = 1
            else:
                sequence_code = torch.tensor(self.code, dtype=torch.int32)
            x7, x8, x3 = x7 * sequence_code[6], x8* sequence_code[7], x3 * sequence_code[2]
            # label = np.maximum(np.maximum(labels[0,:,:,:]*sequence_code[5],labels[1,:,:,:]*sequence_code[6]),labels[2,:,:,:]*sequence_code[3])
            # print(sequence_code)
        # print(part_id[0])
        label  = self.id2trainId(label,part_id)
        x1,x2,x3,x4,x5,x6,x7,x8 = x1[np.newaxis, :],x2[np.newaxis, :],x3[np.newaxis, :],x4[np.newaxis, :],x5[np.newaxis, :],x6[np.newaxis, :],x7[np.newaxis, :],x8[np.newaxis, :]# 1 192 192 64 
        # print('a',name,x1.shape)
        # image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W # 1 64 92 192 
        # x1,x2,x3,x4,x5,x6,x7 = x1.transpose((0, 3, 1, 2)),x2.transpose((0, 3, 1, 2)),x3.transpose((0, 3, 1, 2)),x4.transpose((0, 3, 1, 2)),x5.transpose((0, 3, 1, 2)),x6.transpose((0, 3, 1, 2)),x7.transpose((0, 3, 1, 2))
        # label = label.transpose((0, 3, 1, 2))  # Depth x H x W
        # sequence_code = torch.from_numpy(sequence_code)
        # return x1.astype(np.float32),x2.astype(np.float32),x3.astype(np.float32),x4.astype(np.float32),\
        # x5.astype(np.float32),x6.astype(np.float32),x7.astype(np.float32),name,label.astype(np.float32),sequence_code,labelNII.affine
        # return x1,x2,x3,x4,\
        # x5,x6,x7,name,label,sequence_code,labelNII.affine
        # 在返回之前添加部位编码和任务编码
        # 部位编码：根据part_id确定（0-9），转换为one-hot形式
        region_ids = torch.zeros(10, dtype=torch.float32)  # 10个部位，初始化为全0
        region_ids[part_id] = 1.0  # 将对应位置设为1
        
        # 任务编码：分割任务为0，分类任务为1
        # 由于这是分割数据集，所以任务编码为0
        task_ids = torch.tensor(0, dtype=torch.long)  # 0表示分割任务
        
        return x1.to(torch.float32),x2.to(torch.float32),x3.to(torch.float32),x4.to(torch.float32),\
        x5.to(torch.float32),x6.to(torch.float32),x7.to(torch.float32),x8.to(torch.float32),name,label,sequence_code,labelNII.affine,region_ids,task_ids
        
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
def tr_seg_collate(batch):
    x1,x2,x3,x4,x5,x6,x7,x8,name,label,sequence_code,labelNII,region_ids,task_ids= zip(*batch)
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
    labelNII = np.stack(labelNII, 0)
    region_ids = np.stack(region_ids, 0)  # 现在是 [B, 10] 的one-hot向量
    task_ids = np.stack(task_ids, 0)
    data_dict = {
        'x1': x1,'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6,'x7': x7,'x8': x8, 
        'label': label, 'name': name, 'sequence_code': sequence_code,'labelNII':labelNII,
        'region_ids': region_ids, 'task_ids': task_ids
    }
    return data_dict

def val_seg_collate(batch):
    image, label, name, sequence_id,prompt_id= zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    sequence_id = np.stack(sequence_id, 0)
    prompt_id = np.stack(prompt_id, 0)
    data_dict = {'image': image, 'label': label, 'name': name,  'sequence_id': sequence_id,'prompt_id': prompt_id}
    return data_dict
if __name__=="__main__":
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    
    # # Set your data root and text file
    data_dir = "/data/zzn/UniMRINet/dataset/"
    train_list = "/data/zzn/UniMRINet/dataset/segmentation/seg_val.txt"  # 根据你的实际文件名替换


    crop_size = (128,128,128)  # 根据你的需要设置  
    
    
    # val_seg_dataset = UnisegtestDataset_any(data_dir, train_list,crop_size=crop_size)

    # val_seg_loader = DataLoader(val_seg_dataset,
    #                             batch_size=1,
    #                             num_workers=1,
    #                             drop_last=False,
    #                             shuffle=False)
    
    # for i, data in enumerate(val_seg_loader):
    #     print(f"Batch {i}")
    #     channels_combinations_indices,images_combinations, label, name, part_id, task_id, affine = data
    #     print(name)
    #     for comb_indices,image_combination in zip(channels_combinations_indices,images_combinations):
    #     # 这里可以对每个通道组合的图像进行处理
    #         print(f"Channels Combination: {comb_indices}, image Shape: {image_combination.shape}")
        
        
    max_iters = None  # 或者设置为你想要的训练迭代次数
    batch_size = 1  # 根据你的需要设置
    random_scale = True  # 是否使用随机缩放
    random_mirror = True  # 是否使用随机镜像
   
    position_prompt_dict= {
        '0BraTS':0, 
        '1HNTS':1,
        '2NPC':2,
        '3ISPY':3,
        '4ATLAS':4, 
        '5Colorectal':5,
        '6AMOS':6,
        '7BCa_seg':7,
        '8ProstateX':8,
        '9csPCa_seg':9, 
    } 
    nn_dataset =UnisegDataset(data_dir, train_list, split="train",  crop_size=crop_size,
                                scale=random_scale, mirror=random_mirror)
    
   
    def weight_base_init(nn_dataset):
        # 计算每个器官对应的数量，以生成权重
        position_num_dict = {}
       
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            if position_prompt_dict[dataset_name] not in position_num_dict:
                position_num_dict[position_prompt_dict[dataset_name]] = nn_dataset.subset_len[dataset_index]#数据路径列表
            else:
                position_num_dict[position_prompt_dict[dataset_name]] += nn_dataset.subset_len[dataset_index]
        # 计算权重 1/sqrt(n)
        position_weight_dict = {}
        for position in position_num_dict:
            position_weight_dict[position] = 1 / (position_num_dict[position])

        # 生成权重序列
        all_sample_weight_list = []
        for dataset_index, dataset_name in enumerate(nn_dataset.seg_use_dataset):
            all_sample_weight_list += [position_weight_dict[position_prompt_dict[dataset_name]]] * nn_dataset.subset_len[dataset_index]
            
        return all_sample_weight_list
    
    samples_weight=weight_base_init(nn_dataset)
    # print(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # 创建 DataLoader
    # dataloader = DataLoader(nn_dataset, batch_size=1, sampler= samples_weight)
    
    dataloader  = DataLoader(nn_dataset,
                                batch_size=1,
                                num_workers=1,
                                sampler= samples_weight,
                                drop_last=False,
                                shuffle=False)

    # 遍历 DataLoader 并打印样本
    for batch_idx, batch in enumerate(dataloader):
        x1,x2,x3,x4,x5,x6,x7,x8,name,label,sequence_code,labelNII,region_ids,task_ids= batch
        print(f"Batch {batch_idx}")
        print(f"Region IDs (one-hot): {region_ids}")  # 现在显示one-hot向量
        print(f"Task IDs: {task_ids}")
        print(f"Name: {name}")
        # print(x1.shape)
        

      
