import numpy as np
import SimpleITK as sitk
from utils import CropForeground, ITKReDirection
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb
from matplotlib import image

def ResampleToFixedSize(image, size=[128, 128, 128], interp=sitk.sitkLinear):
    """将图像重采样到固定大小"""
    reference_image = sitk.Image(size, image.GetPixelID())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    
    # 计算新的spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [original_spacing[0] * (original_size[0] / size[0]),
                  original_spacing[1] * (original_size[1] / size[1]),
                  original_spacing[2] * (original_size[2] / size[2])]
    reference_image.SetSpacing(new_spacing)
    
    # 执行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interp)
    resampled_image = resampler.Execute(image)
    
    return resampled_image

def ProcessImageWithFixedSize(imImage, imLabel, save_path, name, target_spacing=[1.0, 1.0, 1.0],target_size=[128, 128, 128], mode='0000'):
    assert imImage.GetSize() == imLabel.GetSize()
    imLabel.CopyInformation(imImage)
    
    # 创建保存路径
    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
        
    # spacing = imImage.GetSpacing()
    # origin = imImage.GetOrigin()
    
    # npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    # nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
        
    # re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkBSpline)
    # re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    
    # re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    # re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    # 先进行前景裁剪
    cropped_img, cropped_lab = CropForeground(imImage, imLabel, context_size=[10, 10, 10]) # z, y, x
    
    # 重采样到固定大小
    resized_img = ResampleToFixedSize(cropped_img, size=target_size, interp=sitk.sitkBSpline)
    resized_lab = ResampleToFixedSize(cropped_lab, size=target_size, interp=sitk.sitkNearestNeighbor)
    
    if mode=="t1c":
        # 保存处理后的图像
        sitk.WriteImage(resized_img, '%s/%s/%s_t1ce.nii.gz'%(save_path, name, name))
        sitk.WriteImage(resized_lab, '%s/%s/%s_seg.nii.gz'%(save_path, name, name))
    elif mode=="t1":
        # 保存处理后的图像
        sitk.WriteImage(resized_img, '%s/%s/%s_t1.nii.gz'%(save_path, name, name))
        # sitk.WriteImage(resized_lab, '%s/%s/%s_seg.nii.gz'%(save_path, name, name))
    elif mode=="t2":
        # 保存处理后的图像
        sitk.WriteImage(resized_img, '%s/%s/%s_t2.nii.gz'%(save_path, name, name))
        # sitk.WriteImage(resized_lab, '%s/%s/%s_seg.nii.gz'%(save_path, name, name))

if __name__ == '__main__':
    src_path = '/data/zzn/UniMRINet/dataset/seg_orginal/cls/12NPC/'
    mr_tgt_path = '/data/zzn/UniMRINet/dataset/MR_Dataset_1.5_1.5_1.5/12NPC/'
    os.makedirs(mr_tgt_path, exist_ok=True)
    
    print('Start processing training set')
    mr_name_list = []
    for name in os.listdir(f"{src_path}imagesTr/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('-')[0]
        mr_name_list.append(idx)
        
        target_path = os.path.join(mr_tgt_path, idx)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    
    os.chdir(src_path)
    
    # for name in mr_name_list:
    #     img = sitk.ReadImage(src_path+f"imagesTr/{name}-T1c.nii.gz")
    #     lab = sitk.ReadImage(src_path+f"labelsTr/{name}-T1c.nii.gz")
    #     ProcessImageWithFixedSize(img, lab, mr_tgt_path, name, target_size=[128, 128, 128], mode='t1c')
    #     print(name, 'done')
        
    # for name in mr_name_list:
    #     img = sitk.ReadImage(src_path+f"imagesTr/{name}-T1n.nii.gz")
    #     lab = sitk.ReadImage(src_path+f"labelsTr/{name}-T1n.nii.gz")
    #     ProcessImageWithFixedSize(img, lab, mr_tgt_path, name, target_size=[128, 128, 128], mode='t1')
    #     print(name, 'done')
        
    # for name in mr_name_list:
    #     img = sitk.ReadImage(src_path+f"imagesTr/{name}-T2w.nii.gz")
    #     lab = sitk.ReadImage(src_path+f"labelsTr/{name}-T2w.nii.gz")
    #     ProcessImageWithFixedSize(img, lab, mr_tgt_path, name, target_size=[128, 128, 128], mode='t2')
    #     print(name, 'done')
    for name in mr_name_list:
        for mode, suffix in [('t1c', 'T1c'), ('t1', 'T1n'), ('t2', 'T2w')]:
            img_path = src_path + f"imagesTr/{name}-{suffix}.nii.gz"
            lab_path = src_path + f"labelsTr/{name}-{suffix}.nii.gz"
            
            if not os.path.exists(img_path) or not os.path.exists(lab_path):
                print(f"Skipping {name}-{suffix}, file not found.")
                continue
            
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(lab_path)
            ProcessImageWithFixedSize(img, lab, mr_tgt_path, name, target_size=[128, 128, 128], mode=mode)
        print(f"{name}-{suffix} done")
        