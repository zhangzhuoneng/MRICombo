import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.), modality='ct'):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    imLabel.CopyInformation(imImage)
    
    # imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    # imLabel = ITKReDirection(imLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
       

    re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    
    if modality == 'mr':
        np_lab = sitk.GetArrayFromImage(re_lab_xyz)
        np_lab[np_lab == 14] = 0
        np_lab[np_lab == 15] = 0

        re_lab_xyz = sitk.GetImageFromArray(np_lab)
        re_lab_xyz.CopyInformation(re_img_xyz)

    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30]) # z, y, x

    sitk.WriteImage(cropped_img, '%s/amos_%04d/amos_%04d.nii.gz'%(save_path, name,name))
    sitk.WriteImage(cropped_lab, '%s/amos_%04d/amos_%04d_seg.nii.gz'%(save_path, name,name))


if __name__ == '__main__':


    src_path = '/data/zzn/UniMRINet/dataset/seg_orginal/6AMOS/'
    # ct_tgt_path = '/research/cbim/medical/yg397/universal_model/amos_ct/'
    mr_tgt_path = '/data/zzn/UniMRINet/dataset/MR_Dataset_1.5_1.5_1.5/6AMOS/'
    os.makedirs(mr_tgt_path,exist_ok=True)


    print('Start processing training set')
    ct_name_list = []
    mr_name_list = []
    for name in os.listdir(f"{src_path}imagesTr/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('.')[0]
        idx = int(idx.split('_')[1])
        if idx > 500:
            mr_name_list.append(idx)

        

    # if not os.path.exists(ct_tgt_path+'list'):
    #     os.mkdir('%slist'%(ct_tgt_path))
    # with open("%slist/dataset.yaml"%ct_tgt_path, "w",encoding="utf-8") as f:
    #     yaml.dump(ct_name_list, f)
    
            target_path = os.path.join(mr_tgt_path, f"amos_{idx:04d}")
        
            if not os.path.exists(target_path):
                os.mkdir(target_path)
    # with open("%slist/dataset.yaml"%mr_tgt_path, "w",encoding="utf-8") as f:
    #     yaml.dump(mr_name_list, f)

    os.chdir(src_path)
    
    for name in mr_name_list:
        img = sitk.ReadImage(src_path+f"imagesTr/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsTr/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, mr_tgt_path, name, (1.5, 1.5, 1.5), modality='mr')
        print(name, 'done')

    
    ct_name_list = []
    mr_name_list = []
    for name in os.listdir(f"{src_path}imagesVa/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('.')[0]
        idx = int(idx.split('_')[1])
        if idx > 500:
            mr_name_list.append(idx)

        

    # if not os.path.exists(ct_tgt_path+'list'):
    #     os.mkdir('%slist'%(ct_tgt_path))
    # with open("%slist/dataset.yaml"%ct_tgt_path, "w",encoding="utf-8") as f:
    #     yaml.dump(ct_name_list, f)
    
            target_path = os.path.join(mr_tgt_path, f"amos_{idx:04d}")
        
            if not os.path.exists( target_path):
                os.mkdir(target_path)
    # with open("%slist/dataset.yaml"%mr_tgt_path, "w",encoding="utf-8") as f:
    #     yaml.dump(mr_name_list, f)

    os.chdir(src_path)
    
    for name in mr_name_list:
        img = sitk.ReadImage(src_path+f"imagesVa/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsVa/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, mr_tgt_path, name, (1.5, 1.5, 1.5), modality='mr')
        print(name, 'done')