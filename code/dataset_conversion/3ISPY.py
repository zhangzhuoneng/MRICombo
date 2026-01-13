import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

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
    
    

    # cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30]) # z, y, x

    # sitk.WriteImage(cropped_img, '%s/ISPY_%s/ISPY_%s_dce.nii.gz'%(save_path, name,name))
    # sitk.WriteImage(cropped_lab, '%s/ISPY_%s/ISPY_%s_seg.nii.gz'%(save_path, name,name))
    
    sitk.WriteImage(re_img_xyz, '%s/ISPY_%s/ISPY_%s_dce.nii.gz'%(save_path, name,name))
    sitk.WriteImage(re_lab_xyz, '%s/ISPY_%s/ISPY_%s_seg.nii.gz'%(save_path, name,name))


if __name__ == '__main__':


    src_path = '/data/zzn/UniMRINet/dataset/seg_orginal/seg/3ISPY_v2/'
    # ct_tgt_path = '/research/cbim/medical/yg397/universal_model/HNTS_ct/'
  
    tgt_path = '/data/zzn/UniMRINet/dataset/ispy'
    
    if not os.path.exists(tgt_path):
        os.mkdir(tgt_path)
        print(tgt_path)


    print('Start processing training set')

    name_list = []
    for name in os.listdir(f"{src_path}imagesTr/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('.')[0]
        idx = int(idx.split('_DCE')[0])
       
        name_list.append(idx)
        
        target_path = os.path.join(tgt_path, idx)
    
        if not os.path.exists( target_path):
            os.mkdir(target_path)
    # with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
    #     yaml.dump(mr_name_list, f)

    os.chdir(src_path)
    
    for name in name_list:
        img = sitk.ReadImage(src_path+f"imagesTr/{name}_DCE_0000_N3_zscored.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsTr/{name}_DCE_0000_N3_zscored.nii.gz")

        ResampleImage(img, lab, tgt_path, f'{name:03d}', (1.5, 1.5, 1.5))
        print(name, 'done')
        
        
    
        