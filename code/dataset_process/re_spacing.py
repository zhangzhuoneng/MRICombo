import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import SimpleITK as sitk
import concurrent.futures

# 4: [1.5, 0.8, 0.8],
spacing = {
    0: [1.5,1.2,1.2],
    1: [1.5,1.2,1.2],
    2: [1.5,1.2,1.2],
    3: [1.5,1.2,1.2],
    4: [1.5,1.2,1.2],
    5: [1.5,1.2,1.2],
    6: [1.5,1.2,1.2],
    7: [1.5,1.2,1.2],
    8: [1.5,1.2,1.2],
    9: [1.5,1.2,1.2]
}

# ori_path = '/data/zzn/UniMRINet/dataset/seg_orginal'
# new_path = '/data/zzn/UniMRINet/dataset/MR_dataset1'
ori_path = '/data/zzn/UniMRINet/dataset/seg_orginal/seg/ISPY'
new_path = '/data/zzn/UniMRINet/dataset/ispy'
os.makedirs(new_path, exist_ok=True)

def process_image(root3, i_files3, i_dirs1, i_dirs2):
    if i_files3[0] == '.':
        return
    # read img
    print("Processing %s" % (i_files3))
    img_path = os.path.join(root3, i_files3)
    imageITK = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(imageITK)
    ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    task_id = int(i_dirs1[0])
    target_spacing = np.array(spacing[task_id])
    spc_ratio = ori_spacing / target_spacing

    data_type = image.dtype
    if i_dirs2 == 'labelsTr':  
        image[image == 14] = 0
        image[image == 15] = 0
        order = 0
        mode_ = 'edge'
        data_type = np.int32
    else:  
        order = 3
        mode_ = 'constant'

    image = image.astype(float)

    image_resize = resize(image, (int(np.round(ori_size[0] * spc_ratio[0])),
                                 int(np.round(ori_size[1] * spc_ratio[1])),
                                 int(np.round(ori_size[2] * spc_ratio[2]))),
                          order=order, mode=mode_, cval=0, clip=True, preserve_range=True)

    image_resize = np.round(image_resize).astype(data_type)

    # save
    file_prefix = i_files3.split('_T')[0]  
    
    if "NPC" in i_files3:
       file_prefix = i_files3.split('-T')[0]
            
    elif "HNT" in i_files3:
       file_prefix = i_files3.split('_T')[0]
       
    elif "Bra" in i_files3:
       file_prefix = i_files3.split('_t')[0]
            
    elif "ISP" in i_files3:
       file_prefix = i_files3.split('_DCE')[0]
    elif "Liv" in i_files3:
       file_prefix = "_".join(i_files3.split('-')[:-1])
        
    elif "Col" in i_files3:
       file_prefix = "_".join(i_files3.split('-')[:-1])
    elif "amo" in i_files3:
       file_prefix = i_files3.split('.nii')[0]
    elif "cen" in i_files3:
       file_prefix = i_files3.split('.nii')[0]
    elif "Pro" in i_files3:
       file_prefix = "_".join(i_files3.split('-')[:-1])
    elif "csP" in i_files3:
       file_prefix = "_".join(i_files3.split('_')[:-1])
    else:
        print("error")
    save_path = os.path.join(new_path, i_dirs1, file_prefix)
    os.makedirs(save_path, exist_ok=True)
    
    saveITK = sitk.GetImageFromArray(image_resize)
    saveITK.SetSpacing(target_spacing[[2, 1, 0]])
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    
    if i_dirs2 == 'labelsTr':
        if "NPC" in i_files3:
            if "T1c" in i_files3:
                save_filename = i_files3.split('-T')[0] +'_t1ce_seg.nii.gz'
            elif "T2w" in i_files3:
                save_filename = i_files3.split('-T')[0] + '_t2_seg.nii.gz'
            elif "T1n" in i_files3:
                save_filename = i_files3.split('-T')[0] + '_t1_seg.nii.gz'
            else:
                print("error")
        elif "Bra" in i_files3:
            if "t1c" in i_files3:
                save_filename = i_files3.split('_t')[0] +'_t1ce_seg.nii.gz'
            elif "t2w" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_t2_seg.nii.gz'
            elif "t1n" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_t1_seg.nii.gz'
            elif "t2f" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_flair_seg.nii.gz'
            else:
                print("no brain error")
            
        elif "HNT" in i_files3:
            save_filename = i_files3.split('_T')[0] + '_seg.nii.gz'
            
        # elif "ISP" in i_files3:
        #     save_filename = i_files3.split('_DCE')[0] + '_seg.nii.gz'
        elif "ISP" in i_files3:
            save_filename = i_files3.split('_DCE')[0] + '_seg.nii.gz'
        elif "Liv" in i_files3:
            save_filename = "_".join(i_files3.split('-')[:-1]) + '_seg.nii.gz'
            
        elif "Col" in i_files3:
            save_filename =  "_".join(i_files3.split('-')[:-1]) + '_seg.nii.gz'
        elif "amo" in i_files3:
            save_filename = i_files3.split('.nii')[0] + '_seg.nii.gz'
        elif "cen" in i_files3:
            save_filename = i_files3.split('.nii')[0] + '_seg.nii.gz'
        elif "Pro" in i_files3:
            
            if "t2w" in i_files3:
                save_filename = "_".join(i_files3.split('-')[:-1]) + '_t2_seg.nii.gz'
            elif "adc" in i_files3:
                save_filename = "_".join(i_files3.split('-')[:-1]) + '_adc_seg.nii.gz'
            else:
                print("error")
                
        elif "csP" in i_files3:
            if "t2w" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_t2_seg.nii.gz'
            elif "adc" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_adc_seg.nii.gz'
                
            elif "hbv" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_dwi_seg.nii.gz'
        else:
            print("error")
            
    else:
        
        if "NPC" in i_files3:
            if "T1c" in i_files3:
                save_filename = i_files3.split('-T')[0] + '_t1ce.nii.gz'
            elif "T2w" in i_files3:
                save_filename = i_files3.split('-T')[0] + '_t2.nii.gz'
            elif "T1n" in i_files3:
                save_filename = i_files3.split('-T')[0] + '_t1.nii.gz'
            else:
                print("error")
        elif "Bra" in i_files3:
            if "t1c" in i_files3:
                save_filename = i_files3.split('_t')[0] +'_t1ce.nii.gz'
            elif "t2w" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_t2.nii.gz'
            elif "t1n" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_t1.nii.gz'
            elif "t2f" in i_files3:
                save_filename = i_files3.split('_t')[0] + '_flair.nii.gz'
            else:
                print("error")
                
        elif "HNT" in i_files3:
            save_filename = i_files3.split('_T')[0] + '_t2.nii.gz'
            
        # elif "ISP" in i_files3:
        #     save_filename = i_files3.split('_DCE')[0] + '_dce.nii.gz'
        elif "ISP" in i_files3:
            if "0000" in i_files3:
                save_filename = i_files3.split('_DCE')[0] + '_pre-dce.nii.gz'
            if "0001" in i_files3:
                save_filename = i_files3.split('_DCE')[0] + '_pos-dce.nii.gz'
            # else:
            #     print("error")
        elif "Liv" in i_files3:
            save_filename = "_".join(i_files3.split('-')[:-1]) + '_t1ce.nii.gz'
            
        elif "Col" in i_files3:
            save_filename =  "_".join(i_files3.split('-')[:-1]) + '_t2.nii.gz'
        elif "amo" in i_files3:
            save_filename = i_files3.split('.nii')[0] + '.nii.gz'
        elif "cen" in i_files3:
            save_filename = i_files3.split('.nii')[0] + '_t2.nii.gz'
            
        elif "Pro" in i_files3: 
            if "t2w" in i_files3:
                save_filename = "_".join(i_files3.split('-')[:-1]) + '_t2.nii.gz'
            elif "adc" in i_files3:
                save_filename = "_".join(i_files3.split('-')[:-1]) + '_adc.nii.gz'
            else:
                print("error")
        elif "csP" in i_files3:
            if "t2w" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_t2.nii.gz'
            elif "adc" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_adc.nii.gz'
                
            elif "hbv" in i_files3:
                save_filename = "_".join(i_files3.split('_')[:-1]) + '_dwi.nii.gz'
            
    sitk.WriteImage(saveITK, os.path.join(save_path, save_filename))

count = -1
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for root1, dirs1, _ in os.walk(ori_path):
        print(root1)
        for i_dirs1 in tqdm(sorted(dirs1)):
            for root2, dirs2, files2 in os.walk(os.path.join(root1, i_dirs1)):
                for i_dirs2 in sorted(dirs2):  # imagesTr
                    for root3, dirs3, files3 in os.walk(os.path.join(root2, i_dirs2)):
                        for i_files3 in sorted(files3):
                            futures.append(executor.submit(process_image, root3, i_files3, i_dirs1, i_dirs2))
    for future in concurrent.futures.as_completed(futures):
        future.result()


def merge_labels(label_paths, output_path):
    merged_label = None
    for label_path in label_paths:
        label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label)
        if merged_label is None:
            merged_label = label_array
        else:
            merged_label = np.maximum(merged_label, label_array)
    
    merged_label_itk = sitk.GetImageFromArray(merged_label)
    merged_label_itk.SetSpacing(label.GetSpacing())
    merged_label_itk.SetOrigin(label.GetOrigin())
    merged_label_itk.SetDirection(label.GetDirection())
    sitk.WriteImage(merged_label_itk, output_path)
    
    # 删除原始路径下的各个序列标签
    for label_path in label_paths:
        os.remove(label_path)

def get_all_files_with_extension(directory, extension):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths
# Merge Pro and NPC labels
pro_label_paths = get_all_files_with_extension(os.path.join(new_path, '8ProstateX'), '_seg.nii.gz')
npc_label_paths = get_all_files_with_extension(os.path.join(new_path, '2NPC'), '_seg.nii.gz')
csp_label_paths = get_all_files_with_extension(os.path.join(new_path, '9csPCa_seg'), '_seg.nii.gz')
bra_label_paths = get_all_files_with_extension(os.path.join(new_path, '0BraTS'), '_seg.nii.gz')

# # Group files by patient
pro_patient_groups = {}
npc_patient_groups = {}
csp_patient_groups = {}
bra_patient_groups = {}

for path in pro_label_paths:
    # print(os.path.basename(path))
    patient_id = '_'.join(os.path.basename(path).split('_')[:-2])
    if patient_id not in pro_patient_groups:
        pro_patient_groups[patient_id] = []
    pro_patient_groups[patient_id].append(path)

for path in npc_label_paths:
    patient_id ='_'.join(os.path.basename(path).split('_')[:-2])
    if patient_id not in npc_patient_groups:
        npc_patient_groups[patient_id] = []
    npc_patient_groups[patient_id].append(path)
    
for path in csp_label_paths:
    patient_id ='_'.join(os.path.basename(path).split('_')[:-2])
    # print(patient_id)
    if patient_id not in csp_patient_groups:
        csp_patient_groups[patient_id] = []
    csp_patient_groups[patient_id].append(path)

for path in bra_label_paths:
    patient_id ='_'.join(os.path.basename(path).split('_')[:-2])
    # print(patient_id)
    if patient_id not in bra_patient_groups:
        bra_patient_groups[patient_id] = []
    bra_patient_groups[patient_id].append(path)

# print(pro_patient_groups)
# Merge labels for each patient
for patient_id, paths in pro_patient_groups.items():
    print(paths)
    output_path = os.path.join(new_path, '8ProstateX',patient_id, f'{patient_id}_seg.nii.gz')
    merge_labels(paths, output_path)

for patient_id, paths in npc_patient_groups.items():
    output_path = os.path.join(new_path, '2NPC',patient_id, f'{patient_id}_seg.nii.gz')
    merge_labels(paths, output_path)
    
for patient_id, paths in csp_patient_groups.items():
    output_path = os.path.join(new_path, '9csPCa_seg',patient_id, f'{patient_id}_seg.nii.gz')
    merge_labels(paths, output_path)

for patient_id, paths in bra_patient_groups.items():
    output_path = os.path.join(new_path, '0BraTS',patient_id, f'{patient_id}_seg.nii.gz')
    merge_labels(paths, output_path)