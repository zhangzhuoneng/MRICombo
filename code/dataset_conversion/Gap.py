
import os
import nibabel as nib
import numpy as np
import glob
import SimpleITK as sitk

def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()
    same_ori = np.all(np.isclose(ori1, ori2, atol=0.005))  # np.isclose 判断两个数组是否相近
    if not same_ori:
        print("the origin does not match between the images:")
        print(ori1)
        print(ori2)
    same_spac = np.all(np.isclose(spacing1, spacing2, atol=0.005))
    if not same_spac:
        print("the spacing does not match between the images")
        print(spacing1)
        print(spacing2)
    same_dir = np.all(np.isclose(direction1, direction2, atol=0.005))
    if not same_dir:
        print("the direction does not match between the images")
        print(direction1)
        print(direction2)
    same_size = np.all(np.isclose(size1, size2, atol=0.005))
    if not same_size:
        print("the size does not match between the images")
        print(size1)
        print(size2)
    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False

def resample_label_to_image(label, image):
    resampler = sitk.ResampleImageFilter()
    # 设置参考图像，确保重采样后的图像与参考图像具有相同的几何属性
    resampler.SetReferenceImage(image)
    # 设置默认像素值，使用标签图像自身的像素 ID 值
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    # 使用最近邻插值，确保标签值的离散性
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # 设置输出图像的数据类型与标签图像一致
    resampler.SetOutputPixelType(image.GetPixelID())

    try:
        # 执行重采样操作
        resampled_image = resampler.Execute(label)
        return resampled_image
    except Exception as e:
        print(f"Error during resampling: {e}")
        return None

# 设置包含 .nii.gz 文件的目录模式
directory_pattern = r'E:\MRI_dataset\original\seg_orginal\9csPCa_seg\imagesTr'
label_pattern = r'E:\MRI_dataset\original\seg_orginal\9csPCa_seg\labelsTr'

target_path = r'E:\MRI_dataset_1\data_Geometry properties\9csPCa_seg\labelsTr'
if not os.path.exists(target_path):
    os.makedirs(target_path)

# 使用 glob 找到所有匹配的目录
directories = glob.glob(directory_pattern)
label_directories = glob.glob(label_pattern)

# 初始化 spacing 列表，用于存储每个方向的 spacing 值
spacings = []
y_spacing_list = []
z_spacing_list = []

sizes = []
y_size_list = []
z_size_list = []

# 打开文件，准备将输出保存到 txt 文件
output_file = '../spacing_seg_info.txt'
with open(output_file, 'w') as f:
    # 遍历所有找到的目录
    for directory in directories:
        # 初始化参考图像
        reference_image = None
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            if filename.endswith('.nii.gz'):
                # 构造完整的文件路径
                file_path = os.path.join(directory, filename)
                # 读取 nii.gz 文件
                nii_image = nib.load(file_path)
                # 获取 spacing 信息
                spacing = nii_image.header.get_zooms()[:3]  # 通常为 x, y, z 的 spacing
                # 格式化每个文件的 spacing 信息保留小数点后 4 位
                data = nii_image.get_fdata()

                spacing_info = f"{filename}: x={spacing[0]:.4f}, y={spacing[1]:.4f}, z={spacing[2]:.4f}\n"

                f.write(spacing_info)

                spacings.append([spacing[0], spacing[1], spacing[2]])

                sizes.append([data.shape[0], data.shape[1], data.shape[2]])

                # 读取 SimpleITK 图像
                sitk_image = sitk.ReadImage(file_path)

                # 如果没有设置参考图像，则将第一个图像作为参考图像
                if reference_image is None:
                    reference_image = sitk_image

                # 查找对应的标签文件
                label_found = False
                for label_dir in label_directories:
                    label_file_path = os.path.join(label_dir, filename)
                    if os.path.exists(label_file_path):
                        sitk_label = sitk.ReadImage(label_file_path)

                        # 重采样标签使其与参考图像的几何属性一致
                        is_same_geometry = verify_same_geometry(reference_image, sitk_label)

                        if not is_same_geometry:
                            print(f"Geometry mismatch detected for {filename} and its label.")
                            resampled_image = resample_label_to_image(sitk_label, reference_image)
                            # 保存重采样后的标签
                            sitk.WriteImage(resampled_image, os.path.join(target_path, filename))

                        label_found = True
                        break
                if not label_found:
                    print(f"Label file not found for {filename}.")

# # 参数设置
target_spacing_percentile = 50
anisotropy_threshold = 3

# # 计算初始目标间距
target = np.percentile(np.vstack(spacings), target_spacing_percentile, 0)

# This should be used to determine the new median shape. The old implementation is not 100% correct.
# Fixed in 2.4
# sizes = [np.array(i) / target * np.array(j) for i, j in zip(spacings, sizes)]

target_size = np.percentile(np.vstack(sizes),target_spacing_percentile, 0)
target_size_mm = np.array(target) * np.array(target_size)
# we need to identify datasets for which a different target spacing could be beneficial. These datasets have
# the following properties:
# - one axis which much lower resolution than the others
# - the lowres axis has much less voxels than the others
# - (the size in mm of the lowres axis is also reduced)
worst_spacing_axis = np.argmax(target)
other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
other_spacings = [target[i] for i in other_axes]
other_sizes = [target_size[i] for i in other_axes]

has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings))
has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)
# we don't use the last one for now
#median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)

if has_aniso_spacing and has_aniso_voxels:
    spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
    target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
    # don't let the spacing of that axis get higher than the other axes
    if target_spacing_of_that_axis < max(other_spacings):
        target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
    target[worst_spacing_axis] = target_spacing_of_that_axis

print("调整后的目标间距:", target)
