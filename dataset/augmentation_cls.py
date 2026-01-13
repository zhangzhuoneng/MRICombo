import numpy as np
from skimage.measure import regionprops
import random
import torch
import math
import torch.nn.functional as F
import SimpleITK as sitk
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_nii(path):
    nii_file = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return nii_file

def Crop_brain_Foreground( patient_image, patient_label,name):
    nonzero_index = torch.nonzero(torch.sum(patient_image, axis=0)!=0)
    z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
    zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
    patient_label = patient_label[zmin:zmax, ymin:ymax, xmin:xmax]

    return  patient_image,patient_label

# def Crop_Foreground(patient_image, patient_label,context_size=[10, 10, 10]):# h w d
#    # create foreground mask
#     mask = (patient_label > 0).any(dim=0).to(torch.uint8)  # foreground mask

#     # use regionprops to get foreground region properties
#     mask_np = mask.cpu().numpy()  # 将 mask 转换为 NumPy 格式以兼容 regionprops
#     regions = regionprops(mask_np)
#     assert len(regions) == 1, "Expected exactly one foreground region."

#     # get the shape of the input tensor
#     Bs, zz, yy, xx = patient_image.shape

#     # get the centroid and bounding box of the foreground region
#     z, y, x = regions[0].centroid
#     z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox

#     # 转换为整数
#     z, y, x = int(z), int(y), int(x)

#     # 考虑上下文大小调整边界框
#     z_min = max(0, z_min - context_size[0])
#     z_max = min(zz, z_max + context_size[0])
#     y_min = max(0, y_min - context_size[1])
#     y_max = min(yy, y_max + context_size[1])
#     x_min = max(0, x_min - context_size[2])
#     x_max = min(xx, x_max + context_size[2])

#     # 裁剪图像和标签
#     patient_image = patient_image[:, z_min:z_max, y_min:y_max, x_min:x_max]
#     patient_label = patient_label[:, z_min:z_max, y_min:y_max, x_min:x_max]

#     return patient_image, patient_label


def Crop_Foreground(patient_image, patient_label,context_size=[10, 10, 10]):# h w d
   # create foreground mask
    mask = (patient_label>0).to(torch.uint8) # foreground mask 
    mask = mask.cpu().numpy()
    regions = regionprops(mask)
    assert len(regions) == 1

    C,zz, yy, xx = patient_image.shape

    z, y, x = regions[0].centroid

    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox

    # convert to integer
    z, y, x = int(z), int(y), int(x)

    # consider the context size to adjust the bounding box
    z_min = max(0, z_min - context_size[0])
    z_max = min(zz, z_max + context_size[0])
    y_min = max(0, y_min - context_size[1])
    y_max = min(yy, y_max + context_size[1])
    x_min = max(0, x_min - context_size[2])
    x_max = min(xx, x_max + context_size[2])

    # crop the image and label
    patient_image = patient_image[:, z_min:z_max, y_min:y_max, x_min:x_max]
    patient_label = patient_label[z_min:z_max, y_min:y_max, x_min:x_max]
   
    return patient_image, patient_label



from scipy.ndimage import rotate

def random_rotation_3d(img, random_plane, random_angle, is_label=False):
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # rotation parameters: label use nearest neighbor, image use bilinear interpolation
    order = 0 if is_label else 1
    mode = 'nearest' if is_label else 'reflect'

    if not is_label:
        # the image is multi-channel [C, D, H, W] → rotate each channel
        rotated_channels = [
            rotate(img[c], angle=random_angle, axes=random_plane,
                   reshape=False, order=order, mode=mode)
            for c in range(img.shape[0])
        ]
        img = np.stack(rotated_channels, axis=0)
    else:
        # the label is single channel [D, H, W] → rotate directly
        img = rotate(img, angle=random_angle, axes=random_plane,
                     reshape=False, order=order, mode=mode)

    # convert back to Tensor, and ensure the label is integer
    img = torch.from_numpy(img)
    if is_label:
        img = img.long()
    return img


def rotate_3d_image_and_label(image, angle_spectrum=10, seed=None):
    """rotate 3D image and label"""
    if seed is not None:
        np.random.seed(seed)

    # optional rotation planes (for [D, H, W] data)
    planes = [(1, 2), (0, 2), (0, 1)]  # D-H, D-W, H-W 平面
    random_plane = planes[np.random.choice(len(planes))]
    random_angle = np.random.randint(-angle_spectrum, angle_spectrum)
    # rotate the image and label
    rotated_image = random_rotation_3d(image, random_plane, random_angle, is_label=False)
    # rotated_label = random_rotation_3d(label, random_plane, random_angle, is_label=True)

    return rotated_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math
import pdb

# This is a PyTorch data augmentation library, that takes PyTorch Tensor as input
# Functions can be applied in the __getitem__ function to do augmentation on the fly during training.
# These functions can be easily parallelized by setting 'num_workers' in pytorch dataloader.

# tensor_img: 1, C, (D), H, W

def gaussian_noise(tensor_img, std, mean=0):
    
    return tensor_img + torch.randn(tensor_img.shape).to(tensor_img.device) * std + mean

def generate_2d_gaussian_kernel(kernel_size, sigma):
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y,indexing='ij')

    # Calculate the 2D Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2)
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)

def generate_3d_gaussian_kernel(kernel_size, sigma):
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    z = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(x, y, z,indexing='ij')

    # Calculate the 3D Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2) ** 1.5
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_blur(tensor_img, sigma_range=[0.5, 1.0]):

    sigma = torch.rand(1) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        kernel = generate_3d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(3)]

        # Apply Gaussian blur to each channel independently
        blurred = []
        for c in range(tensor_img.shape[1]):  # Iterate over channels
            blurred_channel = F.conv3d(
                tensor_img[:, c:c+1, :, :, :],  # Select one channel
                kernel,
                padding=padding
            )
            blurred.append(blurred_channel)
        
        # Concatenate blurred channels
        return torch.cat(blurred, dim=1)
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        kernel = generate_2d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(2)]

        return F.conv2d(tensor_img, kernel, padding=padding)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')


def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    
    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(tensor_img.device)
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(tensor_img.device)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img + rand_brightness


def brightness_multiply(tensor_img, multiply_range=[0.7, 1.3], per_channel=False):

    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    assert multiply_range[1] > multiply_range[0], 'Invalid range'

    span = multiply_range[1] - multiply_range[0]
    if len(tensor_img.shape) == 5:
        rand_brightness = torch.rand(size=(1, C, 1, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.rand(size=(1, C, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img * rand_brightness


def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=True):
    """
    Apply gamma correction to a tensor image.

    Args:
        tensor_img: Input tensor of shape [B, C, D, H, W] (3D) or [B, C, H, W] (2D).
        gamma_range: Tuple specifying the range of gamma adjustment.
        per_channel: Whether to adjust gamma per channel.
        retain_stats: Whether to retain the original mean and std.

    Returns:
        Tensor with adjusted gamma.
    """
    if len(tensor_img.shape) == 5:  # 3D case
        dim = '3d'
        B, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:  # 2D case
        dim = '2d'
        B, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')
    
    tmp_C = C if per_channel else 1
    tensor_img_flat = tensor_img.view(B * tmp_C, -1)  # Flatten the tensor for processing
    minm, _ = tensor_img_flat.min(dim=1, keepdim=True)
    maxm, _ = tensor_img_flat.max(dim=1, keepdim=True)

    rng = maxm - minm
    mean = tensor_img_flat.mean(dim=1, keepdim=True)
    std = tensor_img_flat.std(dim=1, keepdim=True)
    gamma = torch.rand(C if per_channel else 1, 1).to(tensor_img.device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    tensor_img_flat = torch.pow((tensor_img_flat - minm) / rng, gamma) * rng + minm

    if retain_stats:
        tensor_img_flat -= tensor_img_flat.mean(dim=1, keepdim=True)
        tensor_img_flat = tensor_img_flat / tensor_img_flat.std(dim=1, keepdim=True) * std + mean

    # Reshape back to the original shape
    if dim == '3d':
        return tensor_img_flat.view(B, C, D, H, W)
    else:
        return tensor_img_flat.view(B, C, H, W)
        
def contrast(tensor_img, contrast_range=(0.65, 1.5), per_channel=False, preserve_range=True):
    """
    Adjust the contrast of a tensor image.

    Args:
        tensor_img: Input tensor of shape [B, C, D, H, W] (3D) or [B, C, H, W] (2D).
        contrast_range: Tuple specifying the range of contrast adjustment.
        per_channel: Whether to adjust contrast per channel.
        preserve_range: Whether to clamp the values to the original range.

    Returns:
        Tensor with adjusted contrast.
    """
    if len(tensor_img.shape) == 5:  # 3D case
        dim = '3d'
        B, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:  # 2D case
        dim = '2d'
        B, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    tmp_C = C if per_channel else 1
    tensor_img_flat = tensor_img.view(B * tmp_C, -1)  # Flatten the tensor for processing
    minm, _ = tensor_img_flat.min(dim=1, keepdim=True)
    maxm, _ = tensor_img_flat.max(dim=1, keepdim=True)

    mean = tensor_img_flat.mean(dim=1, keepdim=True)
    factor = torch.rand(C if per_channel else 1, 1).to(tensor_img.device) * (contrast_range[1] - contrast_range[0]) + contrast_range[0]

    tensor_img_flat = (tensor_img_flat - mean) * factor + mean

    if preserve_range:
        tensor_img_flat = torch.clamp(tensor_img_flat, min=minm, max=maxm)

    # Reshape back to the original shape
    if dim == '3d':
        return tensor_img_flat.view(B, C, D, H, W)
    else:
        return tensor_img_flat.view(B, C, H, W)

def mirror(tensor_img, axis=0):

    '''
    Args:
        tensor_img: an image with format of pytorch tensor
        axis: the axis for mirroring. 0 for the first image axis, 1 for the second, 2 for the third (if volume image)
    '''


    if len(tensor_img.shape) == 5:
        dim = '3d'
        assert axis in [0, 1, 2], "axis should be either 0, 1 or 2 for volume images"

    elif len(tensor_img.shape) == 4:
        dim = '2d'
        assert axis in [0, 1], "axis should be either 0 or 1 for 2D images"
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')


    return torch.flip(tensor_img, dims=[2+axis])


def random_scale_rotate_translate_2d(tensor_img, tensor_lab, scale, rotate, translate):

    # implemented with affine transformation

    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 2
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 2
    

    scale_x = 1 - scale[0] + np.random.random() * 2*scale[0]
    scale_y = 1 - scale[1] + np.random.random() * 2*scale[1]
    shear_x = np.random.random() * 2*scale[0] - scale[0] 
    shear_y = np.random.random() * 2*scale[1] - scale[1]
    translate_x = np.random.random() * 2*translate[0] - translate[0]
    translate_y = np.random.random() * 2*translate[1] - translate[1]

    theta_scale = torch.tensor([[scale_x, shear_x, translate_x], 
                                [shear_y, scale_y, translate_y],
                                [0, 0, 1]]).float()
    angle = (float(np.random.randint(-rotate, max(rotate, 1))) / 180.) * math.pi

    theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0],
                                [0, 0, 1]]).float()
    
    theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
    grid = F.affine_grid(theta.unsqueeze(0), tensor_img.size(), align_corners=True).to(tensor_img.device)

    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()

    return tensor_img, tensor_lab

def random_scale_rotate_translate_3d(tensor_img, tensor_lab, scale=0.3, rotate=45, translate=0.1, shear=0.1):
    '''
    The axis order of SimpleITK is x,y,z
    The axis order of numpy/tensor is z,y,x
    The arguments of all transformation should use the numpy/tensor order: [z,y,x]

    '''
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 3

    scale_x = np.random.uniform(low=1-scale[0], high=1/(1-scale[0]))
    scale_y = np.random.uniform(low=1-scale[1], high=1/(1-scale[1]))
    scale_z = np.random.uniform(low=1-scale[2], high=1/(1-scale[2]))

    shear_xy = np.random.uniform(-shear[0], shear[0]) # contribution of y index to x axis
    shear_xz = np.random.uniform(-shear[0], shear[0]) # contribution of z index to x axis
    shear_yx = np.random.uniform(-shear[1], shear[1]) # contribution of x index to y axis
    shear_yz = np.random.uniform(-shear[1], shear[1]) # contribution of z index to y axis
    shear_zx = np.random.uniform(-shear[2], shear[2]) # contribution of x index to z axis
    shear_zy = np.random.uniform(-shear[2], shear[2]) # contribution of y index to z axis

    translate_x = np.random.uniform(-translate[0], translate[0])
    translate_y = np.random.uniform(-translate[1], translate[1])
    translate_z = np.random.uniform(-translate[2], translate[2])


    theta_scale = torch.tensor([[scale_x, shear_xy, shear_xz, translate_x],
                                [shear_yx, scale_y, shear_yz, translate_y],
                                [shear_zx, shear_zy, scale_z, translate_z], 
                                [0, 0, 0, 1]]).float()
    angle_x = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi 
    # rotate along x axis (x index fix, rotae in yz plane)
    angle_y = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
    # rotate along y axis (y index fix, rotate in xz plane)
    angle_z = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    # rotate along z axis (z index fix, rotate in xy plane)
    
    theta_rotate_x = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(angle_x), -math.sin(angle_x), 0],
                                    [0, math.sin(angle_x), math.cos(angle_x), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_y = torch.tensor([[math.cos(angle_y), 0, -math.sin(angle_y), 0],
                                    [0, 1, 0, 0],
                                    [math.sin(angle_y), 0, math.cos(angle_y), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_z = torch.tensor([[math.cos(angle_z), -math.sin(angle_z), 0, 0],
                                    [math.sin(angle_z), math.cos(angle_z), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).float()

    theta = torch.mm(theta_rotate_x, theta_rotate_y)
    theta = torch.mm(theta, theta_rotate_z)
    
    theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    grid = F.affine_grid(theta, tensor_img.size(), align_corners=True).to(tensor_img.device)
    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()

    return tensor_img, tensor_lab
    


  

def crop_2d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 2

    _, _, H, W = tensor_img.shape

    diff_H = H - crop_size[0]
    diff_W = W - crop_size[1]
    
    if mode == 'random':
        rand_x = np.random.randint(0, max(diff_H, 1))
        rand_y = np.random.randint(0, max(diff_W, 1))
    else:
        rand_x = diff_H // 2
        rand_y = diff_W // 2

    cropped_img = tensor_img[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]
    cropped_lab = tensor_lab[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]

    return cropped_img.contiguous(), cropped_lab.congiguous()


def crop_3d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()

def np_crop_3d(np_img, np_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    D, H, W = np_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = np_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = np_lab[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)
import torch

def tensor_crop_3d(tensor_img, tensor_lab, crop_size, mode):
    """
    Args:
        tensor_img: PyTorch tensor of shape [D, H, W]
        tensor_lab: PyTorch tensor of shape [D, H, W]
        crop_size: List or int, size of the crop [crop_D, crop_H, crop_W]s
        mode: 'random' or 'center', determines the cropping mode
    Returns:
        cropped_img: Cropped tensor image
        cropped_lab: Cropped tensor label
    """
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]

    if mode == 'random':
        rand_z = torch.randint(0, max(diff_D, 1), (1,)).item()
        rand_y = torch.randint(0, max(diff_H, 1), (1,)).item()
        rand_x = torch.randint(0, max(diff_W, 1), (1,)).item()
    else:  # mode == 'center'
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = tensor_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()

def crop_around_coordinate_3d(tensor_img, tensor_lab, crop_size, coordinate, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    z, y, x = coordinate

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    
    if mode == 'random':
        min_z = max(0, z-crop_size[0])
        max_z = min(diff_D, z+crop_size[0])
        min_y = max(0, y-crop_size[1])
        max_y = min(diff_H, y+crop_size[1])
        min_x = max(0, x-crop_size[2])
        max_x = min(diff_W, x+crop_size[2])
        
        rand_z = np.random.randint(min_z, max_z)
        rand_y = np.random.randint(min_y, max_y)
        rand_x = np.random.randint(min_x, max_x)
    else:
        min_z = max(0, z - math.ceil(crop_size[0] / 2))
        rand_z = min(min_z, D - crop_size[0])
        min_y = max(0, y - math.ceil(crop_size[1] / 2))
        rand_y = min(min_y, H - crop_size[1])
        min_x = max(0, x - math.ceil(crop_size[2] / 2))
        rand_x = min(min_x, W - crop_size[2])

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()


def np_crop_around_coordinate_3d(np_img, np_lab, crop_size, coordinate, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    z, y, x = coordinate

    D, H, W = np_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    
    if mode == 'random':
        min_z = max(0, z-crop_size[0])
        max_z = min(diff_D, z+crop_size[0])
        min_y = max(0, y-crop_size[1])
        max_y = min(diff_H, y+crop_size[1])
        min_x = max(0, x-crop_size[2])
        max_x = min(diff_W, x+crop_size[2])
        
        rand_z = np.random.randint(min_z, max_z)
        rand_y = np.random.randint(min_y, max_y)
        rand_x = np.random.randint(min_x, max_x)
    else:
        min_z = max(0, z - math.ceil(crop_size[0] / 2))
        rand_z = min(min_z, D - crop_size[0])
        min_y = max(0, y - math.ceil(crop_size[1] / 2))
        rand_y = min(min_y, H - crop_size[1])
        min_x = max(0, x - math.ceil(crop_size[2] / 2))
        rand_x = min(min_x, W - crop_size[2])

    cropped_img = np_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = np_lab[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)