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

def locate_bbx(label, scaler, crop_h, crop_w, crop_d):
    import math
    import random
    import numpy as np
    import torch
    
    # check if the input is a tensor
    is_tensor = torch.is_tensor(label)
    
    # calculate the crop dimension according to the provided scaler
    scale_d = int(crop_d * scaler)
    scale_h = int(crop_h * scaler)
    scale_w = int(crop_w * scaler)
    
    # get the coordinates of the foreground region
    if is_tensor:
        bbx = torch.where(label >= 1)
        img_h, img_w, img_d = label.shape
        # for tensor input, convert the bounding box to a NumPy array on the CPU for processing
        boud_h = bbx[0].cpu().numpy() if torch.is_tensor(bbx[0]) else bbx[0]
        boud_w = bbx[1].cpu().numpy() if torch.is_tensor(bbx[1]) else bbx[1]
        boud_d = bbx[2].cpu().numpy() if torch.is_tensor(bbx[2]) else bbx[2]
    else:
        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = np.where(label >= 1)
    
    margin = 32  # extra pixels added around the bounding box
    
    # find the minimum and maximum coordinates of the bounding box
    bbx_h_min = boud_h.min()
    bbx_h_max = boud_h.max()
    bbx_w_min = boud_w.min()
    bbx_w_max = boud_w.max()
    bbx_d_min = boud_d.min()
    bbx_d_max = boud_d.max()
    
    # if the height dimension of the bounding box is less than the required scale_h
    if (bbx_h_max - bbx_h_min) <= scale_h:
        bbx_h_maxt = bbx_h_max + math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
        bbx_h_mint = bbx_h_min - math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
        if bbx_h_mint < 0:
            bbx_h_maxt -= bbx_h_mint
            bbx_h_mint = 0
        bbx_h_max = bbx_h_maxt
        bbx_h_min = bbx_h_mint
    
    # width dimension processing
    if (bbx_w_max - bbx_w_min) <= scale_w:
        bbx_w_maxt = bbx_w_max + math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
        bbx_w_mint = bbx_w_min - math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
        if bbx_w_mint < 0:
            bbx_w_maxt -= bbx_w_mint
            bbx_w_mint = 0
        bbx_w_max = bbx_w_maxt
        bbx_w_min = bbx_w_mint
    
                # depth dimension processing
    if (bbx_d_max - bbx_d_min) <= scale_d:
        bbx_d_maxt = bbx_d_max + math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
        bbx_d_mint = bbx_d_min - math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
        if bbx_d_mint < 0:
            bbx_d_maxt -= bbx_d_mint
            bbx_d_mint = 0
        bbx_d_max = bbx_d_maxt
        bbx_d_min = bbx_d_mint
    
    # add margin but ensure it does not exceed the image boundary
    if is_tensor:
        # for tensor input
        bbx_h_min = max(bbx_h_min - margin, 0)
        bbx_h_max = min(bbx_h_max + margin, img_h)
        bbx_w_min = max(bbx_w_min - margin, 0)
        bbx_w_max = min(bbx_w_max + margin, img_w)
        bbx_d_min = max(bbx_d_min - margin, 0)
        bbx_d_max = min(bbx_d_max + margin, img_d)
    else:
        # for numpy array
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])
    
    # ensure the region size is large enough to contain the crop window
    if bbx_h_max - bbx_h_min < scale_h:
        if bbx_h_min + scale_h <= img_h:
            bbx_h_max = bbx_h_min + scale_h
        else:
            bbx_h_min = max(0, img_h - scale_h)
            bbx_h_max = img_h
    
    if bbx_w_max - bbx_w_min < scale_w:
        if bbx_w_min + scale_w <= img_w:
            bbx_w_max = bbx_w_min + scale_w
        else:
            bbx_w_min = max(0, img_w - scale_w)
            bbx_w_max = img_w
    
    if bbx_d_max - bbx_d_min < scale_d:
        if bbx_d_min + scale_d <= img_d:
            bbx_d_max = bbx_d_min + scale_d
        else:
            bbx_d_min = max(0, img_d - scale_d)
            bbx_d_max = img_d
    
    # 80% probability to place the crop window randomly in the boundary region, the key fix point here
    # if random.random() < 0.8:
        # fix the problem that the random range may be empty
    h_start = int(bbx_h_min)
    h_end = max(int(bbx_h_max - scale_h), h_start)
    w_start = int(bbx_w_min)
    w_end = max(int(bbx_w_max - scale_w), w_start)
    d_start = int(bbx_d_min)
    d_end = max(int(bbx_d_max - scale_d), d_start)
    
    # if the start point equals the end point, use the start point instead of random
    h0 = h_start if h_start == h_end else random.randint(h_start, h_end)
    w0 = w_start if w_start == w_end else random.randint(w_start, w_end)
    d0 = d_start if d_start == d_end else random.randint(d_start, d_end)
    # else:
    #     # 20% probability to place the crop window randomly anywhere in the image
    #     h_max = max(0, int(img_h - scale_h))
    #     w_max = max(0, int(img_w - scale_w))
    #     d_max = max(0, int(img_d - scale_d))
        
    #     h0 = 0 if h_max == 0 else random.randint(0, h_max)
    #     w0 = 0 if w_max == 0 else random.randint(0, w_max)
    #     d0 = 0 if d_max == 0 else random.randint(0, d_max)
    
    # calculate the end coordinates
    h1 = h0 + scale_h
    w1 = w0 + scale_w
    d1 = d0 + scale_d
    
    # ensure it does not exceed the image boundary
    h1 = min(h1, img_h)
    w1 = min(w1, img_w)
    d1 = min(d1, img_d)
    
    # if the size is not enough due to the boundary limit, adjust the start point
    if h1 - h0 < scale_h:
        h0 = max(0, h1 - scale_h)
    if w1 - w0 < scale_w:
        w0 = max(0, w1 - scale_w)
    if d1 - d0 < scale_d:
        d0 = max(0, d1 - scale_d)
    
    # recalculate the end point to ensure the size is correct
    h1 = h0 + scale_h
    w1 = w0 + scale_w
    d1 = d0 + scale_d
    
    # if the input is a tensor, convert the result to a tensor
    if is_tensor:
        return torch.tensor([h0, h1, w0, w1, d0, d1], device=label.device)
    else:
        # for numpy input, return a python list
        return [h0, h1, w0, w1, d0, d1]


import torch
import numpy as np
from scipy.ndimage import rotate

def random_rotation_3d(img, random_plane, random_angle, is_label=False):
    """for 3D data rotation (supports [C, D, H, W] or [D, H, W])"""
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
        # the label is single channel [D, H, W] → directly rotate
        img = rotate(img, angle=random_angle, axes=random_plane,
                     reshape=False, order=order, mode=mode)

    # convert back to Tensor and ensure the label is integer
    img = torch.from_numpy(img)
    if is_label:
        img = img.long()
    return img


def rotate_3d_image_and_label(image, label, angle_spectrum=10, seed=None):
    """rotate 3D image and label"""
    if seed is not None:
        np.random.seed(seed)

    # optional rotation planes (for [D, H, W] data)
    planes = [(1, 2), (0, 2), (0, 1)]  # D-H, D-W, H-W 平面
    random_plane = planes[np.random.choice(len(planes))]
    random_angle = np.random.randint(-angle_spectrum, angle_spectrum)

        # rotate image and label
    rotated_image = random_rotation_3d(image, random_plane, random_angle, is_label=False)
    rotated_label = random_rotation_3d(label, random_plane, random_angle, is_label=True)

    return rotated_image, rotated_label

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

def brightness_multiply(tensor_img, multiply_range=[0.7, 1.3], per_channel=False):

    if per_channel:
        C = tensor_img.shape[0]
    else:
        C = 1

    assert multiply_range[1] > multiply_range[0], 'Invalid range'

    span = multiply_range[1] - multiply_range[0]
    if len(tensor_img.shape) == 4:
        rand_brightness = torch.rand(size=(C, 1, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    elif len(tensor_img.shape) == 3:
        rand_brightness = torch.rand(size=(1, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img * rand_brightness

def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    
    if per_channel:
        C = tensor_img.shape[0]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(C, 1, 1, 1)).to(tensor_img.device)
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1,1, 1)).to(tensor_img.device)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img + rand_brightness

def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=True):
    """
    apply gamma correction to the input tensor.
    Args:
        tensor_img (torch.Tensor): the input tensor, shape is [C, D, W, H].
        gamma_range (tuple): the range of gamma values.
        per_channel (bool): whether to apply gamma correction to each channel separately.
        retain_stats (bool): whether to retain the original statistical characteristics (mean and standard deviation).
    Returns:
        torch.Tensor: the tensor after gamma correction.
    """
    if len(tensor_img.shape) == 4:  # [C, D, W, H]
        C, D, W, H = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 4d for volume image.')

    # if per_channel is True, process each channel separately, otherwise视为单通道
    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)  # use reshape to flatten each channel

    # calculate the minimum, maximum, and range of each channel
    minm, _ = tensor_img.min(dim=1, keepdim=True)
    maxm, _ = tensor_img.max(dim=1, keepdim=True)
    rng = maxm - minm

    # calculate the mean and standard deviation of each channel
    mean = tensor_img.mean(dim=1, keepdim=True)
    std = tensor_img.std(dim=1, keepdim=True)

    # randomly generate gamma values
    gamma = torch.rand(tmp_C, 1).to(tensor_img.device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    # apply gamma correction
    tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

    # if retain_stats is True, retain the original statistical characteristics
    if retain_stats:
        tensor_img -= tensor_img.mean(dim=1, keepdim=True)
        tensor_img = tensor_img / tensor_img.std(dim=1, keepdim=True) * std + mean

    # restore the original shape
    return tensor_img.reshape(C, D, W, H)  # 使用 reshape 恢复原始形状

def contrast(tensor_img, contrast_range=(0.65, 1.5), per_channel=False, preserve_range=True):
    """
                    apply contrast adjustment to the input tensor.
    Args:
        tensor_img (torch.Tensor): the input tensor, shape is [C, D, W, H].
        contrast_range (tuple): the range of contrast adjustment.
        per_channel (bool): whether to adjust contrast per channel.
        preserve_range (bool): whether to preserve the original value range.
    Returns:
        torch.Tensor: the tensor after contrast adjustment.
    """
    if len(tensor_img.shape) == 4:  # [C, D, W, H]
        C, D, W, H = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 4d for volume image.')

    # if per_channel is True, process each channel separately, otherwise视为单通道
    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)  # flatten each channel

    # calculate the minimum, maximum, and mean of each channel
    minm, _ = tensor_img.min(dim=1, keepdim=True)
    maxm, _ = tensor_img.max(dim=1, keepdim=True)
    mean = tensor_img.mean(dim=1, keepdim=True)

    # randomly generate contrast adjustment factor
    factor = torch.rand(tmp_C, 1).to(tensor_img.device) * (contrast_range[1] - contrast_range[0]) + contrast_range[0]

    # apply contrast adjustment
    tensor_img = (tensor_img - mean) * factor + mean

    # if preserve_range is True, preserve the original value range
    if preserve_range:
        tensor_img = torch.clamp(tensor_img, min=minm, max=maxm)

    # restore the original shape
    return tensor_img.reshape(C, D, W, H)
def generate_3d_gaussian_kernel(kernel_size, sigma, channels):
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    z = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    return kernel

def gaussian_blur(tensor_img, sigma_range=[0.5, 1.0]):
    if len(tensor_img.shape) == 4:  # [C, D, W, H]
        C, D, W, H = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 4d for volume image.')

    sigma = torch.rand(1).item() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    kernel = generate_3d_gaussian_kernel(kernel_size, sigma, C).to(tensor_img.device)
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = F.conv3d(tensor_img, kernel, padding=kernel_size // 2, groups=C)
    return tensor_img.squeeze(0)

def gaussian_noise(tensor_img, std, mean=0):
    """
    add Gaussian noise to the input tensor.
    Args:
        tensor_img (torch.Tensor): the input tensor, shape is [C, D, W, H].
        std (float): the standard deviation of Gaussian noise.
        mean (float): the mean of Gaussian noise.
    Returns:
        torch.Tensor: the tensor after adding Gaussian noise.
    """
    if len(tensor_img.shape) == 4:  # [C, D, W, H]
        C, D, W, H = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 4d for volume image.')

    # add Gaussian noise
    noise = torch.randn_like(tensor_img) * std + mean
    return tensor_img + noise