"""
Example preprocessing script for MRICombo datasets

This script demonstrates how to preprocess MRI data to match the expected format.
It includes resampling, normalization, and cropping/padding.

Usage:
    python preprocessing_example.py --input_dir /path/to/raw/data --output_dir ./dataset/MR_Dataset/0BraTS --dataset brats

Author: MRICombo Team
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm


def resample_to_spacing(image, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    """
    Resample image to target spacing
    
    Args:
        image: SimpleITK image
        target_spacing: tuple of (spacing_x, spacing_y, spacing_z)
        is_label: if True, use nearest neighbor interpolation
        
    Returns:
        Resampled SimpleITK image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    
    # Use nearest neighbor for labels, linear for images
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)


def normalize_intensity(array, lower_percentile=0.5, upper_percentile=99.5):
    """
    Normalize intensity using percentile clipping and z-score
    
    Args:
        array: numpy array
        lower_percentile: lower percentile for clipping
        upper_percentile: upper percentile for clipping
        
    Returns:
        Normalized array
    """
    # Create mask (non-zero values)
    mask = array > 0
    
    if not np.any(mask):
        return array
    
    # Clip to percentiles
    p_low = np.percentile(array[mask], lower_percentile)
    p_high = np.percentile(array[mask], upper_percentile)
    array = np.clip(array, p_low, p_high)
    
    # Z-score normalization
    mean = np.mean(array[mask])
    std = np.std(array[mask])
    
    if std > 0:
        array[mask] = (array[mask] - mean) / std
    
    return array


def center_crop_or_pad(array, target_shape=(128, 128, 128)):
    """
    Center crop or pad array to target shape
    
    Args:
        array: numpy array (D, H, W)
        target_shape: target shape tuple
        
    Returns:
        Cropped/padded array
    """
    current_shape = array.shape
    
    # Calculate crop/pad for each dimension
    starts = []
    ends = []
    pads = []
    
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            # Need to crop
            start = (current_shape[i] - target_shape[i]) // 2
            starts.append(start)
            ends.append(start + target_shape[i])
            pads.append((0, 0))
        else:
            # Will pad (or exact match)
            starts.append(0)
            ends.append(current_shape[i])
            total_pad = target_shape[i] - current_shape[i]
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pads.append((pad_before, pad_after))
    
    # Crop
    cropped = array[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    # Pad if necessary
    if any(p != (0, 0) for p in pads):
        cropped = np.pad(cropped, pads, mode='constant', constant_values=0)
    
    return cropped


def preprocess_case(input_paths, output_dir, patient_id, target_shape=(128, 128, 128)):
    """
    Preprocess one case (all sequences)
    
    Args:
        input_paths: dict mapping sequence name to file path
        output_dir: output directory
        patient_id: patient identifier
        target_shape: target image shape
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for seq_name, input_path in input_paths.items():
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping")
            continue
        
        is_label = (seq_name == 'seg')
        output_path = os.path.join(output_dir, f"{patient_id}_{seq_name}.nii.gz")
        
        # Read image
        image = sitk.ReadImage(input_path)
        
        # Step 1: Resample to 1mm isotropic
        resampled = resample_to_spacing(image, target_spacing=(1.0, 1.0, 1.0), is_label=is_label)
        
        # Convert to numpy
        array = sitk.GetArrayFromImage(resampled)  # (D, H, W)
        
        # Step 2: Normalize (skip for labels)
        if not is_label:
            array = normalize_intensity(array)
        
        # Step 3: Crop/pad to target size
        array = center_crop_or_pad(array, target_shape=target_shape)
        
        # Convert back to SimpleITK
        output_image = sitk.GetImageFromArray(array)
        output_image.SetSpacing((1.0, 1.0, 1.0))
        output_image.SetOrigin((0, 0, 0))
        output_image.SetDirection(np.eye(3).flatten())
        
        # Save
        sitk.WriteImage(output_image, output_path)


def preprocess_brats_dataset(input_dir, output_dir, target_shape=(128, 128, 128)):
    """Preprocess BraTS dataset"""
    patient_dirs = sorted(glob(os.path.join(input_dir, "*")))
    
    print(f"Found {len(patient_dirs)} patients in {input_dir}")
    
    for patient_dir in tqdm(patient_dirs, desc="Processing BraTS"):
        patient_id = os.path.basename(patient_dir)
        
        # Expected files (adjust naming based on your data)
        input_paths = {
            't1': os.path.join(patient_dir, f"{patient_id}_t1.nii.gz"),
            't1ce': os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz"),
            't2': os.path.join(patient_dir, f"{patient_id}_t2.nii.gz"),
            'flair': os.path.join(patient_dir, f"{patient_id}_flair.nii.gz"),
            'seg': os.path.join(patient_dir, f"{patient_id}_seg.nii.gz"),
        }
        
        patient_output_dir = os.path.join(output_dir, patient_id)
        preprocess_case(input_paths, patient_output_dir, patient_id, target_shape)


def preprocess_npc_dataset(input_dir, output_dir, target_shape=(128, 128, 128)):
    """Preprocess NPC dataset"""
    patient_dirs = sorted(glob(os.path.join(input_dir, "*")))
    
    print(f"Found {len(patient_dirs)} patients in {input_dir}")
    
    for patient_dir in tqdm(patient_dirs, desc="Processing NPC"):
        patient_id = os.path.basename(patient_dir)
        
        input_paths = {
            't1': os.path.join(patient_dir, f"{patient_id}_t1.nii.gz"),
            't1c': os.path.join(patient_dir, f"{patient_id}_t1c.nii.gz"),
            't2': os.path.join(patient_dir, f"{patient_id}_t2.nii.gz"),
            'seg': os.path.join(patient_dir, f"{patient_id}_seg.nii.gz"),
        }
        
        patient_output_dir = os.path.join(output_dir, patient_id)
        preprocess_case(input_paths, patient_output_dir, patient_id, target_shape)


def preprocess_generic_dataset(input_dir, output_dir, sequences, target_shape=(128, 128, 128)):
    """
    Preprocess generic dataset
    
    Args:
        input_dir: input directory containing patient folders
        output_dir: output directory
        sequences: list of sequence names (e.g., ['t1', 't2', 'seg'])
        target_shape: target image shape
    """
    patient_dirs = sorted(glob(os.path.join(input_dir, "*")))
    
    print(f"Found {len(patient_dirs)} patients in {input_dir}")
    print(f"Expected sequences: {sequences}")
    
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        patient_id = os.path.basename(patient_dir)
        
        input_paths = {}
        for seq in sequences:
            input_paths[seq] = os.path.join(patient_dir, f"{patient_id}_{seq}.nii.gz")
        
        patient_output_dir = os.path.join(output_dir, patient_id)
        preprocess_case(input_paths, patient_output_dir, patient_id, target_shape)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MRI data for MRICombo")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=['brats', 'npc', 'ispy', 'prostate', 'generic'],
                        help="Dataset type")
    parser.add_argument("--sequences", type=str, nargs='+', 
                        help="Sequence names for generic dataset (e.g., t1 t2 seg)")
    parser.add_argument("--target_size", type=int, default=128, 
                        help="Target image size (cubic)")
    
    args = parser.parse_args()
    
    target_shape = (args.target_size, args.target_size, args.target_size)
    
    print("="*60)
    print("MRICombo Data Preprocessing")
    print("="*60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Target shape: {target_shape}")
    print("="*60)
    
    if args.dataset == 'brats':
        preprocess_brats_dataset(args.input_dir, args.output_dir, target_shape)
    elif args.dataset == 'npc':
        preprocess_npc_dataset(args.input_dir, args.output_dir, target_shape)
    elif args.dataset == 'generic':
        if not args.sequences:
            raise ValueError("--sequences required for generic dataset")
        preprocess_generic_dataset(args.input_dir, args.output_dir, args.sequences, target_shape)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    print("\n" + "="*60)
    print("Preprocessing completed!")
    print(f"Output saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

