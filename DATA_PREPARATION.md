# 📊 Data Preparation Guide

This document provides detailed instructions for preparing your data to work with MRICombo.

## 📁 Expected Directory Structure

```
MRICombo/
├── code/                          # Source code
├── network/                       # Model architectures
├── dataset/                       # Dataset root directory
│   ├── MR_Dataset/               # Raw MRI data
│   │   ├── 0BraTS/               # Brain tumor (BraTS)
│   │   ├── 1HNTS/                # Head-neck tumor
│   │   ├── 2NPC/                 # Nasopharyngeal carcinoma
│   │   ├── 3ISPY/                # Breast cancer (ISPY)
│   │   ├── 4ATLAS/               # Brain stroke
│   │   ├── 5Colorectal/          # Colorectal
│   │   ├── 6AMOS/                # Abdominal organs
│   │   ├── 7BCa_seg/             # Breast cancer segmentation
│   │   ├── 8ProstateX/           # Prostate cancer
│   │   └── 9csPCa_seg/           # Clinically significant prostate cancer
│   │
│   ├── classification/           # Classification task lists
│   │   ├── cls_train.txt        # Training classification list
│   │   └── cls_test.txt         # Testing classification list
│   │
│   └── segmentation/             # Segmentation task lists
│       ├── seg_train.txt        # Training segmentation list
│       ├── seg_val.txt          # Validation segmentation list
│       └── seg_test.txt         # Testing segmentation list
│
├── snapshots/                     # Model checkpoints
└── log/                          # Training logs
```

## 🗂️ Dataset Organization

### General Structure

Each dataset follows this hierarchical structure:

```
dataset/MR_Dataset/{DATASET_ID}{DATASET_NAME}/
└── {PATIENT_ID}/
    ├── {PATIENT_ID}_{SEQUENCE1}.nii.gz
    ├── {PATIENT_ID}_{SEQUENCE2}.nii.gz
    └── {PATIENT_ID}_seg.nii.gz          # Segmentation label
```

### Dataset-Specific Organization

#### 1. BraTS (Brain Tumor Segmentation)
```
0BraTS/
└── BraTS_001/
    ├── BraTS_001_t1.nii.gz       # T1-weighted
    ├── BraTS_001_t1ce.nii.gz     # T1-contrast enhanced
    ├── BraTS_001_t2.nii.gz       # T2-weighted
    ├── BraTS_001_flair.nii.gz    # FLAIR
    └── BraTS_001_seg.nii.gz      # Labels: 0=background, 1=NCR/NET, 2=ED, 4=ET
```

**Sequences**: T1, T1ce, T2, FLAIR  
**Classes**: 3 (Enhancing Tumor, Tumor Core, Whole Tumor)  
**Label Encoding**:
- 0: Background
- 1: Necrotic/Non-enhancing tumor core
- 2: Peritumoral edema
- 4: Enhancing tumor

#### 2. HNTS (Head and Neck Tumor)
```
1HNTS/
└── HNTS_001/
    ├── HNTS_001_t2.nii.gz        # T2-weighted
    └── HNTS_001_seg.nii.gz       # Labels: 1=tumor
```

**Sequences**: T2  
**Classes**: 1 (Tumor)

#### 3. NPC (Nasopharyngeal Carcinoma)
```
2NPC/
└── NPC_001/
    ├── NPC_001_t1.nii.gz         # T1-weighted
    ├── NPC_001_t1c.nii.gz        # T1-contrast
    ├── NPC_001_t2.nii.gz         # T2-weighted
    └── NPC_001_seg.nii.gz        # Labels: 1=GTVnx, 2=GTVnd
```

**Sequences**: T1, T1c, T2  
**Classes**: 2 (Primary tumor GTVnx, Lymph nodes GTVnd)  
**Classification**: T-staging (T1-T2 vs T3-T4)

#### 4. ISPY (Breast Cancer)
```
3ISPY/
└── ISPY_001/
    ├── ISPY_001_dce1.nii.gz      # DCE phase 1
    ├── ISPY_001_dce2.nii.gz      # DCE phase 2
    └── ISPY_001_seg.nii.gz       # Labels: 1=tumor
```

**Sequences**: DCE (2 phases)  
**Classes**: 1 (Tumor)  
**Classification**: Malignancy (benign vs malignant)

#### 5. ATLAS (Brain Stroke)
```
4ATLAS/
└── ATLAS_001/
    ├── ATLAS_001_t1.nii.gz       # T1-weighted
    └── ATLAS_001_seg.nii.gz      # Labels: 1=lesion
```

**Sequences**: T1  
**Classes**: 1 (Stroke lesion)

#### 6. Colorectal
```
5Colorectal/
└── Colorectal_001/
    ├── Colorectal_001_t2.nii.gz  # T2-weighted
    └── Colorectal_001_seg.nii.gz # Labels: 1=tumor
```

**Sequences**: T2  
**Classes**: 1 (Tumor)

#### 7. AMOS (Abdominal Multi-Organ Segmentation)
```
6AMOS/
└── AMOS_001/
    ├── AMOS_001_ct.nii.gz        # CT or T1
    └── AMOS_001_seg.nii.gz       # Labels: 1-15 (multi-organ)
```

**Sequences**: CT/T1  
**Classes**: 15 organs  
**Label Encoding**:
1. Spleen, 2. Right kidney, 3. Left kidney, 4. Gallbladder
5. Esophagus, 6. Liver, 7. Stomach, 8. Aorta
9. IVC, 10. Portal/Splenic veins, 11. Pancreas, 12. Right adrenal
13. Left adrenal, 14. Duodenum, 15. Bladder

#### 8. BCa_seg (Breast Cancer Segmentation)
```
7BCa_seg/
└── BCa_001/
    ├── BCa_001_t1.nii.gz         # T1-weighted
    └── BCa_001_seg.nii.gz        # Labels: 1=tumor
```

**Sequences**: T1  
**Classes**: 1 (Tumor)

#### 9. ProstateX
```
8ProstateX/
└── ProstateX_001/
    ├── ProstateX_001_adc.nii.gz  # ADC map
    ├── ProstateX_001_t2.nii.gz   # T2-weighted
    └── ProstateX_001_seg.nii.gz  # Labels: 1=lesion
```

**Sequences**: ADC, T2  
**Classes**: 1 (Lesion)  
**Classification**: MIBC vs NMIBC

#### 10. csPCa_seg (Clinically Significant Prostate Cancer)
```
9csPCa_seg/
└── csPCa_001/
    ├── csPCa_001_adc.nii.gz      # ADC map
    ├── csPCa_001_dwi.nii.gz      # DWI
    ├── csPCa_001_t2.nii.gz       # T2-weighted
    └── csPCa_001_seg.nii.gz      # Labels: 1=lesion
```

**Sequences**: ADC, DWI, T2  
**Classes**: 1 (Clinically significant lesion)


### Classification Datasets (11-15)

The following datasets are primarily used for classification tasks (tumor grading, staging, malignancy detection):

#### 11. FedBca (Federated Breast Cancer)
```
11FedBca/
└── center1_001/
    ├── center1_001_t2.nii.gz     # T2-weighted
    └── center1_001_seg.nii.gz    # Segmentation labels
```

**Sequences**: T2  
**Classification Task**: Benign vs Malignant breast tumors  
**Segmentation Classes**: 1 (Tumor)  
**Classification Labels**: 0=benign, 1=malignant

#### 12. NPC (Nasopharyngeal Carcinoma - Classification)
```
12NPC/
└── NPC_001/
    ├── NPC_001_t1.nii.gz         # T1-weighted
    ├── NPC_001_t1c.nii.gz        # T1-contrast
    ├── NPC_001_t2.nii.gz         # T2-weighted
    └── NPC_001_seg.nii.gz        # Segmentation labels
```

**Sequences**: T1, T1c, T2  
**Classification Task**: T-staging (T1-T2 vs T3-T4)  
**Segmentation Classes**: 2 (GTVnx, GTVnd)  
**Classification Labels**: 0=T1-T2 (early stage), 1=T3-T4 (advanced stage)

#### 13. LLD (Liver Lesion Detection)
```
13LLD/
└── LLD_MR102385/
    ├── LLD_MR102385_C+A.nii.gz   # Arterial phase (contrast-enhanced)
    └── LLD_MR102385_C+V.nii.gz   # Venous phase (contrast-enhanced)
```

**Sequences**: C+A (Arterial phase), C+V (Venous phase)  
**Classification Task**: Benign vs Malignant liver lesions  
**Classification Labels**: 0=benign, 1=malignant

#### 14. BraTS (Brain Tumor - Classification)
```
14BraTS/
└── Brats18_2013_0_1/
    ├── Brats18_2013_0_1_t1.nii.gz       # T1-weighted
    ├── Brats18_2013_0_1_t1ce.nii.gz     # T1-contrast enhanced
    ├── Brats18_2013_0_1_t2.nii.gz       # T2-weighted
    ├── Brats18_2013_0_1_flair.nii.gz    # FLAIR
    └── Brats18_2013_0_1_seg.nii.gz      # Segmentation labels
```

**Sequences**: T1, T1ce, T2, FLAIR  
**Classification Task**: Tumor grading (LGG vs HGG)  
**Segmentation Classes**: 3 (ET, TC, WT)  
**Classification Labels**: 
- 0 = LGG (Low-Grade Glioma, WHO grade II)
- 1 = HGG (High-Grade Glioma, WHO grade III-IV)

**Note**: This is the same anatomical region as 0BraTS but used for classification rather than segmentation.

#### 15. BreaDM (Breast Dynamic Malignancy)
```
15BreaDM/
└── BreaDM-Be-1801/
    ├── BreaDM-Be-1801_pre-dce.nii.gz    # Pre-contrast DCE
    └── BreaDM-Be-1801_pos-dce.nii.gz    # Post-contrast DCE
```

**Sequences**: Pre-DCE, Post-DCE (Dynamic contrast-enhanced MRI)  
**Classification Task**: Benign vs Malignant breast tumors  
**Classification Labels**: 0=benign, 1=malignant

## 📝 Data List Files

### Segmentation Lists

**Format**: `{DATASET_ID}{DATASET_NAME}/{PATIENT_ID}`

Example `seg_train.txt`:
```
0BraTS/BraTS_001
0BraTS/BraTS_002
1HNTS/HNTS_001
2NPC/NPC_001
3ISPY/ISPY_001
...
```

### Classification Lists

**Format**: `{DATASET_ID}{DATASET_NAME}/{PATIENT_ID} {LABEL}`

Example `cls_train.txt`:
```
2NPC/NPC_001 1              # T3-T4 (segmentation dataset)
3ISPY/ISPY_001 1            # Malignant (segmentation dataset)
8ProstateX/ProstateX_001 1  # MIBC (segmentation dataset)
11FedBca/center1_001 1      # Malignant (classification dataset)
12NPC/NPC_001 1             # T3-T4 (classification dataset)
13LLD/LLD_MR102385 0        # Benign (classification dataset)
14BraTS/Brats18_2013_0_1 1  # HGG (classification dataset)
15BreaDM/BreaDM-Be-1801 1   # Malignant (classification dataset)
...
```

**Classification Labels by Dataset**:

**Segmentation Datasets (0-10)**:
- **0-BraTS** (segmentation): Not used for classification in this context
- **1-HNTS**: Not used for classification
- **2-NPC** (segmentation+classification): 0=T1-T2, 1=T3-T4
- **3-ISPY** (segmentation+classification): 0=benign, 1=malignant
- **4-ATLAS**: Not used for classification
- **5-Colorectal**: Not used for classification
- **6-AMOS**: Not used for classification
- **7-BCa_seg**: Not used for classification
- **8-ProstateX** (segmentation+classification): 0=NMIBC, 1=MIBC
- **9-csPCa_seg**: Not used for classification

**Classification Datasets (11-15)**:
- **11-FedBca**: 0=benign, 1=malignant (breast)
- **12-NPC**: 0=T1-T2, 1=T3-T4 (NPC T-staging)
- **13-LLD**: 0=benign, 1=malignant (liver)
- **14-BraTS**: 0=LGG, 1=HGG (glioma grading)
- **15-BreaDM**: 0=benign, 1=malignant (breast)

## 🔧 Data Preprocessing

### Step 1: Resampling to 1mm³ Isotropic

```python
import SimpleITK as sitk

def resample_to_isotropic(image_path, output_path, target_spacing=(1.0, 1.0, 1.0)):
    """Resample image to isotropic 1mm³ spacing"""
    image = sitk.ReadImage(image_path)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)  # Use sitkNearestNeighbor for labels
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    
    resampled = resampler.Execute(image)
    sitk.WriteImage(resampled, output_path)
```

### Step 2: Intensity Normalization

```python
import numpy as np

def z_score_normalize(image):
    """Z-score normalization per sequence"""
    image = image.astype(np.float32)
    mask = image > 0  # Ignore background
    
    mean = np.mean(image[mask])
    std = np.std(image[mask])
    
    image[mask] = (image[mask] - mean) / (std + 1e-8)
    return image

def percentile_clip(image, lower=0.5, upper=99.5):
    """Clip intensity to percentile range"""
    mask = image > 0
    p_low = np.percentile(image[mask], lower)
    p_high = np.percentile(image[mask], upper)
    
    image = np.clip(image, p_low, p_high)
    return image
```

### Step 3: Center Cropping

```python
def center_crop_or_pad(image, target_shape=(128, 128, 128)):
    """Crop or pad image to target shape"""
    current_shape = image.shape
    
    # Calculate crop/pad
    starts = []
    ends = []
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            # Crop
            start = (current_shape[i] - target_shape[i]) // 2
            starts.append(start)
            ends.append(start + target_shape[i])
        else:
            # Will pad
            starts.append(0)
            ends.append(current_shape[i])
    
    # Crop
    cropped = image[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    # Pad if necessary
    pad_width = []
    for i in range(3):
        if current_shape[i] < target_shape[i]:
            total_pad = target_shape[i] - current_shape[i]
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    
    if any(p != (0, 0) for p in pad_width):
        cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    
    return cropped
```

### Complete Preprocessing Pipeline

```python
def preprocess_case(input_dir, output_dir, patient_id, sequences):
    """
    Complete preprocessing pipeline for one patient
    
    Args:
        input_dir: Path to raw data directory
        output_dir: Path to save preprocessed data
        patient_id: Patient identifier
        sequences: List of sequence names (e.g., ['t1', 't2', 'seg'])
    """
    import os
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    for seq in sequences:
        input_path = os.path.join(input_dir, f"{patient_id}_{seq}.nii.gz")
        output_path = os.path.join(output_dir, f"{patient_id}_{seq}.nii.gz")
        
        # Read
        image = sitk.ReadImage(input_path)
        
        # Step 1: Resample to 1mm³
        resampled = resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0))
        
        # Convert to numpy
        array = sitk.GetArrayFromImage(resampled)
        
        # Step 2: Intensity normalization (skip for labels)
        if seq != 'seg':
            array = percentile_clip(array, lower=0.5, upper=99.5)
            array = z_score_normalize(array)
        
        # Step 3: Crop/pad to target size
        array = center_crop_or_pad(array, target_shape=(128, 128, 128))
        
        # Convert back to SimpleITK and save
        output_image = sitk.GetImageFromArray(array)
        output_image.CopyInformation(resampled)
        sitk.WriteImage(output_image, output_path)
        
        print(f"Processed: {output_path}")
```

## 📋 Data Augmentation

Training uses the following augmentations (implemented in `dataset/aug_seg.py` and `dataset/aug_cls.py`):

### Spatial Augmentation
- **Random rotation**: ±15 degrees
- **Random scaling**: 0.9-1.1
- **Random flip**: horizontal/vertical
- **Random elastic deformation**

### Intensity Augmentation
- **Random brightness**: ±0.1
- **Random contrast**: 0.9-1.1
- **Gaussian noise**: σ=0.01
- **Gaussian blur**: σ=0.5-1.5

## 🚀 Quick Start Example

### 1. Prepare One Dataset (BraTS)

```python
import os
from glob import glob

# Paths
raw_data_dir = "/path/to/BraTS/raw"
output_dir = "./dataset/MR_Dataset/0BraTS"

# Process all patients
patient_dirs = glob(os.path.join(raw_data_dir, "*"))

for patient_dir in patient_dirs:
    patient_id = os.path.basename(patient_dir)
    sequences = ['t1', 't1ce', 't2', 'flair', 'seg']
    
    preprocess_case(
        input_dir=patient_dir,
        output_dir=os.path.join(output_dir, patient_id),
        patient_id=patient_id,
        sequences=sequences
    )
```

### 2. Create Data List

```python
# Generate segmentation list
with open("./dataset/segmentation/seg_train.txt", "w") as f:
    for patient_id in train_patient_ids:
        f.write(f"0BraTS/{patient_id}\n")
```

### 3. Verify Data

```python
from code.MOE_dataset_seg import UnisegDataset

dataset = UnisegDataset(
    data_dir="./dataset/",
    list_dir="./dataset/segmentation/seg_train.txt",
    split="train"
)

print(f"Dataset size: {len(dataset)}")

# Check one sample
sample = dataset[0]
print(f"Sequences: {[s.shape for s in sample[:8]]}")
print(f"Label shape: {sample[9].shape}")
```

## 🔍 Data Validation Checklist

Before training, verify:

- [ ] All sequences are resampled to 1mm³ isotropic
- [ ] Images are normalized (z-score)
- [ ] All images are same size (e.g., 128×128×128)
- [ ] Labels are in correct format (integer, not one-hot)
- [ ] Label values match expected classes
- [ ] File naming follows convention: `{patient_id}_{sequence}.nii.gz`
- [ ] Directory structure matches expected hierarchy
- [ ] Data list files are correctly formatted
- [ ] No missing files for required sequences

## 📊 Expected Data Statistics

| Dataset | Train | Val | Test | Image Size | Sequences | Classes |
|---------|-------|-----|------|------------|-----------|---------|
| BraTS | 200 | 50 | 100 | 128³ | 4 | 3 |
| HNTS | 150 | 30 | 50 | 128³ | 1 | 1 |
| NPC | 300 | 60 | 100 | 128³ | 3 | 2 |
| ISPY | 100 | 20 | 40 | 128³ | 2 | 1 |

*(Note: Actual numbers depend on your dataset)*

## 🛠️ Preprocessing Scripts

Preprocessing scripts for each dataset are located in:
```
code/dataset_conversion/
├── 1HNTS.py          # HNTS preprocessing
├── 2NPC.py           # NPC preprocessing
├── 3ISPY.py          # ISPY preprocessing
├── 6amos.py          # AMOS preprocessing
└── ...
```

These scripts handle:
- Format conversion (DICOM → NIfTI)
- Resampling
- Intensity normalization
- Cropping/padding

## 📞 Support

If you encounter issues with data preparation:
1. Check the example scripts in `code/dataset_conversion/`
2. Verify your data format matches the expected structure
3. Open an issue on GitHub with your data structure details

---

**Last Updated**: 2026-01-13

