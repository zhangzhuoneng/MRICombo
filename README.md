# 🏥 MRICombo: a deep-learning-based framework for universal volume segmentation, grading-staging, and malignancy detection across heterogeneous MRI

<div align="center">

![python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![license](https://img.shields.io/badge/License-Apache%202.0-green.svg)

</div>

*A deep learning framework for universal volume segmentation, grading-staging, and malignancy detection across heterogeneous MRI sequences*

---

## 📋 Overview

**MRICombo** is a versatile multi-expert framework capable of robustly integrating multiple diagnostic tasks within a single architecture, even with incomplete or variable imaging protocols. A key insight underlying our approach is that different sequences and types of disease share common imaging characteristics and structural manifestations, enabling the framework to learn generalized representations across multiple diseases. Our framework leverages mixture of experts networks with adaptive weighted fusion and task-specific gating mechanisms to dynamically integrate information from available MRI sequences. 

- **Multiple anatomical regions**: Brain, liver, breast, nasopharynx, abdomen, pelvis
- **Multiple MRI sequences**: T1, T2, FLAIR, CT1, DWI, and more (up to 9 sequences)
- **Multiple tasks**: Organ/tumor segmentation, tumor grading, staging, malignancy detection

The framework builds patient-specific disease trajectories and trains a transformer-based model with region-aware and task-aware gating mechanisms for optimal expert selection.

## 🌟 Key Features

- **🔀 Multi-Task Learning**: Simultaneous segmentation and classification with shared representations
- **🎯 Mixture of Experts (MoE)**: Dynamic expert routing based on anatomical region and task type
- **🧠 Multi-Sequence Support**: Handles heterogeneous MRI sequences (up to 9 modalities)
- **🌍 Cross-Domain Generalization**: Trained on 6+ anatomical regions with varying sequence combinations
- **⚡ Efficient Training**: Task-specific expert selection reduces computational overhead
- **🔬 Clinical Applications**: Supports multiple cancer types and clinical decision-making tasks


## 🔧 Requirements

### System Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended
- 50GB+ free disk space for datasets

### Core Dependencies
- **Deep Learning**: PyTorch ≥1.10.0, MONAI ≥1.0.0
- **Medical Imaging**: SimpleITK, nibabel, scikit-image
- **Data Processing**: NumPy, Pandas, batchgenerators
- **Visualization**: matplotlib, seaborn, tensorboard
- **ML Tools**: scikit-learn, PCGrad

See [requirements.txt](requirements.txt) for complete list.

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/zhangzhuoneng/MRICombo.git
cd MRICombo
```

### 2. Create a virtual environment (recommended)
```bash
# Using conda
conda create -n mricombo python=3.8
conda activate mricombo
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "import torch; import monai; print('Installation successful!')"
```

## 📊 Data Preparation

Before running experiments, you need to prepare your datasets following our format.

### Quick Overview

1. **Organize your data** following the expected directory structure
2. **Preprocess** all images (resample, normalize, crop)
3. **Create data lists** for training/validation/testing

### Directory Structure

```
MRICombo/
├── dataset/
│   ├── MR_Dataset/              # Raw MRI data
│   │   ├── 0BraTS/              # Brain tumor dataset
│   │   │   └── BraTS_001/
│   │   │       ├── BraTS_001_t1.nii.gz
│   │   │       ├── BraTS_001_t1ce.nii.gz
│   │   │       ├── BraTS_001_t2.nii.gz
│   │   │       ├── BraTS_001_flair.nii.gz
│   │   │       └── BraTS_001_seg.nii.gz
│   │   ├── 1HNTS/               # Head-neck tumor
│   │   ├── 2NPC/                # Nasopharyngeal carcinoma
│   │   └── ...                  # Other datasets
│   │
│   ├── segmentation/
│   │   ├── seg_train.txt        # Training list
│   │   ├── seg_val.txt          # Validation list
│   │   └── seg_test.txt         # Testing list
│   │
│   └── classification/
│       ├── cls_train.txt        # Training list
│       └── cls_test.txt         # Testing list
│
└── snapshots/                   # Model checkpoints
```

### Data List Format

**Segmentation lists** (`seg_train.txt`):
```
0BraTS/BraTS_001
0BraTS/BraTS_002
1HNTS/HNTS_001
2NPC/NPC_001
...
```

**Classification lists** (`cls_train.txt`):
```
0BraTS/BraTS_001 1
0BraTS/BraTS_002 0
2NPC/NPC_001 1
...
```

### Preprocessing Requirements

All MRI data must be preprocessed with:

1. **Resampling**: 1mm³ isotropic spacing
2. **Normalization**: Z-score normalization per sequence
3. **Size**: 128×128×128 voxels (center crop/pad)
4. **Format**: NIfTI (.nii.gz)

### Detailed Instructions

📖 **See [DATA_PREPARATION.md](DATA_PREPARATION.md)** for:
- Complete directory structure for each dataset
- Naming conventions and file formats
- Step-by-step preprocessing pipeline with code examples
- Data validation checklist
- Dataset-specific label encodings

## 🎯 Pre-trained Weights

We provide pre-trained model weights for reproducibility.

📥 **See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)** for:
- Download links for pretrained checkpoints
- Instructions for loading and using weights
- Fine-tuning guide
- Expected performance metrics

**Note**: Model weights will be publicly released upon paper acceptance. For early access, contact: p2316955@mpu.edu.mo

## 🚀 Quick Start

### Training

```bash
cd code  # Navigate to code directory

# Single-GPU training
python MOENet_train.py \
    --data_dir ../dataset/ \
    --train_seg_list ../dataset/segmentation/seg_train.txt \
    --val_seg_list ../dataset/segmentation/seg_val.txt \
    --train_cls_list ../dataset/classification/cls_train.txt \
    --val_cls_list ../dataset/classification/cls_test.txt \
    --backbone_name MRICombo \
    --batch_size 4 \
    --num_epochs 400 \
    --learning_rate 3e-5

# Multi-GPU training with DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    MOENet_train.py \
    --data_dir ../dataset/ \
    --batch_size 16 \
    --distributed
```

### Testing/Evaluation

```bash
cd code  # Navigate to code directory

python MOENet_test.py \
    --reload_path ../snapshots/Best_MRICombo.pth \
    --data_dir ../dataset/ \
    --val_seg_list ../dataset/segmentation/seg_test.txt \
    --val_cls_list ../dataset/classification/cls_test.txt \
    --backbone_name MRICombo \
    --excel_dir csv/MRICombo_output
```

### Inference on New Data

```python
import torch
from network.OmniNet import omni_seg_cls

# Load model
model = omni_seg_cls(
    img_size=(128, 128, 128),
    seg_in_channels=8,
    out_channels=27,
    cls_in_channels=8,
    cls_classes=5,
    backbone='MRICombo'
)

checkpoint = torch.load('../snapshots/Best_MRICombo.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare your input (8 sequences)
# x1-x8: T1, T1ce, T2, FLAIR, DCE1, DCE2, ADC, DWI
inputs = [x1, x2, x3, x4, x5, x6, x7, x8]
sequence_code = torch.ones(8)  # All sequences available

# Inference
with torch.no_grad():
    seg_output, cls_output = model(inputs, sequence_code, task='seg')

# seg_output: (B, 27, 128, 128, 128)
# cls_output: (B, num_classes)
```

## 🏗️ Model Architecture

The MRICombo framework consists of four main components:

### 1. **Sequence Extraction Module**
- Individual sequence extraction modules for each MRI sequence (T1, T2, FLAIR, etc.)
- Extracts sequence-specific features before fusion
- Handles missing modalities gracefully

### 2. **Encoder Experts with MoE**
- Multiple encoder experts with different architectures
- **Gating Network**: Selects top-k experts based on region and task prompts
- **Dynamic Routing**: Weighted combination of expert outputs
- Shared across segmentation and classification tasks

### 3. **Decoder Experts**
- Task-specific decoder experts for segmentation
- Separate classification head with region-aware features
- Multi-scale feature aggregation

### 4. **Gating Mechanisms**
- **Region-Aware**: Adapts to different anatomical structures
- **Task-Aware**: Switches behavior for seg vs. cls tasks
- **Learnable**: Trained end-to-end with task losses

## 📊 Supported Tasks

### Segmentation Tasks

| Anatomical Region | Target Structures | Sequences | Classes |
|-------------------|-------------------|-----------|---------|
| **Brain** | Whole tumor, tumor core, enhancing tumor | T1, T1ce, T2, FLAIR | 3 |
| **Nasopharynx** | Primary tumor (GTVnx), lymph nodes (GTVnd) | T1, T2 | 2 |
| **Breast** | Tumor segmentation | T1, T2, DWI | 2 |
| **Liver** | Tumor segmentation | T1, T2, DWI | 2 |
| **Abdomen** | 11 organs (liver, spleen, kidney, etc.) | CT/T1 | 11 |
| **Pelvis** | Bladder tumor, prostate | T1, T2, DWI | 2-3 |

### Classification Tasks

| Task | Classes | Clinical Application |
|------|---------|---------------------|
| **Brain Tumor Grading** | HGG (high-grade) / LGG (low-grade) | Glioma grading |
| **Breast Tumor** | Benign / Malignant | Malignancy detection |
| **Liver Tumor** | Benign / Malignant | Malignancy detection |
| **Bladder Cancer Staging** | MIBC / NMIBC | Muscle-invasive detection |
| **NPC T-Staging** | T1-T2 / T3-T4 | Nasopharyngeal carcinoma staging |




### Data Preprocessing

1. **Resampling**: All images resampled to 1mm³ isotropic
2. **Intensity Normalization**: Z-score normalization per sequence
3. **Cropping**: Region-of-interest extraction
4. **Augmentation**: Spatial transforms, intensity shifts, noise injection


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: p2316955@mpu.edu.mo

## 🙏 Acknowledgments

This work was supported by:
- [Your funding sources]
- Built upon [MONAI](https://monai.io/), [PyTorch](https://pytorch.org/)

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### v1.0.0 (2026-01-13)
- Initial release
- Support for 10 anatomical regions
- Multi-task learning with MoE architecture
- Apache 2.0 License
- Comprehensive documentation

## 📚 Documentation Index

- **[README.md](README.md)** (this file) - Project overview and quick start
- **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - Detailed data preparation guide
  - Expected directory structure
  - Dataset-specific formats and naming conventions
  - Preprocessing pipeline with code examples
  - Data validation checklist
- **[MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)** - Pre-trained model weights
  - Download links and usage instructions
  - Fine-tuning guide
  - Expected performance metrics
- **[STRUCTURE.md](STRUCTURE.md)** - Project structure overview
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[LICENSE](LICENSE)** - Apache 2.0 License

## 🔬 Reproducibility

We are committed to reproducibility and provide:

✅ **Complete source code** - All model architectures and training scripts  
✅ **Detailed data preparation** - Step-by-step preprocessing instructions  
✅ **Pre-trained weights** - Model checkpoints (released upon acceptance)  
✅ **Preprocessing scripts** - Located in `code/dataset_conversion/`  
✅ **Configuration files** - All hyperparameters documented  
✅ **Expected results** - Performance metrics for verification  

### Reproducing Paper Results

1. **Setup environment**: Follow [Installation](#%EF%B8%8F-installation)
2. **Prepare data**: Follow [DATA_PREPARATION.md](DATA_PREPARATION.md)
3. **Download weights**: Follow [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)
4. **Run evaluation**: Use testing script with provided checkpoints

Expected results should match Table 2 in the paper within ±0.01 Dice/AUC due to randomness.

### For Questions

- **Data preparation**: See [DATA_PREPARATION.md](DATA_PREPARATION.md) or open an issue
- **Model usage**: See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)
- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Direct contact**: p2316955@mpu.edu.mo

---

<div align="center">
Made with ❤️ by the MRICombo Team
</div>
