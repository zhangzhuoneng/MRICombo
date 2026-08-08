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

The framework builds patient-specific disease trajectories and trains a transformer-based model with region-aware and task-aware gating mechanisms for optimal expert selection. An optional **masked autoencoder (MAE) reconstruction branch** encourages robust fused representations when sequences are partially observed or noisy, complementing supervised segmentation and classification losses during early training.

## 🌟 Key Features

- **🔀 Multi-Task Learning**: Simultaneous segmentation and classification with shared representations
- **🎯 Mixture of Experts (MoE)**: Dynamic expert routing based on anatomical region and task type
- **🧠 Multi-Sequence Support**: Handles heterogeneous MRI sequences (up to 9 modalities)
- **🧩 Reconstruction (MAE) Module**: Input-level masking with a lightweight decoder to reconstruct fused early features from the encoder bottleneck (self-supervised auxiliary objective)
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
│       ├── cls_val.txt          # Validation list
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

1. **Resampling**: 1.2×1.5×1.5mm³ isotropic spacing
2. **Normalization**: Z-score normalization per sequence
3. **Size**: 96×96×96 voxels (center crop/pad)
4. **Format**: NIfTI (.nii.gz)

### Detailed Instructions

📖 **See [DATA_PREPARATION.md](DATA_PREPARATION.md)** for:
- Complete directory structure for each dataset
- Naming conventions and file formats
- Step-by-step preprocessing pipeline with code examples
- Data validation checklist
- Dataset-specific label encodings

## 🎯 Pre-trained Weights

We ship a pretrained checkpoint in the repository for direct evaluation and fine-tuning.

- **Default path**: `snapshots/Best_MRICombo.pth` (tracked via Git LFS; run `git lfs pull` after clone if needed)
- **Details**: [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) — loading, evaluation, and fine-tuning notes

For questions: open an issue or contact p2316955@mpu.edu.mo

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
training tips: "For the first 250 epochs, the model is primarily trained on the segmentation task. From epoch 250 to 400, it is jointly trained on both segmentation and classification tasks."
**Reconstruction (MAE) branch** (optional self-supervised auxiliary loss in `MOENet_train.py`):

| Argument | Default | Meaning |
|----------|---------|---------|
| `--use_mae` | `True` | Enable MAE reconstruction loss during training |
| `--mae_initial_weight` | `1.0` | Starting weight for the MAE term |
| `--mae_warmup_epochs` | `100` | MAE weight linearly decays to 0 over this many epochs |
| `--mae_mask_ratio` | `0.25` | Fraction of low-res patches masked per sequence (higher = more masking) |

Example (supervised-only training — set MAE loss weight to zero):

```bash
python MOENet_train.py ... --mae_initial_weight 0
```

Note: `--use_mae` is parsed as a boolean flag in the training script; using `--mae_initial_weight 0` reliably turns off the reconstruction term.

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
    img_size=(96, 96, 96),
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

# seg_output: (B, 27, 96, 96, 96)
# cls_output: (B, num_classes)
```

## 🏗️ Model Architecture

The MRICombo framework consists of five main components (four task heads plus an optional reconstruction path):

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

### 5. **Reconstruction Module (MAE-style auxiliary branch)**

This branch is **not** a separate standalone model: it plugs into the segmentation forward path when `return_mae_recon=True` (used inside `MOENet_train.py` when MAE is enabled).

- **Masking**: Patch-wise random masking is applied **independently per input sequence** on a low-resolution grid (factor 16), then upsampled to the full volume so each channel can be partially observed.
- **Encoder input**: The network runs on **masked** volumes; the **reconstruction target** is the fused early feature map computed from **unmasked** inputs (detached), so the objective encourages the bottleneck to recover information removed at the input.
- **MAE decoder**: A shallow **transposed-convolution stack** (`mae_decoder` in `network/MRICombo.py`) maps bottleneck features back to the spatial resolution of the fused feature map (four upsampling stages + a final 3×3×3 conv).
- **Loss**: **L1** between prediction and target, averaged over masked voxels (and channels), weighted by a **linear schedule** that starts at `--mae_initial_weight` and goes to zero after `--mae_warmup_epochs`.

At inference or standard testing, this branch is typically **off**; only segmentation and classification heads are used.

## 📊 Supported Tasks

### Segmentation Tasks

| Anatomical Region | Target Structures | Sequences | Classes |
|-------------------|-------------------|-----------|---------|
| **Brain** | Whole tumor, tumor core, enhancing tumor | T1, T1ce, T2, FLAIR | 3 |
| **Head and Neck** | Primary tumor (GTVnx), lymph nodes (GTVnd), Nasopharynx cancer | T1, T1c, T2 | 3 |
| **Breast** | Tumor segmentation | DCE | 2 |
| **Liver** | Tumor segmentation | T1c | 1 |
| **Abdomen** | 13 organs (liver, spleen, kidney, etc.) | T1 | 13 |
| **Pelvis** | Bladder tumor, prostate | T1, T2, DWI, ADC | 2-3 |

### Classification Tasks

| Task | Classes | Clinical Application |
|------|---------|---------------------|
| **Brain Tumor Grading** | HGG (high-grade) / LGG (low-grade) | Glioma grading |
| **Breast Tumor** | Benign / Malignant | Malignancy detection |
| **Liver Tumor** | Benign / Malignant | Malignancy detection |
| **Bladder Cancer Staging** | MIBC / NMIBC | Muscle-invasive detection |
| **NPC T-Staging** | T1-T2 / T3-T4 | Nasopharyngeal carcinoma staging |




### Data Preprocessing

1. **Resampling**: All images resampled to 1.2mm*1.5mm*1.5mm isotropic
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
✅ **Pre-trained weights** - `snapshots/Best_MRICombo.pth` (Git LFS)  
✅ **Reconstruction (MAE) training** - Documented above and implemented in `MOENet_train.py` / `network/MRICombo.py`  
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
