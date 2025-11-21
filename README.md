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

## 🚀 Quick Start

### Training

```bash
# Single-GPU training
python MOENet_train.py \
    --data_dir /path/to/data \
    --batch_size 4 \
    --num_epochs 400 \
    --learning_rate 3e-5 \
    --save_dir ./checkpoints

# Multi-GPU training with DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    MOENet_train.py \
    --data_dir /path/to/data \
    --batch_size 16 \
    --distributed
```

### Testing/Evaluation

```bash
python MOENet_test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --test_dir /path/to/test_data \
    --output_dir ./results \
    --visualize
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

### v1.0.0 (2024)
- Initial release
- Support for 6 anatomical regions
- Multi-task learning with MoE architecture
- Apache 2.0 License

---

<div align="center">
Made with ❤️ by the MRICombo Team
</div>
