# MRICombo: Multi-Task MRI Analysis Framework

## Overview

MRICombo is a unified multi-task deep learning framework for MRI image analysis, featuring a Mixture of Experts (MoE) architecture for both segmentation and classification tasks across multiple anatomical regions.

## Features

- **Multi-Task Learning**: Simultaneous segmentation and classification
- **Mixture of Experts (MoE)**: Dynamic expert routing based on region and task
- **Multi-Sequence Support**: Handles up to 8 different MRI sequences
- **Cross-Domain Generalization**: Trained on multiple anatomical regions

## Project Structure

```
MRICombo/
├── MOE_dataset_cls.py      # Dataset loader for classification
├── MOE_dataset_seg.py      # Dataset loader for segmentation
├── MOENet_train.py         # Training script
├── MOENet_test.py          # Testing/evaluation script
└── network/                # Neural network architectures
    ├── MRICombo.py         # Main MRICombo model
    ├── OmniNet.py          # OmniNet architecture
    ├── SwinUNETR.py        # Swin-UNETR implementation
    ├── Unet.py             # U-Net variants
    ├── conv_layers.py      # Convolutional layer components
    └── unet_utils.py       # U-Net utility functions
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- MONAI
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn

## Installation

```bash
git clone <your-repository-url>
cd MRICombo
pip install -r requirements.txt  # You may need to create this
```

## Usage

### Training

```bash
python MOENet_train.py --config <config_file>
```

### Testing

```bash
python MOENet_test.py --checkpoint <model_checkpoint>
```

## Model Architecture

The MRICombo framework consists of:

1. **Sequence Experts**: Individual experts for each MRI sequence
2. **Encoder Experts**: Mixture of encoder experts with dynamic routing
3. **Decoder Experts**: Task-specific decoder experts
4. **Gating Networks**: Region and task-aware expert selection

## Supported Tasks

### Segmentation
- Brain tumors (whole tumor, tumor core, enhancing tumor)
- Nasopharyngeal carcinoma
- Breast tumors
- Liver tumors
- Abdominal organs (11 organs)
- Pelvic malignancies

### Classification
- Brain tumor grading (HGG/LGG)
- Breast tumor classification (Benign/Malignant)
- Liver tumor classification (Benign/Malignant)
- Bladder cancer staging (MIBC/NMIBC)
- NPC T-staging (T1-T2/T3-T4)

## Citation

If you use this code in your research, please cite:

```
[Your paper citation will go here]
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This work was supported by [funding sources].

