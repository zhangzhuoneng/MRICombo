# MRICombo: Multi-Task MRI Analysis Framework

## Overview

MRICombo is a deep-learning-based framework for universal volume segmentation, grading-staging, and malignancy detection across heterogeneous MRI.

## Features

- **Multi-Task Learning**: Simultaneous segmentation and classification
- **Mixture of Experts (MoE)**: Dynamic expert routing based on region and task
- **Multi-Sequence Support**: Handles up to 9 different MRI sequences
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
- Brain tumor grading
- Breast tumor malignant detection
- Liver tumor malignant detection
- Bladder cancer staging
- NPC T-staging

## Citation

If you use this code in your research, please cite:

```
[Your paper citation will go here]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This work was supported by [funding sources].






