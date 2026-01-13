# 🎯 Pre-trained Model Weights

This document describes how to obtain and use pre-trained MRICombo model weights.

## 📥 Available Checkpoints

We provide pre-trained model weights for the following configurations:

| Model | Datasets | Tasks | Download | Size | Performance |
|-------|----------|-------|----------|------|-------------|
| **MRICombo-Full** | All 10 datasets | Seg + Cls | [Link](#) | ~500MB | See paper Table 2 |
| **MRICombo-Seg** | Segmentation only | Seg only | [Link](#) | ~350MB | Dice: 0.85±0.03 |
| **MRICombo-Brain** | BraTS only | Seg + Cls | [Link](#) | ~400MB | Dice: 0.89±0.02 |

*Note: Model weights will be released upon paper acceptance. For early access, please contact: p2316955@mpu.edu.mo*

## 📂 Checkpoint Structure

Pre-trained checkpoints include:

```python
{
    'model_state_dict': {...},           # Model parameters
    'optimizer_state_dict': {...},       # Optimizer state (optional)
    'epoch': 400,                        # Training epoch
    'best_dice': 0.856,                  # Best validation Dice
    'config': {                          # Model configuration
        'num_encoder_experts': 4,
        'num_decoder_experts': 4,
        'base_ch': 32,
        'top_k_encoder': 2,
        'top_k_decoder': 2
    },
    'dataset_info': {                    # Dataset information
        'train_datasets': [...],
        'num_classes': 27,
        'sequence_info': {...}
    }
}
```

## 🚀 Using Pre-trained Weights

### 1. Download Checkpoint

```bash
# Download from release page
wget https://github.com/zhangzhuoneng/MRICombo/releases/download/v1.0/MRICombo_full.pth

# Or use provided script
python scripts/download_weights.py --model full --output ./snapshots/
```

### 2. Load for Inference

```python
import torch
from network.OmniNet import omni_seg_cls

# Initialize model
model = omni_seg_cls(
    img_size=(128, 128, 128),
    seg_in_channels=8,  # 8 MRI sequences
    out_channels=27,    # 27 segmentation classes
    cls_in_channels=8,
    cls_classes=5,      # 5 classification tasks
    backbone='MRICombo'
)

# Load checkpoint
checkpoint = torch.load('./snapshots/MRICombo_full.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best Dice: {checkpoint['best_dice']:.3f}")
```

### 3. Fine-tune on Your Data

```python
# Load pretrained weights
checkpoint = torch.load('./snapshots/MRICombo_full.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze encoder for fine-tuning (optional)
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False

# Train on your data
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# ... training loop ...
```

### 4. Resume Training

```python
# Load full checkpoint including optimizer
checkpoint = torch.load('./snapshots/checkpoint_epoch_200.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
for epoch in range(start_epoch, num_epochs):
    # ... training loop ...
    pass
```

## 🔧 Using in Testing Script

```bash
# Test with pretrained weights
python code/MOENet_test.py \
    --reload_path ./snapshots/MRICombo_full.pth \
    --data_dir ./dataset/ \
    --val_seg_list ./dataset/segmentation/seg_test.txt \
    --val_cls_list ./dataset/classification/cls_test.txt \
    --backbone_name MRICombo
```

## 📊 Expected Performance

When using our pre-trained weights on the test sets:

### Segmentation Performance (Dice Score)

| Dataset | Our Checkpoint | Expected Range |
|---------|----------------|----------------|
| BraTS | 0.891 | 0.88-0.90 |
| NPC | 0.823 | 0.81-0.83 |
| ISPY | 0.856 | 0.84-0.87 |
| Liver | 0.901 | 0.89-0.91 |
| Prostate | 0.834 | 0.82-0.85 |

### Classification Performance (AUC)

| Task | Our Checkpoint | Expected Range |
|------|----------------|----------------|
| Brain Tumor Grading | 0.923 | 0.91-0.93 |
| NPC T-staging | 0.887 | 0.87-0.90 |
| Breast Malignancy | 0.912 | 0.90-0.92 |

*Note: Performance may vary slightly (±0.01) due to randomness in inference augmentation.*

## 🔄 Model Versioning

We use semantic versioning for model weights:

- **v1.0.0**: Initial release (400 epochs, all datasets)
- **v1.1.0**: Improved with region-specific experts
- **v1.2.0**: Enhanced with diverse expert structures

## ⚠️ Important Notes

### Compatibility

- Checkpoint format is compatible with PyTorch >= 1.10
- Ensure your model architecture matches the checkpoint
- Use `strict=False` if loading partial weights:
  ```python
  model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  ```

### Memory Requirements

- Full model: ~2GB GPU memory for inference
- Batch size 1: ~4GB GPU memory
- Batch size 4: ~12GB GPU memory

### License

Pre-trained weights are released under the same Apache 2.0 license as the code. You are free to:
- Use for research and commercial purposes
- Modify and redistribute
- Include in derivative works

Attribution is appreciated but not required.

## 📝 Checkpoint Metadata

Each checkpoint includes metadata for reproducibility:

```python
# Access metadata
checkpoint = torch.load('MRICombo_full.pth')

print("Training info:")
print(f"  Datasets: {checkpoint['dataset_info']['train_datasets']}")
print(f"  Total epochs: {checkpoint['epoch']}")
print(f"  Best Dice: {checkpoint['best_dice']:.3f}")

print("\nModel config:")
for key, value in checkpoint['config'].items():
    print(f"  {key}: {value}")
```

## 🔍 Verifying Downloaded Weights

Verify integrity of downloaded weights:

```bash
# Check MD5 hash
md5sum MRICombo_full.pth
# Expected: 5f9c2a3b...

# Or use provided script
python scripts/verify_checkpoint.py --checkpoint MRICombo_full.pth
```

## 🆘 Troubleshooting

### Issue: "RuntimeError: size mismatch"

**Solution**: Architecture mismatch. Verify your model initialization matches the checkpoint:
```python
# Check checkpoint config
checkpoint = torch.load('model.pth')
print(checkpoint['config'])

# Match your model initialization
model = omni_seg_cls(**checkpoint['config'])
```

### Issue: "Missing keys" or "Unexpected keys"

**Solution**: Use `strict=False` or update your model architecture:
```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Issue: Poor performance after loading

**Checklist**:
- [ ] Data preprocessing matches training (resampling, normalization)
- [ ] Model is in eval mode: `model.eval()`
- [ ] Input sequences are in correct order
- [ ] Using same image size (128³)

## 📧 Request for Weights

If you need specific model weights not listed here, please:

1. Open a GitHub issue with:
   - Desired configuration
   - Intended use case
   - Expected timeline

2. Or email: p2316955@mpu.edu.mo with subject "MRICombo Weights Request"

We aim to respond within 5 business days.

## 🎓 Citation

If you use our pre-trained weights, please cite:

```bibtex
@article{zhang2024mricombo,
  title={MRICombo: A Universal MRI Analysis Framework with Mixture of Experts},
  author={Zhang, Zhuoneng and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

**Last Updated**: 2026-01-13

**Status**: Weights will be publicly released upon paper acceptance. Early access available upon request.

