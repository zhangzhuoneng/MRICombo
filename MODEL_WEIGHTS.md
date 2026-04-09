# Pre-trained Model Weights

This document describes how to obtain and use the **MRICombo** checkpoint shipped with this repository.

## Available checkpoint

| File | Location | Format | Notes |
|------|----------|--------|--------|
| **Best_MRICombo** | `snapshots/Best_MRICombo.pth` | PyTorch `state_dict` | Trained **MRICombo** backbone inside `omni_seg_cls`; weights are stored via **Git LFS** |

There is **one** public weight file in-repo. Additional variants (e.g. task-only) are not distributed here; open an issue if you need something specific.

### Clone with LFS

The `.pth` file is large; after `git clone`, fetch LFS objects:

```bash
git lfs install
git clone https://github.com/zhangzhuoneng/MRICombo.git
cd MRICombo
git lfs pull
```

Verify the file is a real tensor checkpoint (not a tiny LFS pointer):

```bash
ls -lh snapshots/Best_MRICombo.pth
# Expect hundreds of MB on disk after `git lfs pull`
```

## What is inside the file

`MOENet_test.py` loads this path with `torch.load(..., weights_only=True)` and expects a **flat state dictionary** (key → tensor), as saved from **DistributedDataParallel** training: key names often start with `module.` and are **stripped** before `load_state_dict`:

```text
# Typical key pattern after DataParallel/DDP:
module.backbone.... / module.classifier_.... 
```

If your checkpoint was saved without the `module.` prefix, remove or adjust the stripping logic in the test script accordingly.

The checkpoint does **not** guarantee a nested bundle such as `{'model_state_dict': ..., 'epoch': ...}`—do not rely on that structure unless you produced it yourself during training (see `MOENet_train.py` for checkpoint formats when resuming).

## Evaluation (recommended)

From the repo root, using defaults that point at `snapshots/Best_MRICombo.pth`:

```bash
cd code
python MOENet_test.py \
    --reload_path ../snapshots/Best_MRICombo.pth \
    --reload_from_checkpoint True \
    --data_dir ../dataset/ \
    --val_seg_list ../dataset/segmentation/seg_test.txt \
    --val_cls_list ../dataset/classification/cls_test.txt \
    --backbone_name MRICombo
```

**Input / ROI geometry** must match training (defaults in the test script are **96×96×96**; see `--roi_x`, `--roi_y`, `--roi_z` and `--input_size`).

## Manual load (Python)

Align with `network/OmniNet.omni_seg_cls` and your task definitions (`seg_classes`, `cls_classes`, `in_channels`):

```python
import torch
from collections import OrderedDict
from network.OmniNet import omni_seg_cls

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = omni_seg_cls(
    img_size=(96, 96, 96),
    seg_in_channels=8,
    cls_in_channels=8,
    out_channels=27,   # match your experiment / template
    cls_classes=5,     # match your heads setup
    backbone="MRICombo",
).to(device)

sd = torch.load("snapshots/Best_MRICombo.pth", map_location=device, weights_only=True)

# Strip DDP 'module.' prefix if present (same as MOENet_test.py)
new_sd = OrderedDict()
for k, v in sd.items():
    name = k[7:] if k.startswith("module.") else k
    new_sd[name] = v

missing, unexpected = model.load_state_dict(new_sd, strict=False)
# Inspect missing/unexpected keys if you changed architecture hyperparameters
model.eval()
```

Use `strict=False` only when you intentionally changed heads or channels; otherwise prefer `strict=True` after confirming config matches training.

## Fine-tuning

1. Match **preprocessing** and **label/template** definitions with the training pipeline (`README.md`, `DATA_PREPARATION.md`).
2. Initialize `omni_seg_cls` with the same spatial size and channel settings as the checkpoint.
3. Load weights as above, then run `MOENet_train.py` with `--reload_from_checkpoint` and `--reload_path` set to a training-style checkpoint if you resume from a **dict** checkpoint (`checkpoint['model']`). The public `Best_MRICombo.pth` is optimized for **evaluation** loading in `MOENet_test.py`; for resume-from-training, use checkpoints saved by your own `MOENet_train.py` run.

## Expected performance

Reported metrics are in the **paper** (e.g. main tables). Numbers in this README are not duplicated here to avoid drift from the manuscript; after `git lfs pull`, use the same splits and preprocessing to reproduce.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `size mismatch` / many missing keys | `out_channels`, `cls_classes`, `img_size`, or backbone name do not match the checkpoint. |
| Tiny `.pth` file (~130 bytes) | Run `git lfs pull`; you only have the LFS pointer. |
| `module.` / no `module.` prefix | Adjust key renaming when loading (see manual load snippet). |

## License

Pre-trained weights are released under the same **Apache 2.0** license as the code (see `LICENSE`).

## Citation

If you use this repository or the provided weights, please cite the **paper** associated with this work (bib entry in the manuscript or publisher page).

---

**Last updated:** 2026-04-09
