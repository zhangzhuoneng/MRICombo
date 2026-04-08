from typing import Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from .MRICombo import MRICombo
from typing import Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from .MRICombo import MRICombo


class omni_seg_cls(nn.Module):

    def __init__(self, img_size, seg_in_channels, cls_in_channels, out_channels,
                 backbone='swinunetr', cls_classes=2):
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'MRICombo':
            self.backbone = MRICombo(
                seg_in_ch=seg_in_channels, cls_in_ch=cls_in_channels,
                base_ch=32, num_classes=out_channels,
            )
            self.GAP = nn.Sequential(
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.cls_classes = cls_classes


            self.classifier_cen = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(512, 2))
            self.classifier_npc = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(512, 4))
            self.classifier_lld = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(512, 2))
            self.classifier_bra = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(512, 2))
            self.classifier_bre = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(512, 2))

            self.raw_weight = nn.Parameter(torch.tensor(0.0))


    def get_task_weights(self):
        return torch.sigmoid(self.raw_weight)

    def _get_head(self, key: str) -> nn.Module:
        """Return the classifier module for a given routing key."""
        _map = {
            'cen':    self.classifier_cen,
            'NPC':    self.classifier_npc,
            'LLD':    self.classifier_lld,
            'Bra':    self.classifier_bra,
            'Bre':    self.classifier_bre,
            'amb':    self.classifier_bre,
        }
        if key not in _map:
            raise ValueError(
                f"Unknown cls_head_key '{key}'. "
                f"Valid keys: {list(_map.keys())}")
        return _map[key]



    def forward(self, seg_inputs=None, cls_inputs=None,
                seg_sequence_code=None, cls_sequence_code=None,
                task=None, names=None, seg_region_ids=None, cls_region_ids=None,
                cls_head_keys=None,
                return_features=False, return_mae_recon=False,
                mae_mask_ratio=0.5, use_masked_encoder=False):
        """
        cls_head_keys : list[str] | None
            One routing key per sample in the batch.  When provided, each
            sample is forwarded through its designated head (supports mixing
            different tasks/class-counts in one batch).
            When None, routing falls back to the legacy names[:3] prefix.
        """

        seg_out = None
        cls_out = None
        seg_features = None
        cls_features = None
        mae_patches_pred = None
        mae_patches_orig = None
        mae_mask_patches = None

        seg_lb = None
        cls_lb = None

        if seg_inputs is not None:
            if return_features:
                seg_out, seg_features = self.backbone(*seg_inputs, seg_sequence_code, 'seg', seg_region_ids, return_features=True)
            elif return_mae_recon:

                seg_out, mae_recon, mae_target, mae_mask = self.backbone(
                    *seg_inputs, seg_sequence_code, 'seg', seg_region_ids,
                    return_mae_recon=True, mae_mask_ratio=mae_mask_ratio, use_masked_encoder=use_masked_encoder
                )
            else:
                seg_out = self.backbone(*seg_inputs, seg_sequence_code, 'seg', seg_region_ids)
            if hasattr(self.backbone, 'last_lb_loss'):
                seg_lb = self.backbone.last_lb_loss

        if cls_inputs is not None:
            if return_features:
                features, cls_features = self.backbone(*cls_inputs, cls_sequence_code, 'cls', cls_region_ids, return_features=True)
            else:
                features = self.backbone(*cls_inputs, cls_sequence_code, 'cls', cls_region_ids)
            if hasattr(self.backbone, 'last_lb_loss'):
                cls_lb = self.backbone.last_lb_loss
            cls_GAP = self.GAP(features)
            cls_fc  = cls_GAP.view(cls_GAP.size(0), -1)
            cls_out = []
            for i in range(cls_fc.size(0)):
                if cls_head_keys is not None:

                    head = self._get_head(cls_head_keys[i])
                else:

                    prefix = names[i][:3]
                    if prefix == 'cen':
                        head = self.classifier_cen
                    elif prefix == 'NPC':
                        head = self.classifier_npc
                    elif prefix == 'LLD':
                        head = self.classifier_lld
                    elif prefix == 'Bra':
                        head = self.classifier_bra
                    elif prefix in ('Bre', 'amb'):
                        head = self.classifier_bre
                    elif prefix == 'Ute':
                        head = self.classifier_ute
                    else:
                        raise ValueError(f"Unknown dataset prefix: {prefix}")
                out = head(cls_fc[i].unsqueeze(0))
                cls_out.append(out)

        if (seg_lb is not None) or (cls_lb is not None):
            total_lb = 0.0
            if seg_lb is not None: total_lb = total_lb + seg_lb
            if cls_lb is not None: total_lb = total_lb + cls_lb
            self.last_lb_loss = total_lb
        else:

            dev = None
            if seg_inputs is not None: dev = seg_inputs[0].device
            elif cls_inputs is not None: dev = cls_fc.device
            self.last_lb_loss = torch.tensor(0.0, device=dev)

        if return_features:
            return (seg_out, cls_out), (seg_features, cls_features)
        elif return_mae_recon:

            return seg_out, cls_out, mae_recon, mae_target, mae_mask
        return seg_out, cls_out
