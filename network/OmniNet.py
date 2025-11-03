from typing import Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from .SwinUNETR import SwinUNETR
from .Unet import UNet,UNet_cls, UNet_seg,UNet_MIL,UNet_seg1,UNet_amos,UNet_seg8,UNet_early_fusion,UNet_multi_task,UNet_brain
from .MRICombo import MRICombo
from .UNET import no_share_unet



class omni_seg_cls(nn.Module):
    def __init__(self, img_size, seg_in_channels,cls_in_channels, out_channels, backbone = 'swinunetr', cls_classes=2):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=seg_in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
            self.cls_classes=cls_classes
             # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(256,self.cls_classes),
                nn.Softmax(dim=1)
            )
        elif backbone == 'DeepFusionUniMRINet':
            # self.backbone = UNet(seg_in_ch=seg_in_channels,cls_in_ch=cls_in_channels,base_ch=32,num_classes=out_channels)
            # self.backbone =DeepFusionUniMRINet(seg_in_ch=seg_in_channels,cls_in_ch=cls_in_channels,base_ch=32,num_classes=out_channels)
            self.GAP = nn.Sequential(
                # nn.GroupNorm(16, 256),
                # nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                # nn.Conv3d(256,256, kernel_size=1, stride=1, padding=0)
            )

            self.cls_classes=cls_classes
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,128),
            #    torch.nn.Dropout(0.5), 
            #    nn.Linear(128,self.cls_classes))
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,self.cls_classes))
            # 分类器
            self.classifier_cen = nn.Sequential( 
               torch.nn.Dropout(0.2), 
               nn.Linear(512,2),
               
            )
            self.classifier_npc = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,4)
            )
            
            self.classifier_lld = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.classifier_bra = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            
            self.classifier_bre = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.raw_weight = nn.Parameter(torch.tensor(0.0))  # 可学习参数
        elif backbone == 'MRICombo':
            self.backbone = MRICombo(seg_in_ch=seg_in_channels,cls_in_ch=cls_in_channels,base_ch=32,num_classes=out_channels)
            self.GAP = nn.Sequential(
                # nn.GroupNorm(16, 256),
                # nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                # nn.Conv3d(256,256, kernel_size=1, stride=1, padding=0)
            )

            self.cls_classes=cls_classes
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,128),
            #    torch.nn.Dropout(0.5), 
            #    nn.Linear(128,self.cls_classes))
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,self.cls_classes))
            # 分类器
            self.classifier_cen = nn.Sequential( 
               torch.nn.Dropout(0.2), 
               nn.Linear(512,2),
               
            )
            self.classifier_npc = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,4)
            )
            
            self.classifier_lld = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.classifier_bra = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            
            self.classifier_bre = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.raw_weight = nn.Parameter(torch.tensor(0.0))  # 可学习参数
        elif backbone == 'unet':
            self.backbone = UNet(seg_in_ch=seg_in_channels,cls_in_ch=cls_in_channels,base_ch=32,num_classes=out_channels)
            # self.backbone = MOEUniMRINet(seg_in_ch=seg_in_channels,cls_in_ch=cls_in_channels,base_ch=32,num_classes=out_channels)
            self.GAP = nn.Sequential(
                # nn.GroupNorm(16, 256),
                # nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                # nn.Conv3d(256,256, kernel_size=1, stride=1, padding=0)
            )

            self.cls_classes=cls_classes
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,128),
            #    torch.nn.Dropout(0.5), 
            #    nn.Linear(128,self.cls_classes))
            # self.classifier = nn.Sequential(
            #    nn.Linear(256,self.cls_classes))
            # 分类器
            self.classifier_cen = nn.Sequential( 
               torch.nn.Dropout(0.2), 
               nn.Linear(512,2),
               
            )
            self.classifier_npc = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,4)
            )
            
            self.classifier_lld = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.classifier_bra = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            
            self.classifier_bre = nn.Sequential(torch.nn.Dropout(0.2), 
               nn.Linear(512,2)
            )
            self.raw_weight = nn.Parameter(torch.tensor(0.0))  # 可学习参数
    
    def get_task_weights(self):
        w = torch.sigmoid(self.raw_weight)  # 保证在(0,1)
        return w

    # def forward(self, seg_inputs=None, cls_inputs=None,
    #         seg_sequence_code=None, cls_sequence_code=None,
    #         task=None, names=None):

    #     seg_out = None
    #     cls_out = None

    #     if seg_inputs is not None:
    #         seg_out = self.backbone(*seg_inputs, seg_sequence_code, task='seg')

    #     if cls_inputs is not None:
    #         features = self.backbone(*cls_inputs, cls_sequence_code, task='cls')
    #         cls_GAP = self.GAP(features)
    #         cls_fc  = cls_GAP.view(cls_GAP.size(0), -1)
    #         cls_out = []
    #         for i in range(cls_fc.size(0)):
    #             prefix = names[i][:3]
    #             # print(i,prefix)
    #             if prefix == 'cen':
    #                 out = self.classifier_cen(cls_fc[i].unsqueeze(0))
    #             elif prefix == 'NPC':
    #                 out = self.classifier_npc(cls_fc[i].unsqueeze(0))
    #             elif prefix == 'LLD':
    #                 out = self.classifier_lld(cls_fc[i].unsqueeze(0))
    #             elif prefix == 'Bra':
    #                 out = self.classifier_bra(cls_fc[i].unsqueeze(0))
    #             elif prefix == 'Bre':
    #                 out = self.classifier_bre(cls_fc[i].unsqueeze(0))
    #             else:
    #                 raise ValueError(f"Unknown dataset prefix: {prefix}")
    #             cls_out.append(out)

    #     return seg_out, cls_out
  
    def forward(self, seg_inputs=None, cls_inputs=None,
            seg_sequence_code=None, cls_sequence_code=None,
            task=None, names=None,seg_region_ids=None,cls_region_ids=None):

        seg_out = None
        cls_out = None

        seg_lb = None
        cls_lb = None

        if seg_inputs is not None:
            seg_out = self.backbone(*seg_inputs, seg_sequence_code, 'seg',seg_region_ids)
            if hasattr(self.backbone, 'last_lb_loss'):
                seg_lb = self.backbone.last_lb_loss

        if cls_inputs is not None:
            features = self.backbone(*cls_inputs, cls_sequence_code, 'cls',cls_region_ids)
            if hasattr(self.backbone, 'last_lb_loss'):
                cls_lb = self.backbone.last_lb_loss
            cls_GAP = self.GAP(features)
            cls_fc  = cls_GAP.view(cls_GAP.size(0), -1)
            cls_out = []
            for i in range(cls_fc.size(0)):
                prefix = names[i][:3]
                # print(i,prefix)
                if prefix == 'cen':
                    out = self.classifier_cen(cls_fc[i].unsqueeze(0))
                elif prefix == 'NPC':
                    out = self.classifier_npc(cls_fc[i].unsqueeze(0))
                elif prefix == 'LLD':
                    out = self.classifier_lld(cls_fc[i].unsqueeze(0))
                elif prefix == 'Bra':
                    out = self.classifier_bra(cls_fc[i].unsqueeze(0))
                elif prefix == 'Bre' or prefix == 'amb':
                    out = self.classifier_bre(cls_fc[i].unsqueeze(0))
                else:
                    raise ValueError(f"Unknown dataset prefix: {prefix}")
                cls_out.append(out)

        # 合并 seg/cls 的负载均衡正则（不存在则按0处理）
        if (seg_lb is not None) or (cls_lb is not None):
            total_lb = 0.0
            if seg_lb is not None: total_lb = total_lb + seg_lb
            if cls_lb is not None: total_lb = total_lb + cls_lb
            self.last_lb_loss = total_lb
        else:
            # 确保有属性，便于训练端安全读取
            dev = None
            if seg_inputs is not None: dev = seg_inputs[0].device
            elif cls_inputs is not None: dev = cls_fc.device
            self.last_lb_loss = torch.tensor(0.0, device=dev)

        return seg_out, cls_out
