# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
# from .utils import get_block, get_norm
from .conv_layers import BasicBlock, Bottleneck, SingleConv, MBConv, FusedMBConv,ConvNormAct
import pdb

def get_block(name):
    block_map = { 
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'MBConv': MBConv,
        'FusedMBConv': FusedMBConv,
        'ConvNeXtBlock': ConvNormAct
    }   
    return block_map[name]

def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d
                }

    return norm_map[name]
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, num_classes):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        # 三维特征提取部分
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(20, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # 全连接部分
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4 * 4, self.M),  # 修改为适应三维数据
            nn.ReLU(),
        )

        # 注意力模块
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        # 特征提取
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4 * 4)
        H = self.feature_extractor_part2(H)

        # 注意力机制
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        # 加权特征
        Z = torch.mm(A, H)

        # 分类
        Y_prob = self.classifier(Z)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return Y_prob, Y_hat, A

    def calculate_objective(self, X, Y):
        Y_prob, _, A = self.forward(X)
        Y = Y.long()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(Y_prob, Y)

        return loss, A

class UNet_MIL(nn.Module):
    def __init__(self,seg_in_ch,cls_in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='in'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        
        self.inc_seg = inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.inc_cls = inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
            
        self.down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        
        
        # 三维特征提取部分
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(7, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(20, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
       
        self.up1 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)


    def forward(self, x,task): 
        if task=="seg":
            x1 = self.inc_seg(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            out = self.up1(x5, x4) 
            out = self.up2(out, x3) 
            out = self.up3(out, x2)
            out = self.up4(out, x1)
            out = self.outc(out)
        elif task=="cls":
            x  = x.squeeze(0)
            x11 = self.feature_extractor_part1(x)
            # print(x11.shape)
        else:
            print("no task error")
      
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape) torch.Size([8, 32, 96, 96, 96]) torch.Size([8, 32, 48, 48, 48]) torch.Size([8, 64, 24, 24, 24]) torch.Size([8, 128, 12, 12, 12]) torch.Size([8, 256, 6, 6, 6])
        
        # print(out.shape)

        return x11
class UNet(nn.Module):
    def __init__(self,seg_in_ch,cls_in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        
        self.cls_inconv_t1c=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t1n=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t2f=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t2w=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        # self.cls_inconv_pre_dce=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        # self.cls_inconv_post_dce=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_CA=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_CV=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
       
        self.seg_inconv_t1c=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t1n=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t2f=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t2w=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_pre_dce=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_pos_dce=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_adc=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_dwi=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        
        # self.inc_seg = inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        # self.inc_cls = inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
            
        self.seg_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.seg_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.seg_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.seg_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        
        # self.seg_down1_t1c = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t1n = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t2f = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t2w = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_dce = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_adc = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_dwi = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
        # self.cls_down1_t1c = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t1n = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t2f = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t2w = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_dce = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_CA = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_CV = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
        
        self.cls_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.cls_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.cls_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.cls_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

        self.fc1=nn.Linear(8,8)
        self.fc2=nn.Linear(6,6)
        self.epision = 1e-8
    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8,sequence_code,task): 
        if task=="seg":
            # x1 = self.inc_seg(x)
            
            x1_t1c = self.seg_inconv_t1c(x1)
            x1_t1n = self.seg_inconv_t1n(x2)
            x1_t2f = self.seg_inconv_t2f(x3)
            x1_t2w = self.seg_inconv_t2w(x4)
            x1_post_dce = self.seg_inconv_pos_dce(x5)
            x1_pre_dce = self.seg_inconv_pre_dce(x6)
            x1_adc = self.seg_inconv_adc(x7)
            x1_dwi = self.seg_inconv_dwi(x8)
            
            sequence_code = sequence_code.float()
            # weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
            # weight = weight/weight.sum(dim=1, keepdim=True)
            softmax_output = torch.softmax(self.fc1(sequence_code), dim=-1) + self.epision
            weight = sequence_code * softmax_output
            weight = weight / weight.sum(dim=1, keepdim=True)  # 归一化
                        # print(weight)
            aggrevate_feature = (
                x1_t1c * weight[:, 0:1, None, None, None] +
                x1_t1n * weight[:, 1:2, None, None, None] +
                x1_t2f * weight[:, 2:3, None, None, None] +
                x1_t2w * weight[:, 3:4, None, None, None] +
                x1_post_dce * weight[:, 4:5, None, None, None] +
                x1_pre_dce * weight[:, 5:6, None, None, None] +
                x1_adc * weight[:, 6:7, None, None, None] +
                x1_dwi * weight[:, 7:8, None, None, None] 
            )
            
            
            x2 = self.seg_down1(aggrevate_feature)
            x3 = self.seg_down2(x2)
            x4 = self.seg_down3(x3)
            x5 = self.seg_down4(x4)
            out = self.up1(x5, x4) 
            out = self.up2(out, x3) 
            out = self.up3(out, x2)
            out = self.up4(out, aggrevate_feature)
            out = self.outc(out)
            # print(x5.shape)
            return out
          
        elif task=="cls":
            # x1 = self.inc_cls(x)
            x1_t1c = self.cls_inconv_t1c(x1)
            x1_t1n = self.cls_inconv_t1n(x2)
            x1_t2f = self.cls_inconv_t2f(x3)
            x1_t2w = self.cls_inconv_t2w(x4)
            # x1_pre_dce = self.cls_inconv_pre_dce(x5)
            # x1_post_dce = self.cls_inconv_post_dce(x6)
            x1_lca = self.cls_inconv_CA(x5)
            x1_lcv = self.cls_inconv_CV(x6)
            
            # x1_t1c = self.cls_down1_t1c(x1_t1c)
            # x1_t1n = self.cls_down1_t1n(x1_t1n)
            # x1_t2f = self.cls_down1_t2f(x1_t2f)
            # x1_t2w = self.cls_down1_t2w(x1_t2w)
            # x1_dce = self.cls_down1_dce(x1_dce)
            # x1_lca = self.cls_down1_CA(x1_lca)
            # x1_lcv = self.cls_down1_CV(x1_lcv)
            
            
            sequence_code = sequence_code.float()
            # weight = sequence_code*torch.softmax(self.fc2(sequence_code),dim=-1)
            # weight = weight/weight.sum(dim=1, keepdim=True)
            softmax_output = torch.softmax(self.fc2(sequence_code), dim=-1) + self.epision
            weight = sequence_code * softmax_output
            weight = weight / weight.sum(dim=1, keepdim=True)  # 归一化
            
            # (B, C, H, W, D): 各通道加权求和
            aggrevate_feature = (
            x1_t1c * sequence_code[:, 0:1, None, None, None] +
            x1_t1n * sequence_code[:, 1:2, None, None, None] +
            x1_t2f * sequence_code[:, 2:3, None, None, None] +
            x1_t2w * sequence_code[:, 3:4, None, None, None] +
            # x1_pre_dce * sequence_code[:, 4:5, None, None, None] +
            # x1_post_dce * sequence_code[:, 5:6, None, None, None] +
            # x1_lca * sequence_code[:, 6:7, None, None, None] +
            # x1_lcv * sequence_code[:, 7:8, None, None, None]
            x1_lca * sequence_code[:, 4:5, None, None, None] +
            x1_lcv * sequence_code[:, 5:6, None, None, None]
             )
            # #非0模态的数量
            # N = mask_code.sum(dim=1, keepdim=True)
            # # print(mask_code,mask_code.sum(dim=1, keepdim=True))
            # # 平均特征
            # avg_feature = aggrevate_feature / N[..., None, None, None]
            x2 = self.cls_down1(aggrevate_feature)
            x3 = self.cls_down2(x2)
            x4 = self.cls_down3(x3)
            x5 = self.seg_down4(x4)
            # print(x5.shape)
            return x5
        else:
            print("no task error")
        
class UNet_multi_task(nn.Module):
    def __init__(self,seg_in_ch,cls_in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        
        self.cls_inconv_t1c=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t1n=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t2f=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_t2w=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_pre_dce=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_post_dce=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_CA=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_inconv_CV=inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
       
        self.seg_inconv_t1c=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t1n=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t2f=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_t2w=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_pre_dce=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_pos_dce=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_adc=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_inconv_dwi=inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        
        # self.inc_seg = inconv(seg_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        # self.inc_cls = inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
            
        self.seg_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.seg_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.seg_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.seg_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        
        # self.seg_down1_t1c = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t1n = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t2f = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_t2w = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_dce = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_adc = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.seg_down1_dwi = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
        # self.cls_down1_t1c = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t1n = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t2f = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_t2w = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_dce = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_CA = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.cls_down1_CV = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
        
        self.cls_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.cls_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.cls_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.cls_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

        self.fc1=nn.Linear(8,8)
        self.fc2=nn.Linear(8,8)
        self.epision = 1e-8
    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8,sequence_code,task): 
        if task=="seg":
            # x1 = self.inc_seg(x)
            
            x1_t1c = self.seg_inconv_t1c(x1)
            x1_t1n = self.seg_inconv_t1n(x2)
            x1_t2f = self.seg_inconv_t2f(x3)
            x1_t2w = self.seg_inconv_t2w(x4)
            x1_post_dce = self.seg_inconv_pos_dce(x5)
            x1_pre_dce = self.seg_inconv_pre_dce(x6)
            x1_adc = self.seg_inconv_adc(x7)
            x1_dwi = self.seg_inconv_dwi(x8)
            
            sequence_code = sequence_code.float()
            # weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
            # weight = weight/weight.sum(dim=1, keepdim=True)
            softmax_output = torch.softmax(self.fc1(sequence_code), dim=-1) + self.epision
            weight = sequence_code * softmax_output
            weight = weight / weight.sum(dim=1, keepdim=True)  # 归一化
                        # print(weight)
            aggrevate_feature = (
                x1_t1c * weight[:, 0:1, None, None, None] +
                x1_t1n * weight[:, 1:2, None, None, None] +
                x1_t2f * weight[:, 2:3, None, None, None] +
                x1_t2w * weight[:, 3:4, None, None, None] +
                x1_post_dce * weight[:, 4:5, None, None, None] +
                x1_pre_dce * weight[:, 5:6, None, None, None] +
                x1_adc * weight[:, 6:7, None, None, None] +
                x1_dwi * weight[:, 7:8, None, None, None] 
            )
            
            
            x2 = self.seg_down1(aggrevate_feature)
            x3 = self.seg_down2(x2)
            x4 = self.seg_down3(x3)
            x5 = self.seg_down4(x4)
            out = self.up1(x5, x4) 
            out = self.up2(out, x3) 
            out = self.up3(out, x2)
            out = self.up4(out, aggrevate_feature)
            out = self.outc(out)
            # print(x5.shape)
            return out
          
        elif task=="cls":
            # x1 = self.inc_cls(x)
            x1_t1c = self.cls_inconv_t1c(x1)
            x1_t1n = self.cls_inconv_t1n(x2)
            x1_t2f = self.cls_inconv_t2f(x3)
            x1_t2w = self.cls_inconv_t2w(x4)
            x1_pre_dce = self.cls_inconv_pre_dce(x5)
            x1_post_dce = self.cls_inconv_post_dce(x6)
            x1_lca = self.cls_inconv_CA(x7)
            x1_lcv = self.cls_inconv_CV(x8)
            
            # x1_t1c = self.cls_down1_t1c(x1_t1c)
            # x1_t1n = self.cls_down1_t1n(x1_t1n)
            # x1_t2f = self.cls_down1_t2f(x1_t2f)
            # x1_t2w = self.cls_down1_t2w(x1_t2w)
            # x1_dce = self.cls_down1_dce(x1_dce)
            # x1_lca = self.cls_down1_CA(x1_lca)
            # x1_lcv = self.cls_down1_CV(x1_lcv)
            
            
            sequence_code = sequence_code.float()
            # weight = sequence_code*torch.softmax(self.fc2(sequence_code),dim=-1)
            # weight = weight/weight.sum(dim=1, keepdim=True)
            softmax_output = torch.softmax(self.fc2(sequence_code), dim=-1) + self.epision
            weight = sequence_code * softmax_output
            weight = weight / weight.sum(dim=1, keepdim=True)  # 归一化
            
            # (B, C, H, W, D): 各通道加权求和
            aggrevate_feature = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_pre_dce * weight[:, 4:5, None, None, None] +
            x1_post_dce * weight[:, 5:6, None, None, None] +
            x1_lca * weight[:, 6:7, None, None, None] +
            x1_lcv * weight[:, 7:8, None, None, None]
            
             )
            # #非0模态的数量
            # N = mask_code.sum(dim=1, keepdim=True)
            # # print(mask_code,mask_code.sum(dim=1, keepdim=True))
            # # 平均特征
            # avg_feature = aggrevate_feature / N[..., None, None, None]
            x2 = self.cls_down1(aggrevate_feature)
            x3 = self.cls_down2(x2)
            x4 = self.cls_down3(x3)
            x5 = self.seg_down4(x4)
            # print(x5.shape)
            return x5
        else:
            print("no task error")
class UNet_early_fusion(nn.Module):
    def __init__(self,seg_in_ch,cls_in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='in'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        
        self.seg_inconv =inconv(seg_in_ch, base_ch,block=block, kernel_size=kernel_size[0], norm=norm)
        self.seg_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.seg_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.seg_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.seg_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        
        
        
        self.cls_inconv = inconv(cls_in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.cls_down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.cls_down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.cls_down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.cls_down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

      
    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8,sequence_code,task): 
        if task=="seg":
            # x1 = self.inc_seg(x)
            
            aggrevate_feature = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim=1)
            # print('seg',aggrevate_feature.shape)
            aggrevate_feature = self.seg_inconv(aggrevate_feature)
            # print('seg1',aggrevate_feature.shape)
            x2 = self.seg_down1(aggrevate_feature)
            x3 = self.seg_down2(x2)
            x4 = self.seg_down3(x3)
            x5 = self.seg_down4(x4)
            out = self.up1(x5, x4) 
            out = self.up2(out, x3) 
            out = self.up3(out, x2)
            out = self.up4(out, aggrevate_feature)
            out = self.outc(out)
            # print(x5.shape)
            return out
          
        elif task=="cls":
            # x1 = self.inc_cls(x)
            aggrevate_feature = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim=1)
            # print('cls',aggrevate_feature.shape)
            aggrevate_feature = self.cls_inconv(aggrevate_feature)
            # print('cls1',aggrevate_feature.shape)
            x2 = self.cls_down1(aggrevate_feature)
            x3 = self.cls_down2(x2)
            x4 = self.cls_down3(x3)
            x5 = self.cls_down4(x4)
            # print(x5.shape)
            return x5
        else:
            print("no task error")
class UNet_cls(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='in'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_pr_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_po_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_CA=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_CV=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
            
            
        # self.down1_t1c = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_t1n = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_t2f = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_t2w = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_dce = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_CA = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down1_CV = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
        # self.down2_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_CA = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down2_CV = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        
        self.down1 = down_block(base_ch, base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

        self.fc1=nn.Linear(8,8)
    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8,sequence_code): 
        
        x1_t1c = self.inconv_t1c(x1)
        x1_t1n = self.inconv_t1n(x2)
        x1_t2f = self.inconv_t2f(x3)
        x1_t2w = self.inconv_t2w(x4)
        x1_pr_dce = self.inconv_pr_dce(x5)
        x1_po_dce = self.inconv_po_dce(x6)
        x1_lca = self.inconv_CA(x7)
        x1_lcv = self.inconv_CV(x8)
        
        sequence_code = sequence_code.float()
        weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
        weight = weight/weight.sum(dim=1, keepdim=True)
        
        # x1_t1c = self.down1_t1c(x1_t1c)
        # x1_t1n = self.down1_t1n(x1_t1n)
        # x1_t2f = self.down1_t2f(x1_t2f)
        # x1_t2w = self.down1_t2w(x1_t2w)
        # # x1_dce = self.down1_dce(x1_dce)
        # x1_lca = self.down1_CA(x1_lca)
        # x1_lcv = self.down1_CV(x1_lcv)
        
        # x1_t1c = self.down2_t1c(x1_t1c)
        # x1_t1n = self.down2_t1n(x1_t1n)
        # x1_t2f = self.down2_t2f(x1_t2f)
        # x1_t2w = self.down2_t2w(x1_t2w)
        # # x1_dce = self.down2_dce(x1_dce)
        # x1_lca = self.down2_CA(x1_lca)
        # x1_lcv = self.down2_CV(x1_lcv)
        
       # (B, C, H, W, D): 各通道加权求和
        aggrevate_feature = (
            x1_t1c *sequence_code[:, 0:1, None, None, None] +
            x1_t1n *sequence_code[:, 1:2, None, None, None] +
            x1_t2f *sequence_code[:, 2:3, None, None, None] +
            x1_t2w *sequence_code[:, 3:4, None, None, None] +
            x1_pr_dce*sequence_code[:, 4:5, None, None, None] +
            x1_po_dce*sequence_code[:, 5:6, None, None, None] +
            x1_lca *sequence_code[:, 6:7, None, None, None] +
            x1_lcv *sequence_code[:, 7:8, None, None, None]
        )
        #非0模态的数量
        # N = mask_code.sum(dim=1, keepdim=True)
        # # print(mask_code,mask_code.sum(dim=1, keepdim=True))
        # # 平均特征
        # avg_feature = aggrevate_feature / N[..., None, None, None]
        
        x2 = self.down1(aggrevate_feature)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        out = self.down4(x4)
        return out



class UNet_seg(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_adc=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dwi=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_adc = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dwi = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        self.down2_t1c = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_t1n = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_t2f = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_t2w = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_dce = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_adc = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down2_dwi = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)

        self.down3_t1c = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_t1n = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_t2f = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_t2w = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_dce = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_adc = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down3_dwi = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)


        self.down4_t1c = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_t1n = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_t2f = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_t2w = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_dce = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_adc = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.down4_dwi = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        # self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        # self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        # self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        self.fc1=nn.Linear(7,7)
        self.fc2=nn.Linear(7,7)

    def forward(self, x1_t1c,x2_t1n,x3_t2f,x4_t2w,x5_dce,x6_adc,x7_dwi,sequence_code): #mask_code (B,7)
        
        x1_t1c = self.inconv_t1c(x1_t1c)
        x1_t1n = self.inconv_t1n(x2_t1n)
        x1_t2f = self.inconv_t2f(x3_t2f)
        x1_t2w = self.inconv_t2w(x4_t2w)
        x1_dce = self.inconv_dce(x5_dce)
        x1_adc = self.inconv_adc(x6_adc)
        x1_dwi = self.inconv_dwi(x7_dwi)
        
        sequence_code = sequence_code.float()
        weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
        weight = weight/weight.sum(dim=1, keepdim=True)

        aggrevate_feature_1 = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_dce * weight[:, 4:5, None, None, None] +
            x1_adc * weight[:, 5:6, None, None, None] +
            x1_dwi * weight[:, 6:7, None, None, None]
        )

        x1_t1c = self.down1_t1c(x1_t1c)
        x1_t1n = self.down1_t1n(x1_t1n)
        x1_t2f = self.down1_t2f(x1_t2f)
        x1_t2w = self.down1_t2w(x1_t2w)
        x1_dce = self.down1_dce(x1_dce)
        x1_adc = self.down1_adc(x1_adc)
        x1_dwi = self.down1_dwi(x1_dwi)
        
   
        aggrevate_feature_2 = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_dce * weight[:, 4:5, None, None, None] +
            x1_adc * weight[:, 5:6, None, None, None] +
            x1_dwi * weight[:, 6:7, None, None, None]
        )

        x1_t1c = self.down2_t1c(x1_t1c)
        x1_t1n = self.down2_t1n(x1_t1n)
        x1_t2f = self.down2_t2f(x1_t2f)
        x1_t2w = self.down2_t2w(x1_t2w)
        x1_dce = self.down2_dce(x1_dce)
        x1_adc = self.down2_adc(x1_adc)
        x1_dwi = self.down2_dwi(x1_dwi)
        
   
        aggrevate_feature_3 = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_dce * weight[:, 4:5, None, None, None] +
            x1_adc * weight[:, 5:6, None, None, None] +
            x1_dwi * weight[:, 6:7, None, None, None]
        )

        x1_t1c = self.down3_t1c(x1_t1c)
        x1_t1n = self.down3_t1n(x1_t1n)
        x1_t2f = self.down3_t2f(x1_t2f)
        x1_t2w = self.down3_t2w(x1_t2w)
        x1_dce = self.down3_dce(x1_dce)
        x1_adc = self.down3_adc(x1_adc)
        x1_dwi = self.down3_dwi(x1_dwi)
        
   
        aggrevate_feature_4 = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_dce * weight[:, 4:5, None, None, None] +
            x1_adc * weight[:, 5:6, None, None, None] +
            x1_dwi * weight[:, 6:7, None, None, None]
        )


        x1_t1c = self.down4_t1c(x1_t1c)
        x1_t1n = self.down4_t1n(x1_t1n)
        x1_t2f = self.down4_t2f(x1_t2f)
        x1_t2w = self.down4_t2w(x1_t2w)
        x1_dce = self.down4_dce(x1_dce)
        x1_adc = self.down4_adc(x1_adc)
        x1_dwi = self.down4_dwi(x1_dwi)
        
   
        aggrevate_feature_5 = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_dce * weight[:, 4:5, None, None, None] +
            x1_adc * weight[:, 5:6, None, None, None] +
            x1_dwi * weight[:, 6:7, None, None, None]
        )
     
        out = self.up1(aggrevate_feature_5, aggrevate_feature_4) 
        out = self.up2(out, aggrevate_feature_3) 
        out = self.up3(out,aggrevate_feature_2)
        out = self.up4(out, aggrevate_feature_1)
        out = self.outc(out)
        # print(out.shape)
        return  out
    
    
class UNet_seg1(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_adc=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dwi=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_adc = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dwi = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
            
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        self.fc1=nn.Linear(6,6)
        self.fc2=nn.Linear(6,6)

    def forward(self, x1_t1c,x2_t1n,x3_t2f,x4_t2w,x5_adc,x6_dwi,sequence_code): #mask_code (B,7)
        
        x1_t1c = self.inconv_t1c(x1_t1c)
        x1_t1n = self.inconv_t1n(x2_t1n)
        x1_t2f = self.inconv_t2f(x3_t2f)
        x1_t2w = self.inconv_t2w(x4_t2w)
        x1_adc = self.inconv_adc(x5_adc)
        x1_dwi = self.inconv_dwi(x6_dwi)
        # x1_dwi = self.inconv_dwi(x7_dwi)
        
        sequence_code = sequence_code.float()
        weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
        weight = weight/weight.sum(dim=1, keepdim=True)
        # print(weight)
        aggrevate_feature = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_adc * weight[:, 4:5, None, None, None] +
            x1_dwi * weight[:, 5:6, None, None, None] 
            # x1_dwi * weight[:, 6:7, None, None, None]
        )
        #非0模态的数量
        # N = sequence_code.sum(dim=1, keepdim=True)
        # aggrevate_feature = aggrevate_feature / N[..., None, None, None]
      
    #     x1_t1c = self.down1_t1c(x1_t1c)
    #     x1_t1n = self.down1_t1n(x1_t1n)
    #     x1_t2f = self.down1_t2f(x1_t2f)
    #     x1_t2w = self.down1_t2w(x1_t2w)
    #     x1_dce = self.down1_dce(x1_dce)
    #     x1_adc = self.down1_adc(x1_adc)
    #     x1_dwi = self.down1_dwi(x1_dwi)
        
    # #   # (B, C, H, W, D): 各通道加权求和
    #     aggrevate_feature_2 = (
    #         x1_t1c * weight[:, 0:1, None, None, None] +
    #         x1_t1n * weight[:, 1:2, None, None, None] +
    #         x1_t2f * weight[:, 2:3, None, None, None] +
    #         x1_t2w * weight[:, 3:4, None, None, None] +
    #         x1_dce * weight[:, 4:5, None, None, None] +
    #         x1_adc * weight[:, 5:6, None, None, None] +
    #         x1_dwi * weight[:, 6:7, None, None, None]
    #     )
     
        x2 = self.down1(aggrevate_feature)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
       
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out,x2)
        out = self.up4(out, aggrevate_feature)
        out = self.outc(out)
        # print(out.shape)

        return out
class UNet_brain(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_post_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_pre_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_adc=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dwi=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_post_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_pre_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_adc = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dwi = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
            
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
    def _compute_equal_weights(self, sequence_code):
        """
        计算学习的权重：通过门控网络学习权重
        
        Args:
            sequence_code: [B, 8] 的tensor
            gate_network: 门控网络
            
        Returns:
            weight: [B, 8] 的tensor
        """
        sequence_code = sequence_code.float()
        softmax_output = torch.softmax(sequence_code, dim=-1) 
        weight = sequence_code * softmax_output
        # 防止除零错误，先检查是否有序列被选择
        weight_sum = weight.sum(dim=1, keepdim=True)
        # 确保权重和不为零
        # weight_sum = torch.clamp(weight_sum, min=self.epision)
        weight = weight / weight_sum  # 归一化
        return weight
    def forward(self, x1_t1c,x2_t1n,x3_t2f,x4_t2w,x5_post_dce,x6_pre_dce,x7_adc,x8_dwi,sequence_code,use_equal_weights=True): #mask_code (B,7)
        
        
        if use_equal_weights:
            # 使用等权重
            weight = self._compute_equal_weights(sequence_code)
        else:
        
            sequence_code = sequence_code.float()
            weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
            weight = weight/weight.sum(dim=1, keepdim=True)
        # print(weight)
        aggrevate_feature = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x2_t1n * weight[:, 1:2, None, None, None] +
            x3_t2f * weight[:, 2:3, None, None, None] +
            x4_t2w * weight[:, 3:4, None, None, None] +
            x5_post_dce * weight[:, 4:5, None, None, None] +
            x6_pre_dce * weight[:, 5:6, None, None, None] +
            x7_adc * weight[:, 6:7, None, None, None] +
            x8_dwi * weight[:, 7:8, None, None, None] 
            # x1_dwi * weight[:, 6:7, None, None, None]
        )
        aggrevate_feature =self.inconv_adc(aggrevate_feature)
        #非0模态的数量
        # N = sequence_code.sum(dim=1, keepdim=True)
        # aggrevate_feature = aggrevate_feature / N[..., None, None, None]
      
    #     x1_t1c = self.down1_t1c(x1_t1c)
    #     x1_t1n = self.down1_t1n(x1_t1n)
    #     x1_t2f = self.down1_t2f(x1_t2f)
    #     x1_t2w = self.down1_t2w(x1_t2w)
    #     x1_dce = self.down1_dce(x1_dce)
    #     x1_adc = self.down1_adc(x1_adc)
    #     x1_dwi = self.down1_dwi(x1_dwi)
        
    # #   # (B, C, H, W, D): 各通道加权求和
    #     aggrevate_feature_2 = (
    #         x1_t1c * weight[:, 0:1, None, None, None] +
    #         x1_t1n * weight[:, 1:2, None, None, None] +
    #         x1_t2f * weight[:, 2:3, None, None, None] +
    #         x1_t2w * weight[:, 3:4, None, None, None] +
    #         x1_dce * weight[:, 4:5, None, None, None] +
    #         x1_adc * weight[:, 5:6, None, None, None] +
    #         x1_dwi * weight[:, 6:7, None, None, None]
    #     )
     
        x2 = self.down1(aggrevate_feature)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
       
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out,x2)
        out = self.up4(out, aggrevate_feature)
        out = self.outc(out)
        # print(out.shape)

        return out  
class UNet_seg8(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_post_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_pre_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_adc=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dwi=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_post_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_pre_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_adc = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dwi = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
            
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        self.fc1=nn.Linear(8,8)
        self.fc2=nn.Linear(8,8)

    
    def _compute_equal_weights(self, sequence_code):
        """
        计算学习的权重：通过门控网络学习权重
        
        Args:
            sequence_code: [B, 8] 的tensor
            gate_network: 门控网络
            
        Returns:
            weight: [B, 8] 的tensor
        """
        sequence_code = sequence_code.float()
        softmax_output = torch.softmax(sequence_code, dim=-1) 
        weight = sequence_code * softmax_output
        # 防止除零错误，先检查是否有序列被选择
        weight_sum = weight.sum(dim=1, keepdim=True)
        # 确保权重和不为零
        # weight_sum = torch.clamp(weight_sum, min=self.epision)
        weight = weight / weight_sum  # 归一化
        return weight
    def forward(self, x1_t1c,x2_t1n,x3_t2f,x4_t2w,x5_post_dce,x6_pre_dce,x7_adc,x8_dwi,sequence_code,use_equal_weights=True): #mask_code (B,7)
        
        x1_t1c = self.inconv_t1c(x1_t1c)
        x1_t1n = self.inconv_t1n(x2_t1n)
        x1_t2f = self.inconv_t2f(x3_t2f)
        x1_t2w = self.inconv_t2w(x4_t2w)
        x1_post_dce = self.inconv_post_dce(x5_post_dce)
        x1_pre_dce = self.inconv_pre_dce(x6_pre_dce)
        x1_adc = self.inconv_adc(x7_adc)
        x1_dwi = self.inconv_dwi(x8_dwi)
        
        if use_equal_weights:
            # 使用等权重
            weight = self._compute_equal_weights(sequence_code)
        else:
        
            sequence_code = sequence_code.float()
            weight = sequence_code*torch.softmax(self.fc1(sequence_code),dim=-1)
            weight = weight/weight.sum(dim=1, keepdim=True)
        # print(weight)
        aggrevate_feature = (
            x1_t1c * weight[:, 0:1, None, None, None] +
            x1_t1n * weight[:, 1:2, None, None, None] +
            x1_t2f * weight[:, 2:3, None, None, None] +
            x1_t2w * weight[:, 3:4, None, None, None] +
            x1_post_dce * weight[:, 4:5, None, None, None] +
            x1_pre_dce * weight[:, 5:6, None, None, None] +
            x1_adc * weight[:, 6:7, None, None, None] +
            x1_dwi * weight[:, 7:8, None, None, None] 
            # x1_dwi * weight[:, 6:7, None, None, None]
        )
        
        #非0模态的数量
        # N = sequence_code.sum(dim=1, keepdim=True)
        # aggrevate_feature = aggrevate_feature / N[..., None, None, None]
      
    #     x1_t1c = self.down1_t1c(x1_t1c)
    #     x1_t1n = self.down1_t1n(x1_t1n)
    #     x1_t2f = self.down1_t2f(x1_t2f)
    #     x1_t2w = self.down1_t2w(x1_t2w)
    #     x1_dce = self.down1_dce(x1_dce)
    #     x1_adc = self.down1_adc(x1_adc)
    #     x1_dwi = self.down1_dwi(x1_dwi)
        
    # #   # (B, C, H, W, D): 各通道加权求和
    #     aggrevate_feature_2 = (
    #         x1_t1c * weight[:, 0:1, None, None, None] +
    #         x1_t1n * weight[:, 1:2, None, None, None] +
    #         x1_t2f * weight[:, 2:3, None, None, None] +
    #         x1_t2w * weight[:, 3:4, None, None, None] +
    #         x1_dce * weight[:, 4:5, None, None, None] +
    #         x1_adc * weight[:, 5:6, None, None, None] +
    #         x1_dwi * weight[:, 6:7, None, None, None]
    #     )
     
        x2 = self.down1(aggrevate_feature)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
       
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out,x2)
        out = self.up4(out, aggrevate_feature)
        out = self.outc(out)
        # print(out.shape)

        return out
    
class UNet_amos(nn.Module):
    def __init__(self,in_ch,base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='ConvNeXtBlock', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNeXtBlock' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inconv_t1c=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t1n=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2f=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_t2w=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dce=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_adc=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        self.inconv_dwi=inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1_t1c = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t1n = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2f = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_t2w = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dce = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_adc = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down1_dwi = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
            
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        self.fc1=nn.Linear(7,7)
        self.fc2=nn.Linear(7,7)

    def forward(self, x1_t1c): #mask_code (B,7)
        
        x1_t1c = self.inconv_t1c(x1_t1c)
        x2 = self.down1(x1_t1c)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
       
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out,x2)
        out = self.up4(out, x1_t1c)
        out = self.outc(out)
        # print(out.shape)

        return out
if __name__ == "__main__":
    from thop import profile


    device = torch.device('cuda:0')
    model =  UNet(seg_in_ch=1,cls_in_ch=1, base_ch=32, num_classes=2).to(device)
    x = torch.rand(1, 1,96,96,96).to(device)
    mask_code = torch.tensor([[1,1,1,1,1,1,1]], dtype=torch.float32).to(device)
    # mask_code = mask_code.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # 调整维度为 (B, 7, 1, 1, 1)

    c = model(x,x,x,x,x,x,x,'cls',mask_code)
    
    # flops, params = profile(model, inputs=(x,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')



    # import time
    # print("a")
    # def measure_infer_time(model, x,  num_iter=100):
    #     model.eval()
        
    #     start_time = time.time()
    #     with torch.no_grad():
    #         for _ in range(num_iter):
    #             _ = model(x)
    #     end_time = time.time()
        
    #     infer_time = (end_time - start_time) / num_iter
    #     return infer_time

    # # 示例使用
    # model = UNet(in_ch=1,base_ch=32,num_classes=2).to(device)
    # x = torch.rand(1, 1, 128, 128, 128).to(device)
    # c = model(x)
    # # print(f"Infer time: {measure_infer_time(model, x)} seconds")