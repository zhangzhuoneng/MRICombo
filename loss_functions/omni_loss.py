import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
from utils import get_key_task,TEMPLATE


class FocalBinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalBinaryDiceLoss, self).__init__()
        assert 0 <= alpha <= 1, "alpha must be in [0,1]"
        # 确保smooth是浮点数，而不是字符串
        self.smooth = float(smooth)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predict, target):
        assert predict.shape == target.shape, "Shape mismatch"
        
        # 1. Sigmoid归一化并展平
        probs = torch.sigmoid(predict)
        probs = probs.contiguous().view(probs.shape[0], -1)  # [B, N]
        target = target.contiguous().view(target.shape[0], -1)
        
        # 2. 计算逐点Focal权重
        focal_weights = torch.where(
            target == 1,
            self.alpha * (1 - probs).pow(self.gamma),
            (1 - self.alpha) * probs.pow(self.gamma)
        )  # [B, N]
        
        # 3. 加权后的Dice Loss（逐点加权后再聚合）
        weighted_intersection = torch.sum(focal_weights * probs * target, dim=1)
        weighted_union = torch.sum(focal_weights * (probs + target), dim=1)
        
        # 确保smooth是张量或浮点数，不是字符串
        smooth = torch.tensor(self.smooth, device=predict.device, dtype=predict.dtype)
        dice_loss = 1 - (2 * weighted_intersection + smooth) / (weighted_union + smooth)
        
        # 4. Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
       
        # return dice_loss
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        # dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)
        # self.dice = FocalBinaryDiceLoss(**kwargs)

    def forward(self, predict, target, name, TEMPLATE):
        
        total_loss = []
        predict = torch.sigmoid(predict)
        
        # for i in range(self.num_classes):
        #     # if i != self.ignore_index:
        #     dice_loss = self.dice(predict[:, i], target[:, i])
        #     # if self.weight is not None:
        #     #     assert self.weight.shape[0] == self.num_classes, \
        #     #         'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
        #     #     dice_loss *= self.weights[i]
        #     total_loss.append(dice_loss)
        B = predict.shape[0]
        for b in range(B):
            template_key = get_key_task(name[b])
            organ_list = TEMPLATE[template_key]
            # print( name[b],template_key,organ_list,)
            for organ in organ_list: 
                dice_loss = self.dice(predict[b, organ-1,:,:,:], target[b, organ-1,:,:,:])
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss==total_loss]

        return total_loss.sum()/total_loss.shape[0]
    

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='sum'):
#         """
#         初始化Focal Loss
        
#         参数:
#             alpha (Tensor, optional): 各类别的权重系数，用于处理类别不平衡
#                                     可以是tensor，形状为[C]，C为类别数
#             gamma (float): focusing参数，用于调节简单样本的权重降低程度
#                           gamma越大，对易分样本的惩罚越大
#             reduction (str): 'none' | 'mean' | 'sum'
#                             指定如何减少损失值
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
#     def forward(self, inputs, targets, name, TEMPLATE):
#         """
#         计算Focal Loss
        
#         参数:
#             inputs (Tensor): 预测值，形状为[N, C]，N为批量大小，C为类别数
#             targets (Tensor): 目标值，形状为[N]，值范围为[0, C-1]
            
#         返回:
#             Tensor: 计算得到的Focal Loss
#         """
#         # 获取batch size和类别数
#         N, C = inputs.size()
        
#         # 计算普通的交叉熵损失
#         log_pt = F.log_softmax(inputs, dim=1)
#         log_pt = log_pt.gather(1, targets.view(-1, 1))
#         log_pt = log_pt.view(-1)
#         pt = torch.exp(log_pt)
        
        
#         # for b in range(N):
#         #     template_key = get_key_task(name[b])
#         #     organ_list = TEMPLATE[template_key]
#         #     # print( name[b],template_key,organ_list,)
#         #     for organ in organ_list: 
#         #         dice_loss = self.dice(predict[b, organ-1,:,:,:], target[b, organ-1,:,:,:])
                
#         # 如果提供了alpha，应用类别权重
#         if self.alpha is not None:
#             if isinstance(self.alpha, torch.Tensor):
#                 # 确保alpha的形状正确 [C]
#                 assert self.alpha.size(0) == C, \
#                     f"Alpha size {self.alpha.size(0)} must match number of classes {C}"
                
#                 # 根据目标类别选择对应的alpha值
#                 batch_alpha = self.alpha.gather(0, targets)
#                 loss = -batch_alpha * ((1 - pt) ** self.gamma) * log_pt
#             else:
#                 # 如果alpha是标量，直接使用
#                 loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt
#         else:
#             # 不使用alpha时的focal loss
#             loss = -((1 - pt) ** self.gamma) * log_pt
        
#         # 根据reduction方式处理loss
#         if self.reduction == 'none':
#             return loss
#         elif self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             raise ValueError(f"Unsupported reduction mode: {self.reduction}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='sum'):
        """
        初始化Focal Loss
        
        参数:
            alpha: 可以是以下形式之一:
                  - Tensor: 形状为[C]的tensor，C为类别数，用于单个数据集
                  - dict: 键为数据集名称，值为对应数据集的alpha张量或None
            gamma (float): focusing参数
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, name):
        """
        计算Focal Loss
        
        参数:
            inputs (Tensor): 预测值，形状为[N, C]
            targets (Tensor): 目标值，形状为[N]
            name (list): 每个样本对应的数据集名称
            TEMPLATE: 模板信息(保留原有功能)
            
        返回:
            Tensor: 计算得到的Focal Loss
        """
        # 获取batch size和类别数
        N, C = inputs.size()
        
        # 计算普通的交叉熵损失
        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.view(-1, 1))
        log_pt = log_pt.view(-1)
        pt = torch.exp(log_pt)
        
        # 为batch中的每个样本选择对应数据集的alpha
        if self.alpha is not None:
            if isinstance(self.alpha, dict):
                # 创建一个与batch大小相同的损失列表
                losses = []
                
                # 对batch中的每个样本单独计算loss
                for i in range(N):
                    dataset_name = name[i][:3]  # 获取当前样本的数据集名称
                    # print(dataset_name)
                    # 检查是否有该数据集的alpha且alpha不为None
                    if dataset_name in self.alpha and self.alpha[dataset_name] is not None:
                        dataset_alpha = self.alpha[dataset_name]
                        # print( dataset_alpha)
                        # 获取当前样本的目标类别
                        target_class = targets[i].item()
                        
                        # 获取当前样本对应类别的alpha值
                        sample_alpha = dataset_alpha[target_class]
                        
                        # 计算该样本的focal loss
                        sample_loss = -sample_alpha * ((1 - pt[i]) ** self.gamma) * log_pt[i]
                        losses.append(sample_loss)
                    else:
                        # 如果没有对应的alpha或alpha为None，使用无加权的focal loss
                        # sample_loss = -((1 - pt[i]) ** self.gamma) * log_pt[i]
                        sample_loss = -((1 - pt[i])) * log_pt[i]
                        losses.append(sample_loss)
                
                # 将所有样本的loss合并为一个tensor
                loss = torch.stack(losses)
                
            elif isinstance(self.alpha, torch.Tensor):
                # 如果alpha是tensor，则按照原来的方式处理
                assert self.alpha.size(0) == C, \
                    f"Alpha size {self.alpha.size(0)} must match number of classes {C}"
                
                batch_alpha = self.alpha.gather(0, targets)
                loss = -batch_alpha * ((1 - pt) ** self.gamma) * log_pt
            else:
                # 如果alpha是标量
                loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt
        else:
            # 不使用alpha时的focal loss
            # loss = -((1 - pt) ** self.gamma) * log_pt
            loss = -((1 - pt)) * log_pt
        
        # 根据reduction方式处理loss
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
        
class CELoss(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(CELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.criterion =  FocalLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        B = predict.shape[0]
        total_loss = []
        # for i in range(self.num_classes):
        #     # if i != self.ignore_index:
        #     ce_loss = self.criterion(predict[:, i], target[:, i])
        #     ce_loss = torch.mean(ce_loss, dim=[1,2,3])

        #     # ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

        #     # total_loss.append(ce_loss_avg)
        #     total_loss.append( ce_loss)
        for b in range(B):
            template_key = get_key_task(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ-1,:,:,:], target[b,organ-1,:,:,:])
                
                # ce_loss *= weights[organ-1]
                total_loss.append(ce_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]
    
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super(BCEDiceLoss, self).__init__()

#     def forward(self, inputs, targets):
#         loss = 0
#         for i in range(3):
#             input = inputs[:,i,:,:,:]
#             target = targets[:,i,:,:,:]
#             bce = F.binary_cross_entropy_with_logits(input, target)
#             smooth = 1e-5
#             input = torch.sigmoid(input)
#             target = target.float()
#             num = 2 * (input * target).sum()+smooth
#             den = input.sum() + target.sum() + smooth
#             dice = 1.0 - num / den
#             loss += dice+0.5*bce
#         return loss/3