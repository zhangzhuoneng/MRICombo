import torch
import torch.nn as nn
import os, sys
sys.path.append("..")
import torch.nn.functional as F
from models.unet_utils import inconv, down_block, up_block
from models.conv_layers import BasicBlock, Bottleneck, SingleConv, MBConv, FusedMBConv, ConvNormAct
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

class SequenceExpert(nn.Module):
    """单序列特征提取专家"""
    def __init__(self, in_ch, base_ch, block='BasicBlock', kernel_size=3, norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.conv = inconv(in_ch, base_ch, block=block_fn, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x):
        return self.conv(x)

class EncoderExpert(nn.Module):
    """编码器专家网络"""
    def __init__(self, in_ch, out_ch, scale, kernel_size, block='BasicBlock', pool=True, norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.down = down_block(in_ch, out_ch, num_block=2, block=block_fn, 
                              pool=pool, down_scale=scale, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x):
        return self.down(x)

class DecoderExpert(nn.Module):
    """解码器专家网络"""
    def __init__(self, in_ch, out_ch, scale, kernel_size, block='BasicBlock', norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.up = up_block(in_ch, out_ch, num_block=2, block=block_fn, 
                          up_scale=scale, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x, skip):
        return self.up(x, skip)

class GatingNetwork(nn.Module):
    """专家选择门控网络 - 加入部位编码和任务编码"""
    def __init__(self, in_features, num_experts, top_k=2, num_regions=10, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_regions = num_regions
        self.num_tasks = num_tasks
        
        # 部位编码和任务编码的dimension
        self.region_embedding_dim = 32
        self.task_embedding_dim = 16
        
        # 部位编码Process/Handle层（one-hotinput）
        self.region_embedding = nn.Linear(num_regions, self.region_embedding_dim)
        
        # 任务编码Process/Handle层（one-hotinput）
        self.task_embedding = nn.Linear(num_tasks, self.task_embedding_dim)
        
        # 编码融合层
        self.encoding_fusion = nn.Sequential(
            nn.Linear(self.region_embedding_dim + self.task_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 特征融合层 - 将编码信息与input特征结合
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, x, region_ids=None, task_ids=None):
        """
        Args:
            x: input特征 [B, C, H, W, D] 或 [B, C]
            region_ids: 部位one-hot编码 [B, num_regions] - 10维或6维one-hot向量
            task_ids: 任务one-hot编码 [B, num_tasks] - 2维one-hot向量 [分割, classification]
        """
        # Process/Handleinput特征
        if len(x.shape) > 2:
            # 如果是特征图，先进行全局池化
            batch_size = x.shape[0]
            channels = x.shape[1]
            pooled_x = F.adaptive_avg_pool3d(x, 1).view(batch_size, channels)
        else:
            # 如果已经是向量，直接使用
            pooled_x = x
            batch_size = pooled_x.shape[0]
        
        # 默认编码（如果没有提供）
        if region_ids is None:
            region_ids = torch.zeros(batch_size, self.num_regions, dtype=torch.float32, device=x.device)
        if task_ids is None:
            task_ids = torch.zeros(batch_size, self.num_tasks, dtype=torch.float32, device=x.device)
        
        # Get/Obtain部位编码和任务编码
        region_emb = self.region_embedding(region_ids)  # [B, region_embedding_dim]
        task_emb = self.task_embedding(task_ids)        # [B, task_embedding_dim]
        
        # 拼接部位编码和任务编码
        combined_encoding = torch.cat([region_emb, task_emb], dim=1)  # [B, region_embedding_dim + task_embedding_dim]
        
        # 融合编码信息
        fused_encoding = self.encoding_fusion(combined_encoding)  # [B, 32]
        
        # 将编码信息与input特征结合
        combined_features = torch.cat([pooled_x, fused_encoding], dim=1)  # [B, in_features + 32]
        
        # Calculate/Compute专家权重
        logits = self.feature_fusion(combined_features)
        gates = F.softmax(logits, dim=1)
       
        # 添加一个小值确保数值稳定，防止出现完全为零的情况
        gates = gates + 1e-7
        gates_sum = gates.sum(dim=1, keepdim=True)
        gates_sum = torch.clamp(gates_sum, min=1e-7)  # 确保不为零
        gates = gates / gates_sum
        
        return gates
class MRICombo(nn.Module):
    def __init__(self, seg_in_ch, cls_in_ch, base_ch, scale=[2,2,2,2], kernel_size=[3,3,3,3], 
                 num_classes=1, block='BasicBlock', pool=True, norm='in', 
                 num_encoder_experts=4, num_decoder_experts=4, top_k=2):
        super().__init__()
        
        self.num_encoder_experts = num_encoder_experts
        self.num_decoder_experts = num_decoder_experts
        self.top_k_encoder = min(top_k, num_encoder_experts)
        self.top_k_decoder = min(top_k, num_decoder_experts)
        
        # 序列特征提取专家 - 分割任务
        self.seg_experts = nn.ModuleList([
            SequenceExpert(seg_in_ch, base_ch, block, kernel_size[0], norm) for _ in range(8)
        ])
        
        # 序列特征提取专家 - classification任务
        self.cls_experts = nn.ModuleList([
            SequenceExpert(cls_in_ch, base_ch, block, kernel_size[0], norm) for _ in range(8)
        ])
        
        # 序列权重学习网络
        self.seg_sequence_gate = nn.Linear(8, 8)
        self.cls_sequence_gate = nn.Linear(8, 8)
        self.epision = 1e-8
        
        # 编码器专家 - 共享路径
        self.encoder_experts = nn.ModuleList([
            nn.ModuleList([
                EncoderExpert(
                    base_ch * (2**i) if i > 0 else base_ch,
                    base_ch * (2**(i+1)) if i > 0 else 2*base_ch,
                    scale[i], kernel_size[i], block, pool, norm
                ) for _ in range(num_encoder_experts)
            ]) for i in range(4)
        ])
        
        # 解码器专家 - 分割路径
        self.decoder_experts = nn.ModuleList([
            nn.ModuleList([
                DecoderExpert(
                    base_ch * (2**(4-i)),
                    base_ch * (2**(3-i)),
                    scale[3-i], kernel_size[3-i], block, norm
                ) for _ in range(num_decoder_experts)
            ]) for i in range(4)
        ])
        
        # 编码器门控网络 - 共享路径
        self.encoder_gates = nn.ModuleList([
            GatingNetwork(
                # 使用实际的特征通道数
                base_ch * (2**i) if i > 0 else base_ch, 
                num_encoder_experts, 
                self.top_k_encoder,
                num_regions=10,  # 10个部位
                num_tasks=2      # 2个任务（分割/classification）
            )
            for i in range(4)
        ])
        
        # 解码器门控网络
        self.decoder_gates = nn.ModuleList([
            GatingNetwork(
                base_ch * (2**(4-i)), 
                num_decoder_experts, 
                self.top_k_decoder,
                num_regions=10,  # 10个部位
                num_tasks=2      # 2个任务（分割/classification）
            )
            for i in range(4)
        ])
        
        # output层
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        # Task importance weights (learnable)
            
    
    def _reset_lb_buffers(self):
        self._lb_gates = []

    def _accumulate_lb_gates(self, gates):
        # gates: [B, E]
        if not hasattr(self, '_lb_gates'):
            self._lb_gates = []
        self._lb_gates.append(gates)

    def get_load_balance_loss(self, coeff=1.0, kind='kl'):
        """
        返回本次 forward 收集到的均衡负载正则；调用后自动清空缓冲。
        kind: 'kl' 或 'mse'
        """
        if not hasattr(self, '_lb_gates') or len(self._lb_gates) == 0:
            self.last_lb_loss = torch.tensor(0.0, device=self.outc.weight.device)
            return self.last_lb_loss

        losses = []
        for g in self._lb_gates:
            # 平均到专家dimension [E]
            pj = g.mean(dim=0)  # soft 使用probability
            E = pj.numel()
            if kind == 'kl':
                # KL(p || U) = sum_j p_j * log(p_j * E)
                loss = torch.sum(pj * torch.log(pj.clamp_min(1e-8) * E))
            else:
                # MSE 到均匀分布
                loss = torch.mean((pj - 1.0 / E) ** 2)
            losses.append(loss)

        lb_loss = coeff * torch.mean(torch.stack(losses))
        self.last_lb_loss = lb_loss
        # 用完即清
        self._lb_gates.clear()
        return lb_loss
    def _compute_equal_weights(self, sequence_code):
        """
        Calculate/Compute等权重：对于sequence_code中值为1的序列，给予相等的权重
        
        Args:
            sequence_code: [B, 8] 的tensor，其中1表示该序列存在，0表示不存在
            
        Returns:
            weight: [B, 8] 的tensor，对于存在的序列给予相等权重
        """
        # Calculate/Compute每个sample中有多少个序列存在
        num_sequences = sequence_code.sum(dim=1, keepdim=True)  # [B, 1]
        
        # 防止除零错误
        num_sequences = torch.clamp(num_sequences, min=1.0)
        
        # Calculate/Compute等权重：存在的序列权重为1/num_sequences，不存在的序列权重为0
        equal_weights = sequence_code.float() / num_sequences  # [B, 8]
        
        return equal_weights
        
    def _compute_learned_weights(self, sequence_code, gate_network):
        """
        Calculate/Compute学习的权重：通过门控网络学习权重
        
        Args:
            sequence_code: [B, 8] 的tensor
            gate_network: 门控网络
            
        Returns:
            weight: [B, 8] 的tensor
        """
        sequence_code = sequence_code.float()
        softmax_output = torch.softmax(gate_network(sequence_code), dim=-1) + self.epision
        weight = sequence_code * softmax_output
        # 防止除零错误，先Check是否有序列被选择
        weight_sum = weight.sum(dim=1, keepdim=True)
        # 确保权重和不为零
        weight_sum = torch.clamp(weight_sum, min=self.epision)
        weight = weight / weight_sum  # 归一化
        return weight
    
    def _apply_moe_encoder(self, x, experts, gate_network):
        batch_size = x.shape[0]
        channels = x.shape[1]
        # print(f"Encoder input shape: {x.shape}, channels: {channels}")
        
        # 测试池化后的dimension
        pooled = F.adaptive_avg_pool3d(x, 1).view(batch_size, -1)
        # print(f"Pooled shape: {pooled.shape}")
        # Calculate/Compute门控权重
        gates = gate_network(x)
        
        # Get/Obtaintop-k专家
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
        # 防止除零错误
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  # 确保不为零
        top_k_gates = top_k_gates / top_k_sum  # 重新归一化
        
        # 应用专家网络
        outputs = []
        skips = []
        
        for i in range(batch_size):
            # 加权组合专家output
            output = None
            skip = None
            
            for j, idx in enumerate(top_k_indices[i]):
                # 修改这一行来适应down_block的实际返回值
                result = experts[idx](x[i:i+1])
                
                # 如果down_block返回一个值，就直接使用
                if isinstance(result, tuple):
                    out, skp = result
                else:
                    out = result
                    skp = x[i:i+1]  # 使用input作为skip连接
                    
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                    skip = skp * weight
                else:
                    output += out * weight
                    skip += skp * weight
            
            outputs.append(output)
            skips.append(skip)
        
        # 合并batch结果
        outputs = torch.cat(outputs, dim=0)
        skips = torch.cat(skips, dim=0)
        # print(outputs.shape,skips.shape)
        return outputs, skips
    
    def _apply_moe_decoder(self, x, skip, experts, gate_network):
        batch_size = x.shape[0]
        
        # Calculate/Compute门控权重
        gates = gate_network(x)
        # 收集用于均衡负载
        self._accumulate_lb_gates(gates)
        
        # Get/Obtaintop-k专家
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
        # 防止除零错误
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  # 确保不为零
        top_k_gates = top_k_gates / top_k_sum  # 重新归一化
        
        # 应用专家网络
        outputs = []
        
        for i in range(batch_size):
            # 加权组合专家output
            output = None
            
            for j, idx in enumerate(top_k_indices[i]):
                out = experts[idx](x[i:i+1], skip[i:i+1])
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                else:
                    output += out * weight
            
            outputs.append(output)
        
        # 合并batch结果
        outputs = torch.cat(outputs, dim=0)
        
        return outputs
    
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, sequence_code, task, region_ids=None, use_equal_weights=False):
        """
        Args:
            region_ids: 部位one-hot编码 [B, 10] 或 [B, 6]
        """
        # 根据任务Set/Setup任务ID（one-hot形式）
        batch_size = sequence_code.shape[0]
        # 重置均衡负载缓冲区
        self._reset_lb_buffers()
        
        if task == "seg":
            task_ids = torch.zeros(batch_size, 2, dtype=torch.float32, device=sequence_code.device)
            task_ids[:, 0] = 1.0  # 分割任务：[1, 0]
        elif task == "cls":
            task_ids = torch.zeros(batch_size, 2, dtype=torch.float32, device=sequence_code.device)
            task_ids[:, 1] = 1.0  # classification任务：[0, 1]
        else:
            print("no task error")
            return None
            
        if task == "seg":
            # 序列特征提取
            x1_features = self.seg_experts[0](x1)
            x2_features = self.seg_experts[1](x2)
            x3_features = self.seg_experts[2](x3)
            x4_features = self.seg_experts[3](x4)
            x5_features = self.seg_experts[4](x5)
            x6_features = self.seg_experts[5](x6)
            x7_features = self.seg_experts[6](x7)
            x8_features = self.seg_experts[7](x8)
            
            # 序列加权融合 - 根据parameter选择权重Calculate/Compute方式
            if use_equal_weights:
                # 使用等权重
                weight = self._compute_equal_weights(sequence_code)
            else:
                # 使用学习的权重
                weight = self._compute_learned_weights(sequence_code, self.seg_sequence_gate)
            
            # 可选：Print权重用于调试
            # print(f"Segmentation weights: {weight}")
            
    
            aggrevate_feature = (
                x1_features * weight[:, 0:1, None, None, None] +
                x2_features * weight[:, 1:2, None, None, None] +
                x3_features * weight[:, 2:3, None, None, None] +
                x4_features * weight[:, 3:4, None, None, None] +
                x5_features * weight[:, 4:5, None, None, None] +
                x6_features * weight[:, 5:6, None, None, None] +
                x7_features * weight[:, 6:7, None, None, None] +
                x8_features * weight[:, 7:8, None, None, None] 
            )
            # 暴露聚合特征供外部可视化/分析使用（每次forward都会Update）
            self.aggrevate_feature_out = aggrevate_feature
                
            # 编码器阶段 - 使用MOE，传入部位编码和任务编码
            skips = []
            x = aggrevate_feature
            
            for i in range(4):
                x, skip = self._apply_moe_encoder_with_encoding(
                    x, self.encoder_experts[i], self.encoder_gates[i], 
                    region_ids, task_ids
                )
                skips.append(skip)

            # 解码器阶段 - 使用MOE，传入部位编码和任务编码
            for i in range(4):
                x = self._apply_moe_decoder_with_encoding(
                    x, skips[-(i+1)], self.decoder_experts[i], self.decoder_gates[i],
                    region_ids, task_ids
                )
                                # 暴露encoder_3_out供外部可视化/分析使用（每次forward都会Update）
                if i == 0:
                    self.encoder_0_out = x
                
            
            # output层
            out = self.outc(x)
            # Calculate/Compute均衡负载正则
            self.get_load_balance_loss(coeff=1.0, kind='kl')
            # 返回output和均衡负载正则
           
            return out
            
        elif task == "cls":
            # 序列特征提取
            x1_features = self.cls_experts[0](x1)
            x2_features = self.cls_experts[1](x2)
            x3_features = self.cls_experts[2](x3)
            x4_features = self.cls_experts[3](x4)
            x5_features = self.cls_experts[4](x5)
            x6_features = self.cls_experts[5](x6)
            x7_features = self.cls_experts[6](x7)
            x8_features = self.cls_experts[7](x8)
            
            # 序列加权融合 - 根据parameter选择权重Calculate/Compute方式
            if use_equal_weights:
                # 使用等权重
                weight = self._compute_equal_weights(sequence_code)
            else:
                # 使用学习的权重
                weight = self._compute_learned_weights(sequence_code, self.cls_sequence_gate)
            
            # 可选：Print权重用于调试
            # print(f"Classification weights: {weight}")
            
            aggrevate_feature = (
                x1_features * weight[:, 0:1, None, None, None] +
                x2_features * weight[:, 1:2, None, None, None] +
                x3_features * weight[:, 2:3, None, None, None] +
                x4_features * weight[:, 3:4, None, None, None] +
                x5_features * weight[:, 4:5, None, None, None] +
                x6_features * weight[:, 5:6, None, None, None] +
                x7_features * weight[:, 6:7, None, None, None] +
                x8_features * weight[:, 7:8, None, None, None]
            )
            
            # 编码器阶段 - 使用MOE，传入部位编码和任务编码
            x = aggrevate_feature
            
            for i in range(4):
                x, _ = self._apply_moe_encoder_with_encoding(
                    x, self.encoder_experts[i], self.encoder_gates[i],
                    region_ids, task_ids
                )
            self.get_load_balance_loss(coeff=1.0, kind='kl')
            
            return x
        else:
            print("no task error")

    def _apply_moe_encoder_with_encoding(self, x, experts, gate_network, region_ids, task_ids):
        batch_size = x.shape[0]
        
        # Calculate/Compute门控权重，传入编码信息
        gates = gate_network(x, region_ids, task_ids)
        # 收集用于均衡负载
        self._accumulate_lb_gates(gates)

        # Get/Obtaintop-k专家
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
        # 防止除零错误
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  # 确保不为零
        top_k_gates = top_k_gates / top_k_sum  # 重新归一化
        
        # 应用专家网络
        outputs = []
        skips = []
        
        for i in range(batch_size):
            # 加权组合专家output
            output = None
            skip = None
            
            for j, idx in enumerate(top_k_indices[i]):
                # 修改这一行来适应down_block的实际返回值
                result = experts[idx](x[i:i+1])
                
                # 如果down_block返回一个值，就直接使用
                if isinstance(result, tuple):
                    out, skp = result
                else:
                    out = result
                    skp = x[i:i+1]  # 使用input作为skip连接
                    
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                    skip = skp * weight
                else:
                    output += out * weight
                    skip += skp * weight
            
            outputs.append(output)
            skips.append(skip)
        
        # 合并batch结果
        outputs = torch.cat(outputs, dim=0)
        skips = torch.cat(skips, dim=0)
        
        return outputs, skips
    
    def _apply_moe_decoder_with_encoding(self, x, skip, experts, gate_network, region_ids, task_ids):
        batch_size = x.shape[0]
        
                # Calculate/Compute门控权重，传入编码信息
        gates = gate_network(x, region_ids, task_ids)
        # 收集用于均衡负载
        self._accumulate_lb_gates(gates)

        # Get/Obtaintop-k专家
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
        # 防止除零错误
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  # 确保不为零
        top_k_gates = top_k_gates / top_k_sum  # 重新归一化
        
        # 应用专家网络
        outputs = []
        
        for i in range(batch_size):
            # 加权组合专家output
            output = None
            
            for j, idx in enumerate(top_k_indices[i]):
                out = experts[idx](x[i:i+1], skip[i:i+1])
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                else:
                    output += out * weight
            
            outputs.append(output)
        
        # 合并batch结果
        outputs = torch.cat(outputs, dim=0)
        
        return outputs

## 使用示例：

# # Create模型
# model = MRICombo(
#     seg_in_ch=1, cls_in_ch=1, base_ch=32,
#     num_encoder_experts=3, num_decoder_experts=3
# )

# # 准备input数据
# batch_size = 2
# x1 = torch.randn(batch_size, 1, 64, 64, 64)
# # ... 其他input ...

# # 部位编码：0-9表示10个不同部位
# sequence_code = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
# region_ids = torch.tensor([0, 3])  # 第一个sample是部位0，第二个sample是部位3

# # 使用模型
# output = model(x1, x1, x1, x1, x1, x1, x1, x1, sequence_code, "seg", region_ids)
# print(output.shape)
