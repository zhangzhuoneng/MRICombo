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
   
    def __init__(self, in_ch, base_ch, block='BasicBlock', kernel_size=3, norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.conv = inconv(in_ch, base_ch, block=block_fn, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x):
        return self.conv(x)

class EncoderExpert(nn.Module):
    
    def __init__(self, in_ch, out_ch, scale, kernel_size, block='BasicBlock', pool=True, norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.down = down_block(in_ch, out_ch, num_block=2, block=block_fn, 
                              pool=pool, down_scale=scale, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x):
        return self.down(x)

class DecoderExpert(nn.Module):
   
    def __init__(self, in_ch, out_ch, scale, kernel_size, block='BasicBlock', norm='in'):
        super().__init__()
        block_fn = get_block(block)
        norm_fn = get_norm(norm)
        self.up = up_block(in_ch, out_ch, num_block=2, block=block_fn, 
                          up_scale=scale, kernel_size=kernel_size, norm=norm_fn)
        
    def forward(self, x, skip):
        return self.up(x, skip)

class GatingNetwork(nn.Module):
  
    def __init__(self, in_features, num_experts, top_k=2, num_regions=10, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_regions = num_regions
        self.num_tasks = num_tasks
        
      
        self.region_embedding_dim = 32
        self.task_embedding_dim = 16
        
       
        self.region_embedding = nn.Linear(num_regions, self.region_embedding_dim)
        
       
        self.task_embedding = nn.Linear(num_tasks, self.task_embedding_dim)
        
       
        self.encoding_fusion = nn.Sequential(
            nn.Linear(self.region_embedding_dim + self.task_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
       
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, x, region_ids=None, task_ids=None):
        
        # Process/Handleinput
        if len(x.shape) > 2:
          
            batch_size = x.shape[0]
            channels = x.shape[1]
            pooled_x = F.adaptive_avg_pool3d(x, 1).view(batch_size, channels)
        else:
            
            pooled_x = x
            batch_size = pooled_x.shape[0]
        
       
        if region_ids is None:
            region_ids = torch.zeros(batch_size, self.num_regions, dtype=torch.float32, device=x.device)
        if task_ids is None:
            task_ids = torch.zeros(batch_size, self.num_tasks, dtype=torch.float32, device=x.device)
        
      
        region_emb = self.region_embedding(region_ids)  # [B, region_embedding_dim]
        task_emb = self.task_embedding(task_ids)        # [B, task_embedding_dim]
        
      
        combined_encoding = torch.cat([region_emb, task_emb], dim=1)  # [B, region_embedding_dim + task_embedding_dim]
        
       
        fused_encoding = self.encoding_fusion(combined_encoding)  # [B, 32]
        
       
        combined_features = torch.cat([pooled_x, fused_encoding], dim=1)  # [B, in_features + 32]
        
       
        logits = self.feature_fusion(combined_features)
        gates = F.softmax(logits, dim=1)
       
        
        gates = gates + 1e-7
        gates_sum = gates.sum(dim=1, keepdim=True)
        gates_sum = torch.clamp(gates_sum, min=1e-7)  
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
        
      
        self.seg_experts = nn.ModuleList([
            SequenceExpert(seg_in_ch, base_ch, block, kernel_size[0], norm) for _ in range(8)
        ])
        
       
        self.cls_experts = nn.ModuleList([
            SequenceExpert(cls_in_ch, base_ch, block, kernel_size[0], norm) for _ in range(8)
        ])
        
        
        self.seg_sequence_gate = nn.Linear(8, 8)
        self.cls_sequence_gate = nn.Linear(8, 8)
        self.epision = 1e-8
        
       
        self.encoder_experts = nn.ModuleList([
            nn.ModuleList([
                EncoderExpert(
                    base_ch * (2**i) if i > 0 else base_ch,
                    base_ch * (2**(i+1)) if i > 0 else 2*base_ch,
                    scale[i], kernel_size[i], block, pool, norm
                ) for _ in range(num_encoder_experts)
            ]) for i in range(4)
        ])
        
       
        self.decoder_experts = nn.ModuleList([
            nn.ModuleList([
                DecoderExpert(
                    base_ch * (2**(4-i)),
                    base_ch * (2**(3-i)),
                    scale[3-i], kernel_size[3-i], block, norm
                ) for _ in range(num_decoder_experts)
            ]) for i in range(4)
        ])
        
        
        self.encoder_gates = nn.ModuleList([
            GatingNetwork(
               
                base_ch * (2**i) if i > 0 else base_ch, 
                num_encoder_experts, 
                self.top_k_encoder,
                num_regions=10,  
                num_tasks=2     
            )
            for i in range(4)
        ])
        
      
        self.decoder_gates = nn.ModuleList([
            GatingNetwork(
                base_ch * (2**(4-i)), 
                num_decoder_experts, 
                self.top_k_decoder,
                num_regions=10,  
                num_tasks=2      
            )
            for i in range(4)
        ])
        
       
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
       
            
    
    def _reset_lb_buffers(self):
        self._lb_gates = []

    def _accumulate_lb_gates(self, gates):
        # gates: [B, E]
        if not hasattr(self, '_lb_gates'):
            self._lb_gates = []
        self._lb_gates.append(gates)

    def get_load_balance_loss(self, coeff=1.0, kind='kl'):
        
        if not hasattr(self, '_lb_gates') or len(self._lb_gates) == 0:
            self.last_lb_loss = torch.tensor(0.0, device=self.outc.weight.device)
            return self.last_lb_loss

        losses = []
        for g in self._lb_gates:
          
            pj = g.mean(dim=0)  
            E = pj.numel()
            if kind == 'kl':
                # KL(p || U) = sum_j p_j * log(p_j * E)
                loss = torch.sum(pj * torch.log(pj.clamp_min(1e-8) * E))
            else:
                # MSE 
                loss = torch.mean((pj - 1.0 / E) ** 2)
            losses.append(loss)

        lb_loss = coeff * torch.mean(torch.stack(losses))
        self.last_lb_loss = lb_loss
       
        self._lb_gates.clear()
        return lb_loss
    def _compute_equal_weights(self, sequence_code):
        
       
        num_sequences = sequence_code.sum(dim=1, keepdim=True)  # [B, 1]
        
        num_sequences = torch.clamp(num_sequences, min=1.0)
        
        equal_weights = sequence_code.float() / num_sequences  # [B, 8]
        
        return equal_weights
        
    def _compute_learned_weights(self, sequence_code, gate_network):
        
        sequence_code = sequence_code.float()
        softmax_output = torch.softmax(gate_network(sequence_code), dim=-1) + self.epision
        weight = sequence_code * softmax_output
       
        weight_sum = weight.sum(dim=1, keepdim=True)
       
        weight_sum = torch.clamp(weight_sum, min=self.epision)
        weight = weight / weight_sum  #
        return weight
    
    def _apply_moe_encoder(self, x, experts, gate_network):
        batch_size = x.shape[0]
        channels = x.shape[1]
        pooled = F.adaptive_avg_pool3d(x, 1).view(batch_size, -1)
       
       
        gates = gate_network(x)
        
       
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
       
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  
        top_k_gates = top_k_gates / top_k_sum  
        
       
        outputs = []
        skips = []
        
        for i in range(batch_size):
            
            output = None
            skip = None
            
            for j, idx in enumerate(top_k_indices[i]):
               
                result = experts[idx](x[i:i+1])
                
              
                if isinstance(result, tuple):
                    out, skp = result
                else:
                    out = result
                    skp = x[i:i+1]  
                    
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                    skip = skp * weight
                else:
                    output += out * weight
                    skip += skp * weight
            
            outputs.append(output)
            skips.append(skip)
        
       
        outputs = torch.cat(outputs, dim=0)
        skips = torch.cat(skips, dim=0)
       
        return outputs, skips
    
    def _apply_moe_decoder(self, x, skip, experts, gate_network):
        batch_size = x.shape[0]
        
       
        gates = gate_network(x)
      
        self._accumulate_lb_gates(gates)
        
     
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
       
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  
        top_k_gates = top_k_gates / top_k_sum  
        
       
        outputs = []
        
        for i in range(batch_size):
            
            output = None
            
            for j, idx in enumerate(top_k_indices[i]):
                out = experts[idx](x[i:i+1], skip[i:i+1])
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                else:
                    output += out * weight
            
            outputs.append(output)
        
      
        outputs = torch.cat(outputs, dim=0)
        
        return outputs
    
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, sequence_code, task, region_ids=None, use_equal_weights=False):
        """
        Args:
            region_ids: 
        """
        batch_size = sequence_code.shape[0]
       
        self._reset_lb_buffers()
        
        if task == "seg":
            task_ids = torch.zeros(batch_size, 2, dtype=torch.float32, device=sequence_code.device)
            task_ids[:, 0] = 1.0  # segmentation：[1, 0]
        elif task == "cls":
            task_ids = torch.zeros(batch_size, 2, dtype=torch.float32, device=sequence_code.device)
            task_ids[:, 1] = 1.0  # classification：[0, 1]
        else:
            print("no task error")
            return None
            
        if task == "seg":
          
            x1_features = self.seg_experts[0](x1)
            x2_features = self.seg_experts[1](x2)
            x3_features = self.seg_experts[2](x3)
            x4_features = self.seg_experts[3](x4)
            x5_features = self.seg_experts[4](x5)
            x6_features = self.seg_experts[5](x6)
            x7_features = self.seg_experts[6](x7)
            x8_features = self.seg_experts[7](x8)
            
          
            if use_equal_weights:
             
                weight = self._compute_equal_weights(sequence_code)
            else: 
                weight = self._compute_learned_weights(sequence_code, self.seg_sequence_gate)
            
           
            
    
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
          
            self.aggrevate_feature_out = aggrevate_feature
                
         
            skips = []
            x = aggrevate_feature
            
            for i in range(4):
                x, skip = self._apply_moe_encoder_with_encoding(
                    x, self.encoder_experts[i], self.encoder_gates[i], 
                    region_ids, task_ids
                )
                skips.append(skip)

         
            for i in range(4):
                x = self._apply_moe_decoder_with_encoding(
                    x, skips[-(i+1)], self.decoder_experts[i], self.decoder_gates[i],
                    region_ids, task_ids
                )
                              
                if i == 0:
                    self.encoder_0_out = x
                
            
         
            out = self.outc(x)
           
            self.get_load_balance_loss(coeff=1.0, kind='kl')
          
           
            return out
            
        elif task == "cls":
           
            x1_features = self.cls_experts[0](x1)
            x2_features = self.cls_experts[1](x2)
            x3_features = self.cls_experts[2](x3)
            x4_features = self.cls_experts[3](x4)
            x5_features = self.cls_experts[4](x5)
            x6_features = self.cls_experts[5](x6)
            x7_features = self.cls_experts[6](x7)
            x8_features = self.cls_experts[7](x8)
            
           
            if use_equal_weights:
               
                weight = self._compute_equal_weights(sequence_code)
            else:
                # 使用学习的权重
                weight = self._compute_learned_weights(sequence_code, self.cls_sequence_gate)
            
           
            
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
        
        
        gates = gate_network(x, region_ids, task_ids)
       
        self._accumulate_lb_gates(gates)

     
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
       
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  
        top_k_gates = top_k_gates / top_k_sum  
        
      
        outputs = []
        skips = []
        
        for i in range(batch_size):
           
            output = None
            skip = None
            
            for j, idx in enumerate(top_k_indices[i]):
               
                result = experts[idx](x[i:i+1])
                
              
                if isinstance(result, tuple):
                    out, skp = result
                else:
                    out = result
                    skp = x[i:i+1]  
                    
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                    skip = skp * weight
                else:
                    output += out * weight
                    skip += skp * weight
            
            outputs.append(output)
            skips.append(skip)
        
      
        outputs = torch.cat(outputs, dim=0)
        skips = torch.cat(skips, dim=0)
        
        return outputs, skips
    
    def _apply_moe_decoder_with_encoding(self, x, skip, experts, gate_network, region_ids, task_ids):
        batch_size = x.shape[0]
        
        gates = gate_network(x, region_ids, task_ids)
       
        self._accumulate_lb_gates(gates)

       
        top_k_gates, top_k_indices = torch.topk(gates, gate_network.top_k, dim=1)
       
        top_k_sum = top_k_gates.sum(dim=1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-8)  
        top_k_gates = top_k_gates / top_k_sum  
        
       
        outputs = []
        
        for i in range(batch_size):
           
            output = None
            
            for j, idx in enumerate(top_k_indices[i]):
                out = experts[idx](x[i:i+1], skip[i:i+1])
                weight = top_k_gates[i, j]
                
                if output is None:
                    output = out * weight
                else:
                    output += out * weight
            
            outputs.append(output)
        
       
        outputs = torch.cat(outputs, dim=0)
        
        return outputs

