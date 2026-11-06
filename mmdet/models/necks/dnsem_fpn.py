# DNTR-main/mmdet-dntr/mmdet/models/necks/dnsem_fpn.py
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS
from .fpn import FPN


class FeatureEncoder(nn.Module):
    """特征编码器：将特征图编码为向量表示，用于对比学习
    
    作用：输入多通道特征图 [B, C, H, W]，输出特征描述向量 [B, hidden_dim]
    用于对比学习等任务
    """
    def __init__(self, in_channels=256, hidden_dim=256):
        super(FeatureEncoder, self).__init__()
        # 基本特征提取模块：包含卷积、归一化和激活
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),  # 降低分辨率
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # LeakyReLU激活
            nn.AdaptiveAvgPool2d(1),  # 空间全局平均池化为 [B, C, 1, 1]
        )
        # 两层全连接，用于进一步提取语义嵌入表征
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),  # 投影到隐空间
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),  # 再次线性变换
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        fea = self.E(x).squeeze(-1).squeeze(-1)  # [B, C]，挤掉空间维度
        out = self.mlp(fea)  # [B, hidden_dim]，映射到输出空间
        return out


class SemanticDenoisingModule(nn.Module):
    """语义去噪模块：专注于增强多尺度特征的语义一致性
    
    设计理念：
    1. 在融合后的特征上工作，此时特征已经包含了多尺度信息
    2. 使用轻量级去噪，主要增强语义一致性
    3. 通过跨尺度注意力机制，增强相邻尺度之间的语义对齐
    
    特点：
    - 轻量级设计，计算效率高
    - 专注于语义信息增强
    - 使用跨尺度注意力机制
    """
    def __init__(self, in_channels=256, num_scales=4):
        super(SemanticDenoisingModule, self).__init__()
        self.num_scales = num_scales
        
        # 为每个尺度创建去噪卷积（用于增强语义一致性）
        # 使用1×1卷积可以更好地融合语义信息
        self.denoising_convs = nn.ModuleList([
            ConvModule(
                in_channels, 
                in_channels, 
                kernel_size=1,  # 使用1×1卷积，专注于语义融合
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)
            ) for _ in range(num_scales)
        ])
        
        # 通道注意力模块（用于增强语义通道）
        # 通道注意力可以帮助识别和增强语义相关的通道
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 跨尺度注意力（用于增强相邻尺度之间的语义对齐）
        # 为每个尺度创建与相邻尺度的注意力连接
        # 只连接相邻尺度：每个尺度只与上一个和下一个尺度连接
        # 这是什么
        # cross_scale_attentions_prev/next 是用于实现跨尺度语义注意力的模块列表。
        # 每个模块负责计算“当前尺度”与“前一个/后一个尺度”之间的语义对齐注意力。
        # 具体做法是：将当前输出特征和相邻尺度特征在通道维拼接->通过1x1卷积压缩还原通道数->再用Sigmoid得到注意力权重。
        # 第一个尺度没有上一个，所以 cross_scale_attentions_prev[0] 为 None；
        # 最后一个尺度没有下一个，所以 cross_scale_attentions_next[-1] 为 None。
        self.cross_scale_attentions_prev = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.Sigmoid()
            ) if i > 0 else None
            for i in range(num_scales)
        ])
        self.cross_scale_attentions_next = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.Sigmoid()
            ) if i < num_scales - 1 else None
            for i in range(num_scales)
        ])
        
        # 残差权重（控制去噪强度，可学习）
        self.denoising_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_scales)
        ])
    
    def forward(self, fused_features):
        """
        Args:
            fused_features: List[Tensor], 融合后的特征（上采样+融合后） [B, C, H, W]
        
        Returns:
            denoised_features: List[Tensor], 去噪后的特征
        """
        denoised_features = []
        
        for i, fused_feat in enumerate(fused_features):
            # 1. 使用1×1卷积进行语义融合
            denoised = self.denoising_convs[i](fused_feat)
            
            # 2. 应用通道注意力，增强语义通道
            channel_att = self.channel_attentions[i](denoised)
            denoised = denoised * channel_att
            
            # 3. 跨尺度注意力（增强相邻尺度之间的语义对齐）
            cross_att_sum = torch.ones_like(denoised)  # 初始化为1，用于累积注意力
            
            if i > 0 and self.cross_scale_attentions_prev[i] is not None:  # 与上一尺度对齐
                prev_feat = fused_features[i - 1]
                # 需要将prev_feat上采样到当前尺度
                if prev_feat.shape[2:] != fused_feat.shape[2:]:
                    prev_feat = F.interpolate(prev_feat, size=fused_feat.shape[2:], mode='nearest')
                combined = torch.cat([denoised, prev_feat], dim=1)
                cross_att = self.cross_scale_attentions_prev[i](combined)
                cross_att_sum = cross_att_sum + cross_att
            
            if i < len(fused_features) - 1 and self.cross_scale_attentions_next[i] is not None:  # 与下一尺度对齐
                next_feat = fused_features[i + 1]
                # 需要将next_feat下采样到当前尺度
                if next_feat.shape[2:] != fused_feat.shape[2:]:
                    next_feat = F.interpolate(next_feat, size=fused_feat.shape[2:], mode='nearest')
                combined = torch.cat([denoised, next_feat], dim=1)
                cross_att = self.cross_scale_attentions_next[i](combined)
                cross_att_sum = cross_att_sum + cross_att
            
            # 归一化注意力（如果有多个相邻尺度，取平均）
            num_neighbors = 1
            if i > 0:
                num_neighbors += 1
            if i < len(fused_features) - 1:
                num_neighbors += 1
            cross_att_sum = cross_att_sum / num_neighbors
            denoised = denoised * cross_att_sum
            
            # 4. 残差连接，保留原始信息
            # 使用可学习的权重平衡原始特征和去噪特征
            weight = torch.sigmoid(self.denoising_weights[i])
            denoised_feat = weight * denoised + (1 - weight) * fused_feat
            
            denoised_features.append(denoised_feat)
        
        return denoised_features


@NECKS.register_module()
class DNSem_FPN(FPN):
    """语义去噪FPN：在上采样后进行语义对齐去噪
    
    设计理念：
    1. 在top-down路径（上采样+融合）后进行语义去噪
    2. 通过对比学习（sem_loss）增强相邻尺度之间的语义一致性
    3. 增强多尺度特征的语义对齐，提升语义信息质量
    
    Args:
        继承FPN的所有参数
        use_denoising (bool): 是否启用去噪模块，默认True
        denoising_weight (float): 去噪模块的权重，默认1.0
        return_loss (bool): 是否返回对比学习损失（仅训练时），默认False
        contrastive_temp (float): 对比学习温度参数，默认0.5
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_denoising=True,
                 denoising_weight=1.0,
                 return_loss=True,
                 contrastive_temp=0.5,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        
        # 调用父类FPN的初始化
        super(DNSem_FPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg
        )
        
        self.use_denoising = use_denoising
        self.denoising_weight = denoising_weight
        self.return_loss = return_loss
        self.contrastive_temp = contrastive_temp
        
        if use_denoising:
            # 创建语义去噪模块
            self.semantic_denoising_module = SemanticDenoisingModule(
                in_channels=out_channels,
                num_scales=num_outs
            )
        
        # 如果需要在训练时返回对比学习损失，创建编码器
        if return_loss:
            # 用于编码融合后的FPN特征 - 语义特征编码器
            self.fpn_sem_encoder = FeatureEncoder(out_channels)
    
    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        """
        Args:
            inputs: List[Tensor], Backbone输出的多尺度特征
            backbone_features: List[Tensor], 原始backbone特征（用于对比学习，可选，这里不使用）
        
        Returns:
            outs: Tuple[Tensor], FPN输出的多尺度特征
            或者 (outs, losses): 如果self.return_loss=True且self.training=True
        """
        # 调用父类的forward方法，得到标准FPN输出（包含上采样+融合）
        fpn_outs = super(DNSem_FPN, self).forward(inputs)
        
        # 如果启用去噪模块
        if self.use_denoising:
            # 将tuple转换为list以便处理
            fpn_features = list(fpn_outs)
            
            # 应用语义去噪模块（在融合后的特征上）
            denoised_features = self.semantic_denoising_module(fpn_features)
            
            # 加权融合原始特征和去噪特征
            final_features = []
            for orig, denoised in zip(fpn_features, denoised_features):
                final_feat = orig + self.denoising_weight * (denoised - orig)
                final_features.append(final_feat)
            
            fpn_outs = tuple(final_features)
        
        # 如果需要返回对比学习损失（仅在训练模式且启用return_loss时）
        if self.training and self.return_loss:
            losses = self._compute_sem_loss(fpn_outs)
            return fpn_outs, losses
        
        return fpn_outs
    
    def _compute_sem_loss(self, fpn_features):
        """
        计算语义对齐损失（sem_loss），用于增强多尺度特征的语义一致性
        
        核心思想：
        1. 语义对齐：相同批次、相邻尺度的FPN特征应该相似（正样本）
        2. 不同尺度或不同batch的特征应该分离（负样本）
        3. 通过对比学习，增强语义一致性，提升语义信息质量
        
        Args:
            fpn_features: Tuple[Tensor], FPN最终输出的特征（融合后）
        
        Returns:
            losses: dict, 包含sem_loss
        """
        x_fpn = list(fpn_features)[:4]  # 只处理前4层FPN特征
        
        scale_size = len(x_fpn)
        if scale_size <= 1:
            device = fpn_features[0].device if len(fpn_features) > 0 else torch.device('cuda')
            return {'sem_loss': torch.tensor(0.0, device=device)}
        
        batch_size = x_fpn[0].shape[0]
        device = x_fpn[0].device
        
        # 提取FPN的语义特征表示
        semantic_fpn = []  # FPN的语义特征
        
        for i in range(scale_size):
            # FPN特征编码（融合后的特征）
            semantic_fpn.append(self.fpn_sem_encoder(x_fpn[i]))  # [B, D]
        
        # ========== 计算语义损失 (sem_loss) ==========
        # 目标：相同批次、相邻尺度的FPN特征应该相似
        
        # 锚点(Anchors)是前 scale_size-1 层的 FPN 特征
        # sem_fpn_anchors: [M, D], 其中 M = (scale_size - 1) * batch_size
        sem_fpn_anchors = torch.cat(semantic_fpn[:-1], dim=0)  # [M, D]
        sem_fpn_anchors_norm = F.normalize(sem_fpn_anchors, p=2, dim=1)
        
        # 所有的键(Keys)包括全部 FPN 特征
        all_fpn_keys = torch.cat(semantic_fpn, dim=0)  # [N, D], N = scale_size * batch_size
        all_keys_norm = F.normalize(all_fpn_keys, p=2, dim=1)
        
        # 计算锚点与所有键的相似度矩阵
        # sim_matrix: [M, N]
        sim_matrix = torch.matmul(sem_fpn_anchors_norm, all_keys_norm.t()) / self.contrastive_temp
        
        # --- 创建正样本掩码 ---
        # 锚点索引
        anchor_scale_indices = torch.arange(scale_size - 1, device=device).repeat_interleave(batch_size)  # [M]
        anchor_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size - 1)  # [M]
        
        # 键索引
        key_scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)  # [N]
        key_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)  # [N]
        
        # 正样本条件:
        # 1. 批次索引相同
        # 2. 键的尺度 = 锚点的尺度 + 1（相邻尺度）
        is_same_batch = (anchor_batch_indices.unsqueeze(1) == key_batch_indices.unsqueeze(0))  # [M, N]
        is_adjacent_scale = (anchor_scale_indices.unsqueeze(1) + 1 == key_scale_indices.unsqueeze(0))  # [M, N]
        
        # pos_mask_sem: [M, N]
        pos_mask_sem = is_same_batch & is_adjacent_scale
        
        # --- 计算损失 ---
        exp_sim = torch.exp(sim_matrix)  # [M, N]
        
        # 分子：正样本相似度之和
        numerators_sem = (exp_sim * pos_mask_sem).sum(dim=1)  # [M]
        
        # 分母：所有样本（正+负）的相似度之和
        denominators_sem = exp_sim.sum(dim=1)  # [M]
        
        # 计算最终损失（避免数值不稳定）
        sem_loss = -torch.log(numerators_sem / (denominators_sem + 1e-8) + 1e-8)
        sem_loss = sem_loss.mean()  # 对所有锚点求平均
        
        return {'sem_loss': sem_loss}
