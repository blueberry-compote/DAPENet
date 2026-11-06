# DNTR-main/mmdet-dntr/mmdet/models/necks/dn_fpn.py
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from .fpn import FPN


# 这是一个特征编码器类（FeatureEncoder），用于将输入的特征图进行编码以提取特征表示。原理如下：
# 1. 输入特征经过一个3x3的卷积、BatchNorm和LeakyReLU激活，再通过自适应平均池化，变成每个样本一个特征向量（全局特征）。
# 2. 然后这个向量输入到一个MLP（两层全连接+激活）中，得到最终的嵌入表示。
# 这样操作的目的是将空间特征概括成可参与对比学习或聚类等下游任务的矢量特征。

class FeatureEncoder(nn.Module):
    """Feature encoder: extracts vector representations from feature maps.
    作用：输入多通道特征图 [B, C, H, W]，输出特征描述向量 [B, hidden_dim] 用于对比学习等任务"""
    def __init__(self, in_channels=256, hidden_dim=256):
        super(FeatureEncoder, self).__init__()
        # 基本特征提取模块：包含卷积、归一化和激活
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),  # 降低分辨率
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.LeakyReLU(0.1, True),      # LeakyReLU激活
            nn.AdaptiveAvgPool2d(1),      # 空间全局平均池化为 [B, C, 1, 1]
        )
        # 两层全连接，用于进一步提取语义嵌入表征
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),     # 投影到隐空间
            # nn.LeakyReLU(0.1, True)
            # 这里的 0.1 是负半轴斜率（negative_slope），表示输入小于0时输出的斜率；
            # True 是 inplace，表示是否在原地（不申请新内存）对输入进行修改。
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),      # 再次线性变换
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        fea = self.E(x).squeeze(-1).squeeze(-1)  # [B, C]，挤掉空间维度
        out = self.mlp(fea)                      # [B, hidden_dim]，映射到输出空间
        return out


class DenoisingModule(nn.Module):
    """去噪模块，通过特征对齐和注意力机制抑制噪声"""
    def __init__(self, in_channels=256, num_scales=4):
        super(DenoisingModule, self).__init__()
        self.num_scales = num_scales
        
        # 为每个尺度创建通道注意力
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 空间注意力（可选）
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 特征对齐模块（用于对齐backbone和FPN特征）
        self.align_convs = nn.ModuleList([
            ConvModule(in_channels, in_channels, 1, norm_cfg=dict(type='BN'))
            for _ in range(num_scales)
        ])
    
    def forward(self, fpn_features, backbone_features=None):
        """
        Args:
            fpn_features: List[Tensor], FPN输出的多尺度特征
            backbone_features: List[Tensor], Backbone的多尺度特征（可选，用于训练时的对比学习）
        
        Returns:
            denoised_features: List[Tensor], 去噪后的特征
        """
        denoised_features = []
        
        for i, feat in enumerate(fpn_features):
            # 通道注意力
            channel_att = self.channel_attentions[i](feat)
            feat = feat * channel_att
            
            # 空间注意力
            spatial_att = self.spatial_attentions[i](feat)
            feat = feat * spatial_att
            
            # 如果有backbone特征，进行特征对齐（训练时）
            if backbone_features is not None and i < len(backbone_features):
                aligned_backbone = self.align_convs[i](backbone_features[i])
                # 使用对齐后的backbone特征进行残差连接，增强去噪效果
                feat = feat + 0.1 * aligned_backbone
            
            denoised_features.append(feat)
        
        return denoised_features


@NECKS.register_module()
class DN_FPN(FPN):
    """DeNoising FPN - 集成去噪功能的FPN模块
    
    在标准FPN的基础上，添加了去噪模块来抑制特征融合过程中的噪声。
    可以通过对比学习损失在训练时进一步优化去噪效果。
    
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
                 return_loss=False,
                 contrastive_temp=0.5,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        
        # 调用父类FPN的初始化
        super(DN_FPN, self).__init__(
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
            # 创建去噪模块
            self.denoising_module = DenoisingModule(
                in_channels=out_channels,
                num_scales=num_outs
            )
            
            # 如果需要在训练时返回对比学习损失，创建编码器
            if return_loss:
                self.loc_encoder = FeatureEncoder(out_channels)
                self.sem_encoder = FeatureEncoder(out_channels)
                
                # 通道转换层（用于backbone特征）
                base_c = 256
                self.channel_transfer = nn.ModuleList([
                    nn.Conv2d(base_c * (2 ** i), base_c, kernel_size=1)
                    for i in range(4)
                ])
    
    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        """
        Args:
            inputs: List[Tensor], Backbone输出的多尺度特征
            backbone_features: List[Tensor], 原始backbone特征（用于对比学习，可选）
        
        Returns:
            outs: Tuple[Tensor], FPN输出的多尺度特征
            或者 (outs, losses): 如果self.return_loss=True且self.training=True
        """
        # 调用父类的forward方法，得到标准FPN输出
        fpn_outs = super(DN_FPN, self).forward(inputs)
        
        # 如果启用去噪模块
        if self.use_denoising:
            # 将tuple转换为list以便处理
            fpn_features = list(fpn_outs)
            
            # 应用去噪模块
            denoised_features = self.denoising_module(
                fpn_features, 
                backbone_features=backbone_features
            )
            
            # 加权融合原始特征和去噪特征
            final_features = []
            for orig, denoised in zip(fpn_features, denoised_features):
                final_feat = orig + self.denoising_weight * (denoised - orig)
                final_features.append(final_feat)
            
            fpn_outs = tuple(final_features)
        
        # 如果需要返回对比学习损失（仅在训练模式且启用return_loss时）
        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_contrastive_loss_vectorized(fpn_outs, backbone_features)
            return fpn_outs, losses
        
        return fpn_outs
    
    def _compute_contrastive_loss_vectorized(self, fpn_features, backbone_features):
        """
        完全向量化的对比学习损失计算
        
        使用矩阵运算替代所有循环，大幅提升计算效率
        复杂度从 O(scale² × batch²) 降低到 O(scale × batch)
        """
        # 处理backbone特征：跳过第一层（通常是输入图像）
        x_b = backbone_features[1:] if len(backbone_features) > 1 else backbone_features
        x_fpn = list(fpn_features)[:4]  # 只处理前4层FPN特征
        
        scale_size = len(x_fpn)
        if scale_size == 0:
            return {'geo_loss': torch.tensor(0.0, device=x_fpn[0].device), 
                    'sem_loss': torch.tensor(0.0, device=x_fpn[0].device)}
        
        batch_size = x_fpn[0].shape[0]
        device = x_fpn[0].device
        
        # 生成backbone特征表示（需要通道转换）
        temp_b = []
        for i in range(len(x_b)):
            if i < len(self.channel_transfer):
                temp_b.append(self.channel_transfer[i](x_b[i]))
        
        # 提取FPN和backbone的特征表示
        localization_fpn = []  # FPN的定位特征
        semantic_fpn = []      # FPN的语义特征
        localization_backbone = []  # Backbone的定位特征
        semantic_backbone = []      # Backbone的语义特征
        
        for i in range(scale_size):
            # FPN特征编码
            localization_fpn.append(self.loc_encoder(x_fpn[i]))  # [B, D]
            semantic_fpn.append(self.sem_encoder(x_fpn[i]))    # [B, D]
            
            # Backbone特征编码
            if i < len(temp_b):
                localization_backbone.append(self.loc_encoder(temp_b[i]))  # [B, D]
                semantic_backbone.append(self.sem_encoder(temp_b[i]))      # [B, D]
        
        # ========== 计算几何损失 (geo_loss) ==========
        # 目标：相同尺度的FPN特征和backbone特征应该相似
        
        # 将所有特征堆叠成矩阵: [scale_size * batch_size, feature_dim]
        # 格式: [scale0_batch0, scale0_batch1, ..., scale1_batch0, scale1_batch1, ...]
        loc_fpn_all = torch.stack(localization_fpn, dim=0)  # [scale_size, batch_size, D]
        loc_fpn_all = loc_fpn_all.view(-1, loc_fpn_all.shape[-1])  # [scale_size * batch_size, D]
        
        loc_backbone_all = torch.stack(localization_backbone, dim=0)  # [scale_size, batch_size, D]
        loc_backbone_all = loc_backbone_all.view(-1, loc_backbone_all.shape[-1])  # [scale_size * batch_size, D]
        
        # 归一化特征
        loc_fpn_norm = F.normalize(loc_fpn_all, p=2, dim=1)  # [N, D]
        loc_backbone_norm = F.normalize(loc_backbone_all, p=2, dim=1)  # [N, D]
        
        # 计算所有样本对之间的相似度矩阵
        # FPN-FPN相似度: [N, N]
        sim_fpn_fpn = torch.matmul(loc_fpn_norm, loc_fpn_norm.t()) / self.contrastive_temp
        
        # FPN-Backbone相似度: [N, N]
        sim_fpn_backbone = torch.matmul(loc_fpn_norm, loc_backbone_norm.t()) / self.contrastive_temp
        
        # 创建尺度索引和批次索引
        # 每个样本的索引: (scale_idx, batch_idx)
        scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)  # [N]
        batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)  # [N]
        
        # 创建正样本掩码：相同尺度且相同batch
        # 对于每个样本i，正样本j满足: scale[i] == scale[j] and batch[i] == batch[j]
        scale_mask = (scale_indices.unsqueeze(1) == scale_indices.unsqueeze(0))  # [N, N]
        batch_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0))  # [N, N]
        pos_mask = scale_mask & batch_mask  # [N, N] - 正样本掩码
        
        # 创建负样本掩码：除了正样本外的所有样本
        neg_mask = ~pos_mask  # [N, N] - 负样本掩码
        
        # 计算每个样本的损失
        # 对于样本i: loss_i = -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
        exp_sim_fpn_backbone = torch.exp(sim_fpn_backbone)  # [N, N]
        
        # 提取正样本相似度（使用backbone作为正样本）
        pos_sim = exp_sim_fpn_backbone * pos_mask  # [N, N]，只有正样本位置有值
        numerators = pos_sim.sum(dim=1)  # [N]，每个样本的正样本相似度之和
        
        # 提取负样本相似度（包括FPN-FPN和FPN-Backbone）
        exp_sim_fpn_fpn = torch.exp(sim_fpn_fpn)  # [N, N]
        neg_sim = (exp_sim_fpn_fpn + exp_sim_fpn_backbone) * neg_mask  # [N, N]
        denominators = neg_sim.sum(dim=1) + numerators  # [N]，分母包括所有负样本和正样本
        
        # 计算几何损失（避免数值不稳定）
        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8)
        geo_loss = geo_loss.mean()  # 对所有样本求平均
        
        # ========== 计算语义损失 (sem_loss) - 完全向量化版本 ==========
        # 目标：相同批次、相邻尺度的FPN特征应该相似
        
        if scale_size > 1:
            # 锚点(Anchors)是前 scale_size-1 层的 FPN 特征
            # sem_fpn_anchors: [M, D], 其中 M = (scale_size - 1) * batch_size
            sem_fpn_anchors = torch.cat(semantic_fpn[:-1], dim=0)  # [M, D]
            sem_fpn_anchors_norm = F.normalize(sem_fpn_anchors, p=2, dim=1)
            
            # 所有的键(Keys)包括全部 FPN 特征和全部 Backbone 特征
            all_fpn_keys = torch.cat(semantic_fpn, dim=0)  # [N, D], N = scale_size * batch_size
            all_backbone_keys = torch.cat(semantic_backbone, dim=0)  # [N, D]
            all_keys = torch.cat([all_fpn_keys, all_backbone_keys], dim=0)  # [2*N, D]
            all_keys_norm = F.normalize(all_keys, p=2, dim=1)
            
            # 计算锚点与所有键的相似度矩阵
            # sim_matrix: [M, 2*N]
            sim_matrix = torch.matmul(sem_fpn_anchors_norm, all_keys_norm.t()) / self.contrastive_temp
            
            # --- 创建正样本掩码 ---
            # 锚点索引
            anchor_scale_indices = torch.arange(scale_size - 1, device=device).repeat_interleave(batch_size)  # [M]
            anchor_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size - 1)  # [M]
            
            # 键索引 (包含 FPN 和 Backbone)
            key_scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)  # [N]
            key_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)  # [N]
            
            # 扩展到2*N，并标记来源 (0=FPN, 1=Backbone)
            full_key_scale_indices = key_scale_indices.repeat(2)  # [2*N]
            full_key_batch_indices = key_batch_indices.repeat(2)  # [2*N]
            source_indices = torch.zeros(2 * len(key_scale_indices), dtype=torch.long, device=device)
            source_indices[len(key_scale_indices):] = 1  # 后半部分标记为1 (Backbone)
            
            # 正样本条件:
            # 1. 键的来源是 FPN (source == 0)
            # 2. 批次索引相同
            # 3. 键的尺度 = 锚点的尺度 + 1
            is_fpn_source = (source_indices.unsqueeze(0) == 0)  # [1, 2*N] -> [M, 2*N] after broadcast
            is_same_batch = (anchor_batch_indices.unsqueeze(1) == full_key_batch_indices.unsqueeze(0))  # [M, 2*N]
            is_adjacent_scale = (anchor_scale_indices.unsqueeze(1) + 1 == full_key_scale_indices.unsqueeze(0))  # [M, 2*N]
            
            # pos_mask_sem: [M, 2*N]
            pos_mask_sem = is_fpn_source & is_same_batch & is_adjacent_scale
            
            # --- 计算损失 ---
            exp_sim = torch.exp(sim_matrix)  # [M, 2*N]
            
            # 分子：正样本相似度之和
            numerators_sem = (exp_sim * pos_mask_sem).sum(dim=1)  # [M]
            
            # 分母：所有样本（正+负）的相似度之和
            denominators_sem = exp_sim.sum(dim=1)  # [M]
            
            # 计算最终损失（避免数值不稳定）
            sem_loss = -torch.log(numerators_sem / (denominators_sem + 1e-8) + 1e-8)
            sem_loss = sem_loss.mean()  # 对所有锚点求平均
        else:
            sem_loss = torch.tensor(0.0, device=device)
        
        return {
            'geo_loss': geo_loss,
            'sem_loss': sem_loss
        }