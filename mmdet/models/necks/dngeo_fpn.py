# DNTR-main/mmdet-dntr/mmdet/models/necks/dngeo_fpn.py
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


class PositionDenoisingModule(nn.Module):
    """位置去噪模块：专注于去除1×1卷积引入的位置噪声
    
    设计理念：
    1. 在lateral特征（1×1卷积后）上工作，此时特征还在各自尺度
    2. 使用轻量级去噪，主要去除位置噪声
    3. 通过对比学习，让去噪后的lateral特征与原始backbone特征在位置上对齐
    
    特点：
    - 轻量级设计，计算效率高
    - 专注于位置信息保留
    - 使用可学习的残差权重
    """
    def __init__(self, in_channels=256, num_scales=4):
        super(PositionDenoisingModule, self).__init__()
        self.num_scales = num_scales
        
        # 为每个尺度创建去噪卷积（用于去除1×1卷积的噪声）
        # 使用3×3卷积可以更好地保留空间位置信息
        self.denoising_convs = nn.ModuleList([
            ConvModule(
                in_channels, 
                in_channels, 
                kernel_size=3, 
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)
            ) for _ in range(num_scales)
        ])
        
        # 空间注意力模块（用于抑制位置噪声）
        # 空间注意力可以帮助识别和抑制噪声位置
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 通道注意力模块（用于抑制噪声通道）
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 残差权重（控制去噪强度，可学习）
        self.denoising_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_scales)
        ])
    
    def forward(self, lateral_features, backbone_features=None):
        """
        Args:
            lateral_features: List[Tensor], 1×1卷积后的lateral特征 [B, C, H, W]
            backbone_features: List[Tensor], 原始backbone特征（用于对比学习，可选）
        
        Returns:
            denoised_features: List[Tensor], 去噪后的lateral特征
        """
        denoised_features = []
        
        for i, lateral_feat in enumerate(lateral_features):
            # 1. 使用3×3卷积进行去噪（去除1×1卷积引入的噪声）
            # 3×3卷积可以更好地保留空间位置信息
            denoised = self.denoising_convs[i](lateral_feat)
            
            # 2. 应用空间注意力，抑制位置噪声
            spatial_att = self.spatial_attentions[i](denoised)
            denoised = denoised * spatial_att
            
            # 3. 应用通道注意力，抑制噪声通道
            channel_att = self.channel_attentions[i](denoised)
            denoised = denoised * channel_att
            
            # 4. 残差连接，保留原始信息
            # 使用可学习的权重平衡原始特征和去噪特征
            weight = torch.sigmoid(self.denoising_weights[i])
            denoised_feat = weight * denoised + (1 - weight) * lateral_feat
            
            denoised_features.append(denoised_feat)
        
        return denoised_features


@NECKS.register_module()
class DNGeo_FPN(FPN):
    """位置去噪FPN：在上采样前进行位置对齐去噪
    
    设计理念：
    1. 在1×1卷积后、上采样前进行位置去噪
    2. 通过对比学习（geo_loss）让lateral特征与backbone特征在位置上对齐
    3. 去除1×1卷积引入的位置噪声，保留准确的位置信息
    
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
        super(DNGeo_FPN, self).__init__(
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
            # 创建位置去噪模块
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels,
                num_scales=num_outs
            )
        
        # 如果需要在训练时返回对比学习损失，创建编码器和通道转换层
        if return_loss:
            # 用于编码lateral特征（1×1卷积后）- 位置特征编码器
            self.lateral_loc_encoder = FeatureEncoder(out_channels)
            
            # 用于编码backbone特征（原始特征）- 位置特征编码器
            self.backbone_loc_encoder = FeatureEncoder(out_channels)
            
            # 通道转换层（用于backbone特征，将不同通道数统一为out_channels）
            used_in_channels = in_channels[self.start_level:self.backbone_end_level]
            self.channel_transfer = nn.ModuleList()
            for c in used_in_channels:
                if c == out_channels:
                    self.channel_transfer.append(nn.Identity())
                else:
                    self.channel_transfer.append(nn.Conv2d(c, out_channels, kernel_size=1))


    def _build_laterals_with_denoising(self, inputs, backbone_features=None):
        """
        构建lateral特征并应用位置去噪（核心创新点）
        
        注意：此方法需要在1×1卷积后、top-down路径前插入去噪逻辑
        
        Args:
            inputs: List[Tensor], Backbone输出的多尺度特征
            backbone_features: List[Tensor], 原始backbone特征（用于对比学习，可选）
        
        Returns:
            laterals: List[Tensor], 去噪后的lateral特征
        """
        # 构建laterals（使用1×1卷积）- 复用父类逻辑
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # 对1×1卷积后的特征进行位置去噪（核心创新点）
        if self.use_denoising:
            # 应用位置去噪模块
            denoised_laterals = self.position_denoising_module(laterals)
            
            # 加权融合原始lateral和去噪后的lateral
            final_laterals = []
            for orig, denoised in zip(laterals, denoised_laterals):
                final_lateral = orig + self.denoising_weight * (denoised - orig)
                final_laterals.append(final_lateral)
            
            laterals = final_laterals
        
        return laterals
    
    def _build_topdown_and_outputs(self, laterals, inputs):
        """
        构建top-down路径和最终输出 - 复用父类FPN的逻辑
        
        Args:
            laterals: List[Tensor], lateral特征（已去噪）
            inputs: List[Tensor], Backbone输出的多尺度特征（用于extra_convs）
        
        Returns:
            outs: Tuple[Tensor], FPN输出的多尺度特征
        """
        # 构建top-down路径 - 复用父类逻辑
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        
        # 构建输出（使用3×3卷积）- 复用父类逻辑
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # 添加额外层级 - 复用父类逻辑
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        return tuple(outs)
    
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
        assert len(inputs) == len(self.in_channels)
        
        # 1. 构建laterals并应用位置去噪（核心创新点）
        laterals = self._build_laterals_with_denoising(inputs, backbone_features)
        
        # 2. 构建top-down路径和最终输出（复用父类逻辑）
        fpn_outs = self._build_topdown_and_outputs(laterals, inputs)
        
        # 3. 如果需要返回对比学习损失（仅在训练模式且启用return_loss时）
        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_geo_loss(laterals, backbone_features)
            return fpn_outs, losses
        
        return fpn_outs
    
    def _compute_geo_loss(self, lateral_features, backbone_features):
        """
        计算位置对齐损失（geo_loss），用于去除1×1卷积的位置噪声
        
        核心思想：
        1. 位置对齐：去噪后的lateral特征应该与原始backbone特征在位置上对齐（正样本）
        2. 不同尺度或不同batch的特征应该分离（负样本）
        3. 通过对比学习，学习去除位置噪声，保留准确的位置信息
        
        Args:
            lateral_features: List[Tensor], 去噪后的lateral特征（1×1卷积后）
            backbone_features: List[Tensor], 原始backbone特征
        
        Returns:
            losses: dict, 包含geo_loss
        """
        
        x_b = backbone_features[self.start_level:self.backbone_end_level]
        x_lateral = list(lateral_features)  # laterals 本来就是这段 backbone 做 1x1 得到的

        # 如你担心有 extra levels，保险写法：
        n = min(len(x_b), len(x_lateral))
        x_b = x_b[:n]
        x_lateral = x_lateral[:n]
        
        scale_size = len(x_lateral)
        if scale_size == 0:
            device = lateral_features[0].device if len(lateral_features) > 0 else torch.device('cuda')
            return {'geo_loss': torch.tensor(0.0, device=device)}
        
        batch_size = x_lateral[0].shape[0]
        device = x_lateral[0].device
        
        # 生成backbone特征表示（需要通道转换）
        temp_b = [self.channel_transfer[i](x_b[i]) for i in range(len(x_b))]
        
        # 提取lateral和backbone的位置特征表示
        lateral_loc = []  # lateral的定位特征
        backbone_loc = []  # Backbone的定位特征
        
        for i in range(scale_size):
            # Lateral特征编码（去噪后的1×1卷积特征）
            lateral_loc.append(self.lateral_loc_encoder(x_lateral[i]))  # [B, D]
            
            # Backbone特征编码（原始特征）
            if i < len(temp_b):
                backbone_loc.append(self.backbone_loc_encoder(temp_b[i]))  # [B, D]
        
        # ========== 计算几何损失 (geo_loss) ==========
        # 目标：相同尺度的lateral特征和backbone特征应该相似（去除噪声后应该对齐）
        
        # 将所有特征堆叠成矩阵: [scale_size * batch_size, feature_dim]
        loc_lateral_all = torch.stack(lateral_loc, dim=0)  # [scale_size, batch_size, D]
        loc_lateral_all = loc_lateral_all.view(-1, loc_lateral_all.shape[-1])  # [N, D]
        
        loc_backbone_all = torch.stack(backbone_loc, dim=0)  # [scale_size, batch_size, D]
        loc_backbone_all = loc_backbone_all.view(-1, loc_backbone_all.shape[-1])  # [N, D]
        
        # 归一化特征
        loc_lateral_norm = F.normalize(loc_lateral_all, p=2, dim=1)  # [N, D]
        loc_backbone_norm = F.normalize(loc_backbone_all, p=2, dim=1)  # [N, D]
        
        # 计算所有样本对之间的相似度矩阵
        # 计算相似度矩阵。torch.matmul(a, b.t()) 实际上是将所有样本对在特征维度上进行点积，得到它们的余弦相似度（已归一化后）。
        # 例如，sim_lateral_lateral[i, j] 表示第i个lateral特征与第j个lateral特征的相似度；
        # sim_lateral_backbone[i, j] 表示第i个lateral特征与第j个backbone特征的相似度。
        sim_lateral_lateral = torch.matmul(loc_lateral_norm, loc_lateral_norm.t()) / self.contrastive_temp   # [N, N]
        sim_lateral_backbone = torch.matmul(loc_lateral_norm, loc_backbone_norm.t()) / self.contrastive_temp # [N, N]
        
        # 创建尺度索引和批次索引
        scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)  # [N]
        batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)  # [N]
        
        # 创建正样本掩码：相同尺度且相同batch（lateral和backbone应该对齐）
        scale_mask = (scale_indices.unsqueeze(1) == scale_indices.unsqueeze(0))  # [N, N]
        batch_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0))  # [N, N]
        pos_mask = scale_mask & batch_mask  # [N, N] - 正样本掩码
        
        # 创建负样本掩码
        neg_mask = ~pos_mask  # [N, N] - 负样本掩码
        
        # 计算每个样本的损失
        exp_sim_lateral_backbone = torch.exp(sim_lateral_backbone)  # [N, N]
        # 提取正样本相似度（使用backbone作为正样本）
        pos_sim = exp_sim_lateral_backbone * pos_mask  # [N, N]
        numerators = pos_sim.sum(dim=1)  # [N]
        
        # 提取负样本相似度
        exp_sim_lateral_lateral = torch.exp(sim_lateral_lateral)  # [N, N]
        neg_sim = (exp_sim_lateral_lateral + exp_sim_lateral_backbone) * neg_mask  # [N, N]
        denominators = neg_sim.sum(dim=1) + numerators  # [N]
        
        # 计算几何损失（位置对齐损失）
        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8)
        geo_loss = geo_loss.mean()
        
        return {'geo_loss': geo_loss}
