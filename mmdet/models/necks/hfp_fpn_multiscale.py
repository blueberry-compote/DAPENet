# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


def _as_tuple(x: Optional[Iterable[int]]) -> Tuple[int, ...]:
    if x is None:
        return tuple()
    if isinstance(x, int):
        return (x,)
    if isinstance(x, tuple):
        return x
    return tuple(x)


class MultiScaleHighFrequencyExtractor(nn.Module):
    """多尺度高频特征提取器。
    
    在不同尺度上提取高频特征，然后融合，能够捕获不同频率范围的信息。
    """
    
    def __init__(self, channels: int, scales: Tuple[int, ...] = (3, 5)):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList()
        for scale in scales:
            # 每个尺度使用不同大小的拉普拉斯算子
            if scale == 3:
                kernel = torch.tensor([
                    [0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]
                ], dtype=torch.float32).view(1, 1, 3, 3)
            elif scale == 5:
                kernel = torch.tensor([
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]
                ], dtype=torch.float32).view(1, 1, 5, 5)
            else:
                # 更大的尺度：使用高斯差分近似
                kernel = self._create_gaussian_laplacian(scale)
            
            self.register_buffer(f'kernel_{scale}', kernel)
            self.convs.append(
                nn.Conv2d(channels, channels, kernel_size=scale, 
                         padding=scale//2, groups=channels, bias=False)
            )
            # 初始化权重
            with torch.no_grad():
                self.convs[-1].weight.copy_(kernel.repeat(channels, 1, 1, 1))
                self.convs[-1].weight.requires_grad = False  # 固定高频提取核
        
        # 融合不同尺度的特征
        self.fusion = nn.Conv2d(channels * len(scales), channels, kernel_size=1)
    
    def _create_gaussian_laplacian(self, size: int):
        """创建高斯-拉普拉斯算子（LoG）。"""
        sigma = size / 6.0
        center = size // 2
        kernel = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = -(1 / (math.pi * sigma**4)) * \
                    (1 - (x**2 + y**2) / (2 * sigma**2)) * \
                    math.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel.view(1, 1, size, size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取多尺度高频特征并融合。"""
        hf_features = []
        for i, scale in enumerate(self.scales):
            hf = self.convs[i](x)
            hf_features.append(hf)
        
        # 融合多尺度特征
        hf_fused = self.fusion(torch.cat(hf_features, dim=1))
        return hf_fused


class EnhancedCBAM(nn.Module):
    """增强的CBAM注意力机制。
    
    结合通道注意力和空间注意力，用于增强特征表示。
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x


class MultiScaleHighFrequencyPerception(BaseModule):
    """多尺度高频感知模块。
    
    核心改进：
    1. 多尺度高频提取（不同频率范围）
    2. 增强的CBAM注意力机制
    3. 渐进式特征融合
    4. 残差连接保留原始信息
    """

    def __init__(
        self,
        channels: int,
        scales: Tuple[int, ...] = (3, 5),
        pool_k: int = 16,
        reduction: int = 16,
        residual_weight: float = 0.3,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.channels = channels
        self.residual_weight = float(residual_weight)

        # 多尺度高频提取器
        self.hf_extractor = MultiScaleHighFrequencyExtractor(channels, scales)

        # 增强的CBAM注意力
        self.cbam = EnhancedCBAM(channels, reduction)

        # 高频特征处理
        self.hf_conv = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False,
        )

        # 渐进式融合
        self.fusion_conv1 = ConvModule(
            channels * 2,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False,
        )
        self.fusion_conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False,
        )

        # 输出卷积
        self.out_conv = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False,
        )

    def forward(self, ci: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ci: [B, C, H, W] 输入特征
        Returns:
            out: [B, C, H, W] 增强后的特征
        """
        # 1. 提取多尺度高频特征
        hf_features = self.hf_extractor(ci)
        
        # 2. 对高频特征应用CBAM注意力
        hf_attended = self.cbam(hf_features)
        
        # 3. 处理高频特征
        hf_processed = self.hf_conv(hf_attended)
        
        # 4. 对原始特征也应用CBAM
        ci_attended = self.cbam(ci)
        
        # 5. 渐进式融合：先融合原始和高频，再进一步处理
        fused = self.fusion_conv1(torch.cat([ci_attended, hf_processed], dim=1))
        fused = self.fusion_conv2(fused)
        
        # 6. 残差连接
        out = ci + self.residual_weight * fused
        out = self.out_conv(out)
        
        return out


@NECKS.register_module()
class HFP_FPN_MultiScale(BaseModule):
    """多尺度高频感知FPN。
    
    相比之前的版本：
    1. 多尺度高频提取（捕获不同频率范围）
    2. 增强的CBAM注意力机制
    3. 渐进式特征融合
    4. 残差连接保留原始信息
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs=False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg: dict = dict(mode='nearest'),
        # HFP-MultiScale params
        hfp_in_indices: Optional[Sequence[int]] = (0, 1),
        hfp_scales: Tuple[int, ...] = (3, 5),
        hfp_pool_k: int = 16,
        hfp_reduction: int = 16,
        hfp_residual_weight: float = 0.3,
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, (list, tuple))

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        # HFP modules
        hfp_in_indices = set(_as_tuple(hfp_in_indices))
        self.hfp_in_indices = hfp_in_indices
        self.hfp_modules = nn.ModuleList()
        for i in range(self.num_ins):
            if i in self.hfp_in_indices:
                self.hfp_modules.append(
                    MultiScaleHighFrequencyPerception(
                        channels=self.in_channels[i],
                        scales=hfp_scales,
                        pool_k=hfp_pool_k,
                        reduction=hfp_reduction,
                        residual_weight=hfp_residual_weight,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
            else:
                self.hfp_modules.append(nn.Identity())

        # FPN lateral / fpn conv
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # extra levels
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    extra_in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    extra_in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    extra_in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        inputs = list(inputs)
        for i in range(len(inputs)):
            if i in self.hfp_in_indices:
                inputs[i] = self.hfp_modules[i](inputs[i])

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg
                )
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
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
