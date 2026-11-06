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


def _meshgrid_ij(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容不同 PyTorch 版本的 meshgrid(indexing='ij')."""
    try:
        yy, xx = torch.meshgrid(y, x, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(y, x)
    return yy, xx


def spatial_highpass_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """使用空间卷积算子提取高频特征（替代FFT）。
    
    使用拉普拉斯算子或Sobel算子提取边缘/高频信息，更温和且可导。
    
    Args:
        x (Tensor): [B, C, H, W]
        kernel_size (int): 卷积核大小，3或5
    """
    if kernel_size == 3:
        # 拉普拉斯算子（3x3）
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    else:  # kernel_size == 5
        # 扩展拉普拉斯算子（5x5）
        laplacian_kernel = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=x.dtype, device=x.device).view(1, 1, 5, 5)
    
    b, c, h, w = x.shape
    # 对每个通道应用拉普拉斯算子
    high_freq = F.conv2d(
        x.view(b * c, 1, h, w),
        laplacian_kernel,
        padding=kernel_size // 2
    ).view(b, c, h, w)
    
    return high_freq


class ResidualHighFrequencyPerception(BaseModule):
    """残差增强型高频感知模块。
    
    核心思想：
    1. 使用空间卷积算子（拉普拉斯）提取高频特征，比FFT更温和
    2. 保留原始特征，高频特征作为增强（残差连接）
    3. 改进的注意力机制：自适应融合CP和SP
    4. 多尺度高频特征融合
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        pool_k: int = 16,
        cp_groups: int = 1,
        use_highpass: bool = True,
        residual_weight: float = 0.2,  # 高频增强的权重
        use_adaptive_fusion: bool = True,  # 是否使用自适应融合CP和SP
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if cp_groups < 1:
            raise ValueError(f'cp_groups must be >= 1, got {cp_groups}')
        if channels % cp_groups != 0:
            raise ValueError(
                f'channels ({channels}) must be divisible by cp_groups ({cp_groups})'
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_k = int(pool_k)
        self.cp_groups = int(cp_groups)
        self.use_highpass = bool(use_highpass)
        self.residual_weight = float(residual_weight)
        self.use_adaptive_fusion = bool(use_adaptive_fusion)

        # Channel Path (CP): 改进的通道注意力
        self.cp_conv_gap = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_gmp = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_fuse = nn.Conv2d(
            channels * 2, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )

        # Spatial Path (SP): 改进的空间注意力
        self.sp_conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=True)
        self.sp_conv2 = nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1, bias=True)

        # 自适应融合权重（如果启用）
        if self.use_adaptive_fusion:
            self.fusion_weight = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 2, 1),
                nn.Softmax(dim=1)
            )

        # 高频特征处理卷积
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

        # 输出融合卷积
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
        # 1. 提取高频特征（使用空间卷积算子）
        if self.use_highpass:
            fi = spatial_highpass_filter(ci, kernel_size=self.kernel_size)
            # 归一化高频特征
            fi = F.layer_norm(fi, fi.shape[1:])
        else:
            fi = ci

        b, c, h, w = fi.shape
        k = min(self.pool_k, h, w)
        k = max(1, k)

        # 2. Channel Path (CP)
        f_gap = F.adaptive_avg_pool2d(fi, output_size=(k, k))
        f_gmp = F.adaptive_max_pool2d(fi, output_size=(k, k))
        f_gap = F.relu(f_gap, inplace=False)
        f_gmp = F.relu(f_gmp, inplace=False)
        v_gap = f_gap.sum(dim=(2, 3), keepdim=True)
        v_gmp = f_gmp.sum(dim=(2, 3), keepdim=True)
        s_gap = self.cp_conv_gap(v_gap)
        s_gmp = self.cp_conv_gmp(v_gmp)
        u_cp = torch.sigmoid(self.cp_conv_fuse(torch.cat([s_gap, s_gmp], dim=1)))

        # 3. Spatial Path (SP): 改进的两层卷积
        u_sp = self.sp_conv1(fi)
        u_sp = F.relu(u_sp, inplace=True)
        u_sp = torch.sigmoid(self.sp_conv2(u_sp))  # [B, 1, H, W]

        # 4. 应用注意力到高频特征
        hf_cp = fi * u_cp
        hf_sp = fi * u_sp
        hf_attended = hf_cp + hf_sp

        # 5. 处理高频特征
        hf_processed = self.hf_conv(hf_attended)

        # 6. 残差融合：原始特征 + 加权高频增强
        if self.use_adaptive_fusion:
            # 自适应权重：根据特征内容决定CP和SP的权重
            fusion_weights = self.fusion_weight(ci)  # [B, 2, 1, 1]
            cp_weight = fusion_weights[:, 0:1, :, :]
            sp_weight = fusion_weights[:, 1:2, :, :]
            out_cp = ci * u_cp * cp_weight
            out_sp = ci * u_sp * sp_weight
            out_attended = out_cp + out_sp
        else:
            # 简单融合
            out_cp = ci * u_cp
            out_sp = ci * u_sp
            out_attended = out_cp + out_sp

        # 7. 残差连接：原始特征 + 高频增强
        out = out_attended + self.residual_weight * hf_processed
        out = self.out_conv(out)

        return out


@NECKS.register_module()
class HFP_FPN_Residual(BaseModule):
    """残差增强型高频感知FPN。
    
    相比之前的版本：
    1. 使用空间卷积算子替代FFT，更温和
    2. 保留原始信息（残差连接）
    3. 改进的注意力机制
    4. 自适应融合权重
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
        # HFP-Residual params
        hfp_in_indices: Optional[Sequence[int]] = (0, 1),
        hfp_kernel_size: int = 3,
        hfp_pool_k: int = 16,
        hfp_cp_groups: int = 1,
        hfp_residual_weight: float = 0.2,
        hfp_use_adaptive_fusion: bool = True,
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
                    ResidualHighFrequencyPerception(
                        channels=self.in_channels[i],
                        kernel_size=hfp_kernel_size,
                        pool_k=hfp_pool_k,
                        cp_groups=hfp_cp_groups,
                        use_highpass=True,
                        residual_weight=hfp_residual_weight,
                        use_adaptive_fusion=hfp_use_adaptive_fusion,
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
