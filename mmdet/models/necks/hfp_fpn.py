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
    if isinstance(x, tuple):
        return x
    return tuple(x)


def fft_highpass_filter(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """用 2D FFT 实现的高通滤波器（可导）。

    论文里 Eq.(2) 的高通滤波是在 DCT 频谱的左上角抑制低频；这里用 FFT 实现，
    采用 fftshift 将低频移到频谱中心，然后在中心矩形区域置 0，其余置 1，
    达到“抑制低频、保留高频”的效果。

    Args:
        x (Tensor): [B, C, H, W]
        alpha (float): 低频抑制带宽比例，取值 [0, 1]。
            - alpha=0: 不抑制（输出等于输入）
            - alpha=1: 全抑制（输出全 0）
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f'alpha must be in [0, 1], got {alpha}')

    if alpha == 0.0:
        return x
    if alpha == 1.0:
        return torch.zeros_like(x)

    b, c, h, w = x.shape

    # torch.fft 对 fp16 支持在不同设备/版本上不完全一致，统一转 fp32 再还原。
    x_f = torch.fft.fft2(x.float(), dim=(-2, -1))
    x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

    # 中心低频矩形大小（至少 1，避免空 slice 导致行为不一致）
    cut_h = max(1, int(round(alpha * h)))
    cut_w = max(1, int(round(alpha * w)))
    h0 = (h - cut_h) // 2
    w0 = (w - cut_w) // 2
    h1 = h0 + cut_h
    w1 = w0 + cut_w

    mask = torch.ones((h, w), device=x.device, dtype=x_f.dtype)
    mask[h0:h1, w0:w1] = 0
    x_f = x_f * mask[None, None, :, :]

    x_f = torch.fft.ifftshift(x_f, dim=(-2, -1))
    x_hp = torch.fft.ifft2(x_f, dim=(-2, -1)).real
    return x_hp.to(dtype=x.dtype)


class HighFrequencyPerception(BaseModule):
    """High Frequency Perception (HFP) module.

    结构对应你提供的 3.1 描述：
    - 高频响应生成器：对 Ci 做 FFT 高通滤波得到 Fi
    - Channel Path (CP)：在 Fi 上计算通道权重 uCP (C×1×1)
    - Spatial Path (SP)：在 Fi 上计算空间权重 uSP (1×H×W)
    - 对 Ci 分别施加通道/空间权重，结果相加后 3×3 卷积得到 Ci^HFP
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        pool_k: int = 16,  # 通道池化的目标宽高（将特征图下采样到 pool_k×pool_k，常用于空间注意力或高频特征提取）
        cp_groups: int = 1,
        use_highpass: bool = True,
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
        self.alpha = float(alpha)
        self.pool_k = int(pool_k)
        self.cp_groups = int(cp_groups)
        self.use_highpass = bool(use_highpass)

        # CP：分别对 GAP/GMP 分支做 1×1 group conv，然后 concat 再做 1×1 group conv
        self.cp_conv_gap = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_gmp = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_fuse = nn.Conv2d(
            channels * 2, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )

        # SP：1×1 conv 聚合通道信息得到空间 mask
        self.sp_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        # 输出 3×3 conv（按论文描述：加和后再 3×3 卷积）
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
        # ci: [B, C, H, W]
        if self.use_highpass:
            fi = fft_highpass_filter(ci, alpha=self.alpha)
        else:
            fi = ci

        b, c, h, w = fi.shape
        k = min(self.pool_k, h, w)
        k = max(1, k)

        # Channel Path (CP)
        f_gap = F.adaptive_avg_pool2d(fi, output_size=(k, k))
        f_gmp = F.adaptive_max_pool2d(fi, output_size=(k, k))

        # 保留正值（文中 σ 为 ReLU）
        f_gap = F.relu(f_gap, inplace=False)
        f_gmp = F.relu(f_gmp, inplace=False)

        # “sum across each channel” => 对空间求和，得到 [B, C, 1, 1]
        v_gap = f_gap.sum(dim=(2, 3), keepdim=True)
        v_gmp = f_gmp.sum(dim=(2, 3), keepdim=True)

        s_gap = self.cp_conv_gap(v_gap)
        s_gmp = self.cp_conv_gmp(v_gmp)
        u_cp = torch.sigmoid(self.cp_conv_fuse(torch.cat([s_gap, s_gmp], dim=1)))

        # Spatial Path (SP)
        u_sp = torch.sigmoid(self.sp_conv(fi))  # [B, 1, H, W]

        out_cp = ci * u_cp
        out_sp = ci * u_sp
        out = out_cp + out_sp
        out = self.out_conv(out)
        return out


@NECKS.register_module()
class HFP_FPN(BaseModule):
    """带 High Frequency Perception (HFP) 的 FPN。

    实现方式：
    - 对指定 backbone 输入层（默认 indices=(0,1) 即 {C2,C3}/{P2,P3}）先做 HFP
    - 其余部分保持标准 FPN：lateral 1×1 + top-down 相加 + 3×3 fpn conv + extra levels
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
        # HFP 相关
        hfp_in_indices: Optional[Sequence[int]] = (0, 1),
        hfp_alpha: float = 0.25,
        hfp_pool_k: int = 16,
        hfp_cp_groups: int = 1,
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

        # HFP modules（按输入层 index 对齐）
        hfp_in_indices = set(_as_tuple(hfp_in_indices))
        self.hfp_in_indices = hfp_in_indices
        self.hfp_modules = nn.ModuleList()
        for i in range(self.num_ins):
            if i in self.hfp_in_indices:
                self.hfp_modules.append(
                    HighFrequencyPerception(
                        channels=self.in_channels[i],
                        alpha=hfp_alpha,
                        pool_k=hfp_pool_k,
                        cp_groups=hfp_cp_groups,
                        use_highpass=True,  # 论文提示：高频提取仅在 {P2,P3}
                        conv_cfg=conv_cfg,
                        norm_cfg=None,  # 注意力/滤波更稳，默认不加 norm
                        act_cfg=None,
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

        # extra levels（与标准 FPN 对齐）
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

        # 先对指定输入层做 HFP（不改变分辨率/通道数）
        inputs = list(inputs)
        for i in range(len(inputs)):
            if i in self.hfp_in_indices:
                inputs[i] = self.hfp_modules[i](inputs[i])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
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

        # build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # add extra levels
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
