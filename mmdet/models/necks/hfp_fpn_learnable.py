# Copyright (c) OpenMMLab. All rights reserved.
"""HFP with learnable / adaptive alpha for highpass cutoff (可学习/自适应 alpha)."""
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


def fft_highpass_filter_adaptive(
    x: torch.Tensor,
    alpha: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """可导的 FFT 高通滤波，alpha 为 per-sample 张量 [B] 或 [B,1,1,1]。

    使用软矩形掩码（smooth rectangular mask）使 alpha 可微，梯度可回传到 alpha 预测器。
    掩码形式：中心矩形区域内衰减、区域外保留，过渡由 temperature 控制。

    Args:
        x: [B, C, H, W]
        alpha: [B] 或 [B,1,1,1]，范围建议 [0,1]，每样本一个 cutoff 比例
        temperature: 软掩码过渡陡峭程度，越小越接近硬截断
    """
    B, C, H, W = x.shape
    alpha = alpha.view(B, 1, 1).to(x.dtype).to(x.device)
    alpha = alpha.clamp(0.0, 1.0)

    x_f = torch.fft.fft2(x.float(), dim=(-2, -1))
    x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

    cy, cx = H / 2.0, W / 2.0
    yy = torch.arange(H, device=x.device, dtype=torch.float32) - cy
    xx = torch.arange(W, device=x.device, dtype=torch.float32) - cx
    dist_h = yy.view(H, 1).abs().expand(H, W)
    dist_w = xx.view(1, W).abs().expand(H, W)
    half_h = alpha * (H / 2.0)
    half_w = alpha * (W / 2.0)
    d_h = dist_h.unsqueeze(0) - half_h
    d_w = dist_w.unsqueeze(0) - half_w
    inner = torch.sigmoid(-d_h / temperature) * torch.sigmoid(-d_w / temperature)
    mask = 1.0 - inner

    x_f = x_f * mask.unsqueeze(1)
    x_f = torch.fft.ifftshift(x_f, dim=(-2, -1))
    x_hp = torch.fft.ifft2(x_f, dim=(-2, -1)).real
    return x_hp.to(dtype=x.dtype)


class HighFrequencyPerceptionLearnable(BaseModule):
    """HFP 模块，高通 cutoff 由网络预测的 alpha 决定（自适应 alpha）。"""

    def __init__(
        self,
        channels: int,
        init_alpha: float = 0.25,
        pool_k: int = 16,
        cp_groups: int = 1,
        use_highpass: bool = True,
        highpass_temperature: float = 2.0,
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
        self.init_alpha = float(init_alpha)
        self.pool_k = int(pool_k)
        self.cp_groups = int(cp_groups)
        self.use_highpass = bool(use_highpass)
        self.highpass_temperature = float(highpass_temperature)

        # 自适应 alpha 预测器：从输入特征预测每样本的 alpha in [0,1]
        self.alpha_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        self._init_alpha_predictor()

        self.cp_conv_gap = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_gmp = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_fuse = nn.Conv2d(
            channels * 2, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.sp_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
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

    def _init_alpha_predictor(self) -> None:
        """初始化 alpha 预测器，使初始预测接近 init_alpha。"""
        for m in self.alpha_predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # sigmoid(bias) ≈ init_alpha => bias = logit(init_alpha)
                    init_alpha = max(1e-4, min(1 - 1e-4, self.init_alpha))
                    m.bias.data.fill_(math.log(init_alpha / (1.0 - init_alpha)))

    def forward(self, ci: torch.Tensor) -> torch.Tensor:
        if self.use_highpass:
            alpha = self.alpha_predictor(ci)
            fi = fft_highpass_filter_adaptive(
                ci, alpha, temperature=self.highpass_temperature
            )
        else:
            fi = ci

        b, c, h, w = fi.shape
        k = min(self.pool_k, h, w)
        k = max(1, k)

        f_gap = F.adaptive_avg_pool2d(fi, output_size=(k, k))
        f_gmp = F.adaptive_max_pool2d(fi, output_size=(k, k))
        f_gap = F.relu(f_gap, inplace=False)
        f_gmp = F.relu(f_gmp, inplace=False)

        v_gap = f_gap.sum(dim=(2, 3), keepdim=True)
        v_gmp = f_gmp.sum(dim=(2, 3), keepdim=True)
        s_gap = self.cp_conv_gap(v_gap)
        s_gmp = self.cp_conv_gmp(v_gmp)
        u_cp = torch.sigmoid(self.cp_conv_fuse(torch.cat([s_gap, s_gmp], dim=1)))
        u_sp = torch.sigmoid(self.sp_conv(fi))

        out_cp = ci * u_cp
        out_sp = ci * u_sp
        out = out_cp + out_sp
        out = self.out_conv(out)
        return out


@NECKS.register_module()
class HFP_FPN_Learnable(BaseModule):
    """带可学习/自适应 alpha 的 HFP-FPN：每层 HFP 的 alpha 由特征预测。"""

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
        hfp_in_indices: Optional[Sequence[int]] = (0, 1),
        hfp_init_alpha: float = 0.25,
        hfp_pool_k: int = 16,
        hfp_cp_groups: int = 1,
        hfp_highpass_temperature: float = 2.0,
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

        hfp_in_indices = set(_as_tuple(hfp_in_indices))
        self.hfp_in_indices = hfp_in_indices
        self.hfp_modules = nn.ModuleList()
        for i in range(self.num_ins):
            if i in self.hfp_in_indices:
                self.hfp_modules.append(
                    HighFrequencyPerceptionLearnable(
                        channels=self.in_channels[i],
                        init_alpha=hfp_init_alpha,
                        pool_k=hfp_pool_k,
                        cp_groups=hfp_cp_groups,
                        use_highpass=True,
                        highpass_temperature=hfp_highpass_temperature,
                        conv_cfg=conv_cfg,
                        norm_cfg=None,
                        act_cfg=None,
                    )
                )
            else:
                self.hfp_modules.append(nn.Identity())

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
