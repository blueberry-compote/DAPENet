"""Reorder FPN with DCN alignment and Frequency-aware Enhancement (FAE).

Spatial branch:  P_base -> DCT -> hard HPF -> IDCT -> Conv7x7 -> A_sp
Channel branch:  P_base -> GAP -> MLP -> A_ch
Output:          P_tilde = P_base * A_sp * A_ch
"""
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN


# ---------------------------------------------------------------------------
# DCT utilities (self-contained, no dependency on other neck files)
# ---------------------------------------------------------------------------

def _dct_1d(x, norm=None):
    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().view(-1, n)
    v = torch.cat([x[:, 0::2], x[:, 1::2].flip([1])], dim=1)
    vc = torch.fft.fft(v, dim=1)
    k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    v_out = vc.real * torch.cos(k) - vc.imag * torch.sin(k)
    if norm == 'ortho':
        v_out[:, 0] /= math.sqrt(n) * 2
        v_out[:, 1:] /= math.sqrt(n / 2) * 2
    return (2 * v_out).view(*x_shape)


def _idct_1d(x, norm=None):
    x_shape = x.shape
    n = x_shape[-1]
    x_v = x.contiguous().view(-1, n) / 2
    if norm == 'ortho':
        x_v[:, 0] *= math.sqrt(n) * 2
        x_v[:, 1:] *= math.sqrt(n / 2) * 2
    k = torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    v_r = x_v * torch.cos(k) - torch.cat(
        [x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1) * torch.sin(k)
    v_i = x_v * torch.sin(k) + torch.cat(
        [x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1) * torch.cos(k)
    v = torch.complex(v_r, v_i)
    out = torch.fft.ifft(v, dim=1).real
    result = out.new_zeros(out.shape)
    result[:, 0::2] = out[:, :n - (n // 2)]
    result[:, 1::2] = out.flip([1])[:, :n // 2]
    return result.view(*x_shape)


def dct_2d(x):
    x1 = _dct_1d(x)
    x2 = _dct_1d(x1.transpose(-1, -2))
    return x2.transpose(-1, -2)


def idct_2d(x):
    x1 = _idct_1d(x)
    x2 = _idct_1d(x1.transpose(-1, -2))
    return x2.transpose(-1, -2)


def hard_highpass_dct(x, alpha):
    """Eq.(11-13): zero out upper-left low-frequency block, then IDCT."""
    b, c, h, w = x.shape
    x_dct = dct_2d(x.float())
    u = torch.arange(h, device=x.device, dtype=torch.float32).view(h, 1)
    v = torch.arange(w, device=x.device, dtype=torch.float32).view(1, w)
    mask = (~((u < alpha * h) & (v < alpha * w))).to(x_dct.dtype)
    x_dct = x_dct * mask.view(1, 1, h, w)
    return idct_2d(x_dct).to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# FAE module
# ---------------------------------------------------------------------------

class FrequencyAwareEnhancementModule(nn.Module):
    """Frequency-guided spatial + semantic channel enhancement."""

    def __init__(self, channels, num_scales, alpha=0.25, channel_ratio=16):
        super().__init__()
        self.alpha = alpha
        mid = max(channels // channel_ratio, 1)

        self.base_convs = nn.ModuleList([
            ConvModule(
                channels, channels, 3, padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True),
            ) for _ in range(num_scales)
        ])
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 1, 7, padding=3),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, mid, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channels, 1),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])

    def forward(self, feats, levels=None):
        if levels is None:
            levels = list(range(len(feats)))
        outs = []
        for i, feat in enumerate(feats):
            if i not in levels:
                outs.append(feat)
                continue
            p_base = self.base_convs[i](feat)
            p_freq = hard_highpass_dct(p_base, self.alpha)
            a_sp = self.spatial_attentions[i](p_freq)
            a_ch = self.channel_attentions[i](p_base)
            outs.append(p_base * a_sp * a_ch)
        return outs


# ---------------------------------------------------------------------------
# DCN alignment
# ---------------------------------------------------------------------------

class DCNAlign(nn.Module):

    def __init__(self, channels, kernel_size=3, deform_groups=1,
                 zero_init_offset=True, mask_tau=0.05, gate_mode='soft'):
        super().__init__()
        self.mask_tau = mask_tau
        self.gate_mode = gate_mode
        k2 = kernel_size * kernel_size
        self.offset_dim = 2 * k2 * deform_groups
        self.mask_dim = k2 * deform_groups
        self.offset_and_mask = nn.Conv2d(
            channels * 2, self.offset_dim + self.mask_dim,
            kernel_size, padding=kernel_size // 2)
        self.dcn = ModulatedDeformConv2d(
            channels, channels, kernel_size, padding=kernel_size // 2,
            deform_groups=deform_groups, bias=True)
        normal_init(self.dcn, 0, 0.01)
        if zero_init_offset:
            constant_init(self.offset_and_mask, 0)
        else:
            normal_init(self.offset_and_mask, 0, 0.01)

    def forward(self, p_i, p_up):
        out = self.offset_and_mask(torch.cat([p_i, p_up], dim=1))
        offset = out[:, :self.offset_dim]
        mask = torch.sigmoid(out[:, self.offset_dim:])
        self.last_mask = mask
        keep = (mask > self.mask_tau).to(mask.dtype)
        if self.gate_mode == 'soft':
            mask_sparse = keep * mask
        else:
            mask_sparse = keep - mask.detach() + mask
        mask_sparse = mask_sparse.to(dtype=p_up.dtype, device=p_up.device)
        self.last_keep_ratio = keep.float().mean()
        return self.dcn(p_up.contiguous(), offset, mask_sparse)


# ---------------------------------------------------------------------------
# Neck
# ---------------------------------------------------------------------------

@NECKS.register_module()
class Reorder_DC_FAE_Lite(FPN):

    def __init__(
        self,
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
        use_fae=True,
        fae_weight=1.0,
        fae_after_topdown=True,
        fae_levels=None,
        fae_alpha=0.25,
        fae_channel_ratio=16,
        dcn_kernel_size=3,
        dcn_deform_groups=1,
        dcn_zero_init_offset=True,
        dcn_levels=None,
        dcn_mask_tau=0.03,
        dcn_gate_mode='soft',
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(
            in_channels, out_channels, num_outs,
            start_level=start_level, end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
            upsample_cfg=upsample_cfg, init_cfg=init_cfg)

        self.use_fae = use_fae
        self.fae_weight = fae_weight
        self.fae_after_topdown = fae_after_topdown
        n_lat = self.backbone_end_level - self.start_level
        n_fuse = n_lat - 1

        self.dcn_levels = (list(range(n_fuse)) if dcn_levels in (None, 'all')
                           else list(dcn_levels))
        self.fae_levels = [0, 1] if fae_levels is None else list(fae_levels)

        if use_fae:
            self.fae = FrequencyAwareEnhancementModule(
                out_channels, n_lat, fae_alpha, fae_channel_ratio)

        self.dcn_aligns = nn.ModuleList([
            DCNAlign(out_channels, dcn_kernel_size, dcn_deform_groups,
                     dcn_zero_init_offset, dcn_mask_tau, dcn_gate_mode)
            for _ in range(n_fuse)
        ])

    def get_keep_ratio(self):
        vals = [m.last_keep_ratio for m in self.dcn_aligns
                if hasattr(m, 'last_keep_ratio')]
        if not vals:
            return self.dcn_aligns[0].offset_and_mask.weight.new_tensor(0.)
        return sum(vals) / len(vals)

    def _apply_fae(self, laterals):
        enhanced = self.fae(laterals, self.fae_levels)
        for i in self.fae_levels:
            if i < len(laterals):
                o = laterals[i]
                laterals[i] = o + self.fae_weight * (enhanced[i] - o)
        return laterals

    def _laterals(self, inputs):
        lat = [c(inputs[i + self.start_level])
               for i, c in enumerate(self.lateral_convs)]
        if self.use_fae and not self.fae_after_topdown:
            lat = self._apply_fae(lat)
        return lat

    def _topdown(self, laterals, inputs):
        n = len(laterals)
        for i in range(n - 1, 0, -1):
            up = F.interpolate(laterals[i], laterals[i - 1].shape[2:],
                               **self.upsample_cfg)
            if (i - 1) in self.dcn_levels:
                laterals[i - 1] = laterals[i - 1] + self.dcn_aligns[i - 1](
                    laterals[i - 1], up)
            else:
                laterals[i - 1] = laterals[i - 1] + up

        if self.use_fae and self.fae_after_topdown:
            laterals = self._apply_fae(laterals)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(n)]
        if self.num_outs <= n:
            return tuple(outs)

        if not self.add_extra_convs:
            for _ in range(self.num_outs - n):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            return tuple(outs)

        src_map = dict(
            on_input=inputs[self.backbone_end_level - 1],
            on_lateral=laterals[-1],
            on_output=outs[-1],
        )
        extra = src_map.get(self.add_extra_convs)
        if extra is None:
            raise NotImplementedError
        outs.append(self.fpn_convs[n](extra))
        for j in range(n + 1, self.num_outs):
            src = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
            outs.append(self.fpn_convs[j](src))
        return tuple(outs)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        return self._topdown(self._laterals(inputs), inputs)
