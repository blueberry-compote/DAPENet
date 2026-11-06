"""Reorder FPN + sparse DCN + hybrid denoising.

- P3: DCT-HPF(alpha) * freq_gate(HPF) + P * spatial_gate(P); add (no cat)
- P4/P5 (index 1, 2): original PositionDenoisingModule unchanged
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
from .reorder_dc_dngeo_light import PositionDenoisingModule, _DCNAlign


# ---------------------------------------------------------------------------
# DCT (same as reorder_dc_fae_light.py)
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
    """Zero upper-left low-frequency block in DCT, then IDCT."""
    _, _, h, w = x.shape
    x_dct = dct_2d(x.float())
    u = torch.arange(h, device=x.device, dtype=torch.float32).view(h, 1)
    v = torch.arange(w, device=x.device, dtype=torch.float32).view(1, w)
    mask = (~((u < alpha * h) & (v < alpha * w))).to(x_dct.dtype)
    x_dct = x_dct * mask.view(1, 1, h, w)
    return idct_2d(x_dct).to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# P3: freq (HPF * gate on HPF) + spatial (P * gate on P)
# ---------------------------------------------------------------------------

class P3FreqSpatialAddModule(nn.Module):
    """P3: HPF(P) * gate(HPF(P)) + P * spatial(P)."""

    def __init__(self, channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.denoise_conv = ConvModule(
            channels, channels, 3, padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        self.freq_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, lateral):
        p = self.denoise_conv(lateral)
        p_hpf = hard_highpass_dct(p, self.alpha)
        p_freq = p_hpf * self.freq_attn(p_hpf)
        p_sp = p * self.spatial_attn(p)
        return p_freq + p_sp


# ---------------------------------------------------------------------------
# Neck
# ---------------------------------------------------------------------------

@NECKS.register_module()
class Reorder_DC_DNGeo_FAE_P3_Lite(FPN):
    """laterals -> top-down(DCN) -> P3 freq+spatial add / P4-P5 DNGeo denoise -> fpn_convs."""

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
        use_denoising=True,
        denoising_weight=1.0,
        denoising_after_topdown=True,
        denoising_levels=None,
        p3_level=0,
        use_p3_fae=True,
        p3_fae_alpha=0.1,
        dcn_kernel_size=3,
        dcn_deform_groups=1,
        dcn_zero_init_offset=True,
        dcn_levels=None,
        dcn_mask_tau=0.03,
        dcn_gate_mode='soft',
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(
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
            init_cfg=init_cfg,
        )

        self.use_denoising = use_denoising
        self.denoising_weight = denoising_weight
        self.denoising_after_topdown = denoising_after_topdown
        self.p3_level = p3_level
        self.use_p3_fae = use_p3_fae

        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusion_steps = used_backbone_levels - 1

        if dcn_levels is None or dcn_levels == 'all':
            self.dcn_levels = list(range(num_fusion_steps))
        else:
            self.dcn_levels = list(dcn_levels)

        # P4/P5: original DNGeo denoising (default indices 1, 2)
        if denoising_levels is None:
            self.denoising_levels = [1, 2]
        else:
            self.denoising_levels = list(denoising_levels)

        if self.use_denoising:
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels,
                num_scales=len(self.lateral_convs),
            )

        if self.use_p3_fae:
            self.p3_fae_module = P3FreqSpatialAddModule(
                out_channels, p3_fae_alpha)

        self.align_modules = nn.ModuleList([
            _DCNAlign(
                channels=out_channels,
                kernel_size=dcn_kernel_size,
                deform_groups=dcn_deform_groups,
                init_zero_offset=dcn_zero_init_offset,
                mask_tau=dcn_mask_tau,
                gate_mode=dcn_gate_mode,
            ) for _ in range(num_fusion_steps)
        ])

    def get_keep_ratio(self):
        keep_ratio = None
        count = 0
        for m in self.align_modules:
            if hasattr(m, 'last_keep_ratio'):
                cur = m.last_keep_ratio
                keep_ratio = cur if keep_ratio is None else (keep_ratio + cur)
                count += 1
        if count > 0:
            return keep_ratio / count
        return self.align_modules[0].offset_and_mask.weight.new_tensor(0.0)

    def _apply_p3_fae(self, laterals):
        i = self.p3_level
        if i >= len(laterals):
            return laterals
        laterals[i] = self.p3_fae_module(laterals[i])
        return laterals

    def _apply_dngeo_denoising(self, laterals):
        denoised = self.position_denoising_module(
            laterals, levels_to_denoise=self.denoising_levels)
        for i in self.denoising_levels:
            if i < len(laterals):
                orig = laterals[i]
                laterals[i] = orig + self.denoising_weight * (
                    denoised[i] - orig)
        return laterals

    def _build_laterals_with_denoising(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if not self.denoising_after_topdown:
            if self.use_p3_fae:
                laterals = self._apply_p3_fae(laterals)
            if self.use_denoising:
                laterals = self._apply_dngeo_denoising(laterals)
        return laterals

    def _build_topdown_and_outputs(self, laterals, inputs):
        pre_topdown_laterals = [l.clone() for l in laterals]
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        if self.denoising_after_topdown:
            if self.use_p3_fae:
                laterals = self._apply_p3_fae(laterals)
            if self.use_denoising:
                laterals = self._apply_dngeo_denoising(laterals)

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

        return tuple(outs), pre_topdown_laterals

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_and_outputs(laterals, inputs)
        return fpn_outs
