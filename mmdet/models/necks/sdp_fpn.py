# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class _BlockwiseSDPAlign(nn.Module):
    """Block-wise cross-attention alignment (SDP-style).

    Q comes from low-level feature (P_i), K/V come from upsampled high-level feature (Up(P_{i+1})).
    Attention is computed *within each feature block* (pixels-to-pixels inside the block).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, p_i: torch.Tensor, p_ip1_up: torch.Tensor, block_hw: Tuple[int, int]) -> torch.Tensor:
        """Return delta feature aligned to p_i's space.

        Args:
            p_i: (B, C, H, W)
            p_ip1_up: (B, C, H, W) - already upsampled to match p_i spatial size
            block_hw: (block_h, block_w), typically (H_top, W_top) like P5 resolution
        """
        

        b, c, h, w = p_i.shape
        block_h, block_w = block_hw
        if block_h <= 0 or block_w <= 0:
            raise ValueError(f"Invalid block_hw={block_hw}")

        pad_h = (block_h - (h % block_h)) % block_h
        pad_w = (block_w - (w % block_w)) % block_w
        if pad_h or pad_w:
            p_i = F.pad(p_i, (0, pad_w, 0, pad_h))
            p_ip1_up = F.pad(p_ip1_up, (0, pad_w, 0, pad_h))

        _, _, hp, wp = p_i.shape
        nh = hp // block_h
        nw = wp // block_w
        n = nh * nw
        l = block_h * block_w

        q = self.q_proj(p_i)
        k = self.k_proj(p_ip1_up)
        v = self.v_proj(p_ip1_up)

        # (B, C, Hp, Wp) -> (B, n, L, C)
        def _to_blocks(x: torch.Tensor) -> torch.Tensor:
            x = x.view(b, c, nh, block_h, nw, block_w)
            x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nh, nw, bh, bw, C)
            return x.view(b, n, l, c)  # (B, n, L, C)

        q_b = _to_blocks(q)
        k_b = _to_blocks(k)
        v_b = _to_blocks(v)

        # (B, n, L, L)
        attn = torch.matmul(q_b, k_b.transpose(-1, -2)) / math.sqrt(c)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_b)  # (B, n, L, C)

        # back to (B, C, Hp, Wp)
        out = out.view(b, nh, nw, block_h, block_w, c)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)

        out = self.out_proj(out)

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


@NECKS.register_module()
class FPNWithSDP(FPN):
    """FPN with SDP-style block-wise cross-attention alignment in top-down fusion.

    Default behavior:
    - Use B strategy: align Up(P_{i+1}) first, then add to P_i
    - Apply only to the topmost fusion (usually P5 -> P4) for best cost/perf
    """

    def __init__(
        self,
        *args,
        sdp_enabled: bool = True,
        sdp_num_topdown_levels: int = 1,
        # NOTE: do NOT set to 0.0 by default, otherwise the SDP branch receives
        # almost no gradient through the residual scaling at the beginning.
        sdp_init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sdp_enabled = sdp_enabled
        self.sdp_num_topdown_levels = int(sdp_num_topdown_levels)

        # number of top-down fusions equals used_backbone_levels - 1
        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusions = max(0, used_backbone_levels - 1)

        self.sdp_align = _BlockwiseSDPAlign(self.out_channels)
        self.sdp_gammas = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(sdp_init_scale))) for _ in range(num_fusions)]
        )

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        if used_backbone_levels == 0:
            return tuple()

        # use topmost lateral spatial size as base block size (e.g., P5 resolution)
        base_hw = laterals[-1].shape[2:]

        # build top-down path
        fusion_step = 0  # 0 means topmost fusion (P5->P4 if using 4 levels)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            # Guard: in case scale_factor leads to off-by-one, force match.
            if up.shape[2:] != laterals[i - 1].shape[2:]:
                up = F.interpolate(up, size=laterals[i - 1].shape[2:], mode="nearest")

            # SDP alignment: align upsampled high-level feature to P_i space before addition (strategy B)
            if self.sdp_enabled and fusion_step < self.sdp_num_topdown_levels:
                gamma = self.sdp_gammas[i - 1] if (i - 1) < len(self.sdp_gammas) else 0.0
                delta = self.sdp_align(laterals[i - 1], up, block_hw=base_hw)
                up = up + gamma * delta

            laterals[i - 1] = laterals[i - 1] + up
            fusion_step += 1

        # build outputs (original levels)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # add extra levels (keep identical to original FPN)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
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
