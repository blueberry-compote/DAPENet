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
    # 兼容：有些配置可能误传单个 int（如 (1) 实际是 1）
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


def fft_highpass_filter_radial_sigmoid(
    x: torch.Tensor,
    alpha: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """径向 soft-cutoff 的可导高通（方案 A）。

    在 fftshift 后的频域上构造径向半径 r∈[0,1]（中心低频 r=0，四角高频 r≈1），
    使用 sigmoid((r - alpha)/tau) 生成软高通 mask，实现可学习的截止位置 alpha。

    Args:
        x (Tensor): [B, C, H, W]
        alpha (Tensor): 标量/可 broadcast 的截止半径，建议通过 sigmoid 参数化到 (0,1)。
        tau (float): 过渡带宽（温度），越小越接近硬切。
    """
    if tau <= 0:
        raise ValueError(f'tau must be > 0, got {tau}')

    b, c, h, w = x.shape
    x_f = torch.fft.fft2(x.float(), dim=(-2, -1))
    x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

    # 构造 r 网格（与 fftshift 对齐：中心低频 r=0）
    y = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=torch.float32)
    x_ = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=torch.float32)
    yy, xx = _meshgrid_ij(y, x_)
    r = torch.sqrt(xx * xx + yy * yy) / math.sqrt(2.0)  # [H, W] in [0, 1]

    # 软高通 mask：低频(中心)趋近 0，高频(边缘)趋近 1
    # alpha: 标量 tensor，可学习；tau: float
    mask = torch.sigmoid((r - alpha.to(dtype=r.dtype)) / float(tau))
    x_f = x_f * mask.to(dtype=x_f.dtype)[None, None, :, :]

    x_f = torch.fft.ifftshift(x_f, dim=(-2, -1))
    x_hp = torch.fft.ifft2(x_f, dim=(-2, -1)).real
    return x_hp.to(dtype=x.dtype)


class HighFrequencyPerceptionA(BaseModule):
    """High Frequency Perception (HFP) with learnable radial soft high-pass (方案 A)."""

    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        tau: float = 0.06,
        pool_k: int = 16,
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
        if not (alpha_min < alpha_max):
            raise ValueError(f'alpha_min must be < alpha_max, got {alpha_min}, {alpha_max}')
        if not (alpha_min <= alpha_init <= alpha_max):
            raise ValueError(
                f'alpha_init must be within [alpha_min, alpha_max], got {alpha_init}'
            )
        if tau <= 0:
            raise ValueError(f'tau must be > 0, got {tau}')

        self.channels = channels
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.tau = float(tau)
        self.pool_k = int(pool_k)
        self.cp_groups = int(cp_groups)
        self.use_highpass = bool(use_highpass)

        # learnable alpha via sigmoid(raw_alpha) -> (alpha_min, alpha_max)
        p = (float(alpha_init) - self.alpha_min) / (self.alpha_max - self.alpha_min)
        p = min(max(p, 1e-6), 1 - 1e-6)
        raw_alpha_init = math.log(p / (1 - p))  # logit
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha_init, dtype=torch.float32))

        # CP：GAP/GMP 分支 -> group 1x1 conv -> concat -> group 1x1 conv
        self.cp_conv_gap = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_gmp = nn.Conv2d(
            channels, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )
        self.cp_conv_fuse = nn.Conv2d(
            channels * 2, channels, kernel_size=1, groups=self.cp_groups, bias=True
        )

        # SP：1x1 conv 聚合通道信息得到空间 mask
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

    def _alpha(self) -> torch.Tensor:
        a = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * a

    def forward(self, ci: torch.Tensor) -> torch.Tensor:
        if self.use_highpass:
            fi = fft_highpass_filter_radial_sigmoid(ci, alpha=self._alpha(), tau=self.tau)
        else:
            fi = ci

        b, c, h, w = fi.shape
        k = min(self.pool_k, h, w)
        k = max(1, k)

        # Channel Path (CP)
        f_gap = F.adaptive_avg_pool2d(fi, output_size=(k, k))
        f_gmp = F.adaptive_max_pool2d(fi, output_size=(k, k))
        f_gap = F.relu(f_gap, inplace=False)
        f_gmp = F.relu(f_gmp, inplace=False)
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
class HFP_FPN_A(BaseModule):
    """FPN with learnable radial soft high-pass HFP (方案 A)."""

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
        # HFP-A params
        hfp_in_indices: Optional[Sequence[int]] = (1,), #只对C3做（编号为1）（C2的编号为0……）
        hfp_alpha_init: float = 0.3,
        hfp_alpha_min: float = 0.0,
        hfp_alpha_max: float = 1.0,
        hfp_tau: float = 0.06,
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
                    HighFrequencyPerceptionA(
                        channels=self.in_channels[i],
                        alpha_init=hfp_alpha_init,
                        alpha_min=hfp_alpha_min,
                        alpha_max=hfp_alpha_max,
                        tau=hfp_tau,
                        pool_k=hfp_pool_k,
                        cp_groups=hfp_cp_groups,
                        use_highpass=True,
                        conv_cfg=conv_cfg,
                        norm_cfg=None,
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
