import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


class SPAMFusion(nn.Module):
    """在 FPN top-down 融合处使用的 SPAM 融合模块。

    按你的描述实现：
    1) 将 p(i) 与 upsample(p(i+1)) 在通道维 cat
    2) 从 cat 特征分别求通道注意力与空间注意力
    3) 用通道注意力分别乘到 p(i)、p(i+1)
    4) 两者相加后乘空间注意力
    5) 再加上 p(i)、p(i+1) 的残差（相加）输出
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        assert spatial_kernel in (3, 5, 7), 'spatial_kernel 建议使用 3/5/7'
        self.channels = channels

        cat_channels = channels * 2
        mid = max(1, cat_channels // reduction)

        # 通道注意力：对 cat 做 GAP -> 1x1 -> ReLU -> 1x1 -> Sigmoid，输出 2C 通道权重
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cat_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, cat_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 空间注意力：对 cat 做 channel-wise avg/max -> concat(2, H, W) -> conv -> Sigmoid
        padding = spatial_kernel // 2
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, p_i, p_ip1):
        # p_i/p_ip1: [B, C, H, W]，其中 p_ip1 已对齐到 p_i 的空间尺寸
        x = torch.cat([p_i, p_ip1], dim=1)  # [B, 2C, H, W]

        # channel attention
        ca = self.channel_att(x)  # [B, 2C, 1, 1]
        ca_i, ca_ip1 = ca.chunk(2, dim=1)

        p_i_ca = p_i * ca_i
        p_ip1_ca = p_ip1 * ca_ip1
        fused = p_i_ca + p_ip1_ca

        # spatial attention（从 cat 特征提取）
        sa_in = torch.cat(
            [torch.mean(x, dim=1, keepdim=True),
             torch.amax(x, dim=1, keepdim=True)],
            dim=1,
        )
        sa = self.spatial_att(sa_in)  # [B, 1, H, W]

        out = fused * sa
        out = out + (p_i + p_ip1)  # residual
        return out


@NECKS.register_module()
class SPAM_FPN(BaseModule):
    """标准 FPN 结构，在 top-down 融合处用 SPAMFusion 替代上采样相加。"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=3,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='nearest'),
                 spam_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
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
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        if spam_cfg is None:
            spam_cfg = dict(reduction=16, spatial_kernel=7)

        # 在每个 top-down 融合处插入 SPAMFusion（数量 = used_backbone_levels - 1）
        used_backbone_levels = self.backbone_end_level - self.start_level
        self.spam_fusions = nn.ModuleList()
        for _ in range(used_backbone_levels - 1):
            self.spam_fusions.append(SPAMFusion(out_channels, **spam_cfg))

        # extra convs（保持与标准 FPN 一致）
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
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path with SPAM fusion
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            laterals[i - 1] = self.spam_fusions[i - 1](laterals[i - 1], up)

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
