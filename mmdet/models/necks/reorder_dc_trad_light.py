import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import auto_fp16
from typing import List, Optional, Union

from ..builder import NECKS
from .fpn import FPN


class _TradDCNAlign(nn.Module):
    """传统 DCNv2 对齐：sigmoid mask，无动态稀疏门控。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        deform_groups: int = 1,
        init_zero_offset: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups

        k2 = kernel_size * kernel_size
        self.offset_dim = 2 * k2 * deform_groups
        self.mask_dim = 1 * k2 * deform_groups

        self.offset_and_mask = nn.Conv2d(
            channels * 2,
            self.offset_dim + self.mask_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.dcn = ModulatedDeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            deform_groups=deform_groups,
            bias=True,
        )
        self._init_weights(init_zero_offset)

    def _init_weights(self, init_zero_offset):
        normal_init(self.dcn, mean=0, std=0.01)
        if init_zero_offset:
            constant_init(self.offset_and_mask, 0)
        else:
            normal_init(self.offset_and_mask, mean=0, std=0.01)

    def forward(self, p_i, p_ip1_up):
        offset_and_mask = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        offset = offset_and_mask[:, :self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask)
        return aligned


@NECKS.register_module()
class Reorder_DC_Trad_Lite(FPN):
    """无位置去噪的 Reorder FPN，top-down 融合使用传统 DCNv2 对齐。"""

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
        upsample_cfg=dict(mode="nearest"),
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        dcn_levels: Optional[Union[List[int], str]] = None,
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
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

        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusion_steps = used_backbone_levels - 1

        if dcn_levels is None or dcn_levels == "all":
            self.dcn_levels = list(range(num_fusion_steps))
        else:
            self.dcn_levels = list(dcn_levels)

        self.align_modules = nn.ModuleList([
            _TradDCNAlign(
                channels=out_channels,
                kernel_size=dcn_kernel_size,
                deform_groups=dcn_deform_groups,
                init_zero_offset=dcn_zero_init_offset,
            ) for _ in range(num_fusion_steps)
        ])

    def _build_topdown_and_outputs(self, laterals, inputs):
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

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

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        return self._build_topdown_and_outputs(laterals, inputs)
