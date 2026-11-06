import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS

from mmdet.models.plugins.guided_lsk_plugin import GuidedLSK


@NECKS.register_module()
class GuidedLSKFPN(BaseModule):
    """
    GuidedLSKFPN: 在 lateral conv 之后，对前两层 lateral (P3,P4) 使用 GuidedLSK（用上一层 lateral 引导）
    guided_lsk_cfg 是一个 dict，会被传入 GuidedLSK 的构造器（可以包含 use_channel_guide/use_spatial_guide/reduction）
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 guided_lsk_cfg=None,
                 init_cfg=None,
                 **kwargs):  # 兼容多余关键字，避免 config 中的意外传参导致构造失败
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.no_norm_on_lateral = no_norm_on_lateral
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_cfg = upsample_cfg.copy()

        # compute backbone_end_level
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        # lateral & fpn convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # guided_lsk modules (apply to first two laterals if available)
        self.guided_lsk_cfg = guided_lsk_cfg or {}
        self.guided_lsk_modules = nn.ModuleList()
        num_laterals = len(self.lateral_convs)
        n_add = min(2, num_laterals)
        for i in range(n_add):
            # pass out_channels as both in_channels and guide_channels
            cfg = dict(in_channels=out_channels, guide_channels=out_channels)
            cfg.update(self.guided_lsk_cfg)
            self.guided_lsk_modules.append(GuidedLSK(**cfg))

        # extra convs for P6/P7 if needed
        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and add_extra_convs == 'on_input':
                    in_ch = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_ch = out_channels
                extra_fpn_conv = ConvModule(in_ch, out_channels, 3, stride=2, padding=1,
                                            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # apply guided lsk to first two laterals using the next level as guide
        if len(self.guided_lsk_modules) > 0:
            for idx in range(len(self.guided_lsk_modules)):
                if idx + 1 < len(laterals):
                    laterals[idx] = self.guided_lsk_modules[idx](x=laterals[idx], guide=laterals[idx + 1])

        # top-down fusion
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # extra outputs P6/P7
        if self.num_outs > len(outs):
            if len(self.fpn_convs) > used_backbone_levels:
                for i in range(used_backbone_levels, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
            else:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)
