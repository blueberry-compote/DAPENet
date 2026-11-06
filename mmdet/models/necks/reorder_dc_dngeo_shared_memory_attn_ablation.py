"""Shared-memory DRFPN with FE attention ablations.

Extends Reorder_DC_DNGeo_SharedMemory_Lite without modifying its source.

Ablation modes (controlled by use_spatial_attn / use_channel_attn):
- both False: drop the entire FE block (CBR + spatial + channel)
- spatial False only: keep CBR + channel attention
- channel False only: keep CBR + spatial attention
"""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import NECKS
from .reorder_dc_dngeo_shared_memory_light import (
    Reorder_DC_DNGeo_SharedMemory_Lite,
)


class PositionDenoisingModuleAblation(nn.Module):
    """FE block with optional spatial / channel attention (CBR always kept)."""

    def __init__(
        self,
        in_channels: int = 256,
        num_scales: int = 4,
        use_spatial_attn: bool = True,
        use_channel_attn: bool = True,
    ):
        super().__init__()
        if not use_spatial_attn and not use_channel_attn:
            raise ValueError(
                'When both attentions are disabled, the whole FE should be '
                'turned off via use_denoising=False instead of this module')

        self.num_scales = num_scales
        self.use_spatial_attn = use_spatial_attn
        self.use_channel_attn = use_channel_attn

        self.denoising_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True),
            ) for _ in range(num_scales)
        ])

        if use_spatial_attn:
            self.spatial_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                    nn.Sigmoid(),
                ) for _ in range(num_scales)
            ])
        else:
            self.spatial_attentions = None

        if use_channel_attn:
            self.channel_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 16, in_channels, 1),
                    nn.Sigmoid(),
                ) for _ in range(num_scales)
            ])
        else:
            self.channel_attentions = None

        self.denoising_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(num_scales)]
        )

    def forward(self, lateral_features, levels_to_denoise=None):
        if levels_to_denoise is None:
            levels_to_denoise = list(range(len(lateral_features)))
        denoised_features = []
        for i, lateral_feat in enumerate(lateral_features):
            if i not in levels_to_denoise:
                denoised_features.append(lateral_feat)
                continue

            denoised = self.denoising_convs[i](lateral_feat)
            if self.use_spatial_attn:
                denoised = denoised * self.spatial_attentions[i](denoised)
            if self.use_channel_attn:
                denoised = denoised * self.channel_attentions[i](denoised)

            weight = torch.sigmoid(self.denoising_weights[i])
            denoised_feat = weight * denoised + (1 - weight) * lateral_feat
            denoised_features.append(denoised_feat)

        return denoised_features


@NECKS.register_module()
class Reorder_DC_DNGeo_SharedMemory_AttnAblation(
        Reorder_DC_DNGeo_SharedMemory_Lite):
    """Shared-memory neck with configurable FE attention ablations."""

    def __init__(
        self,
        *args,
        use_spatial_attn: bool = True,
        use_channel_attn: bool = True,
        **kwargs,
    ):
        # Drop entire FE (CBR + attentions) when both attentions are off.
        disable_fe = (not use_spatial_attn) and (not use_channel_attn)
        if disable_fe:
            kwargs['use_denoising'] = False

        super().__init__(*args, **kwargs)

        self.use_spatial_attn = use_spatial_attn
        self.use_channel_attn = use_channel_attn

        # Replace the default FE module for partial attention ablations.
        if self.use_denoising:
            self.position_denoising_module = PositionDenoisingModuleAblation(
                in_channels=self.out_channels,
                num_scales=len(self.lateral_convs),
                use_spatial_attn=use_spatial_attn,
                use_channel_attn=use_channel_attn,
            )
