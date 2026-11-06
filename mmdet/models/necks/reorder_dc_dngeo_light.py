import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# 1. 位置去噪模块 (Position Denoising Module)
# ---------------------------------------------------------------------------
class PositionDenoisingModule(nn.Module):
    

    def __init__(self, in_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.denoising_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="ReLU", inplace=True),
            ) for _ in range(num_scales)
        ])

        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])

        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])

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
            
            # 特征增强
            denoised = self.denoising_convs[i](lateral_feat)
            denoised = denoised * self.spatial_attentions[i](denoised)
            denoised = denoised * self.channel_attentions[i](denoised)
            
            # 权重融合
            weight = torch.sigmoid(self.denoising_weights[i])
            denoised_feat = weight * denoised + (1 - weight) * lateral_feat
            denoised_features.append(denoised_feat)

        return denoised_features

# ---------------------------------------------------------------------------
# 2. 动态稀疏对齐模块 (_DCNAlign)
# ---------------------------------------------------------------------------
class _DCNAlign(nn.Module):
    """
    核心机制：
    1. 动态稀疏采样：根据 Mask 的响应强度自动关闭不重要的采样点。
    2. 支持软门控/硬门控两种掩码策略。
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        deform_groups: int = 1,
        init_zero_offset: bool = True,
        mask_tau: float = 0.05,
        gate_mode: str = "soft",
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups
        self.mask_tau = mask_tau
        if gate_mode not in ("soft", "hard"):
            raise ValueError(f"gate_mode must be 'soft' or 'hard', got {gate_mode}")
        self.gate_mode = gate_mode

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
        out = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        
        offset = out[:, :self.offset_dim, :, :]
        mask = torch.sigmoid(out[:, self.offset_dim:, :, :])
        
        # 记录用于计算 L1 Sparsity Loss
        self.last_mask = mask

        # 动态稀疏化逻辑（与 mask 同 dtype，避免 fp16 训练时 .float() 变 float32 导致 DCN 报错）
        keep_binary = (mask > self.mask_tau).to(mask.dtype)
        if self.gate_mode == "soft":
            # 软门控：通过阈值筛掉低响应点，同时保留通过点的幅值信息。
            mask_sparse = keep_binary * mask
        else:
            # 硬门控：前向 0/1，反向使用 STE 保持梯度可传。
            mask_sparse = keep_binary - mask.detach() + mask
        mask_sparse = mask_sparse.to(dtype=p_ip1_up.dtype, device=p_ip1_up.device)
        self.last_keep_ratio = keep_binary.float().mean()

        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask_sparse)
        return aligned

# ---------------------------------------------------------------------------
# 3. 主 Neck 模块 (Reorder_DC_DNGeo)
# ---------------------------------------------------------------------------
@NECKS.register_module()
class Reorder_DC_DNGeo_Lite(FPN):
    """
    流程：laterals -> top-down(DCN Align) -> 部分层去噪 -> fpn_convs -> outs
    """
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
        use_denoising=True,
        denoising_weight=1.0,
        denoising_after_topdown: bool = True,
        denoising_levels: Optional[List[int]] = None,
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        dcn_levels: Optional[Union[List[int], str]] = None,
        dcn_mask_tau: float = 0.03,
        dcn_gate_mode: str = "soft",
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

        self.use_denoising = use_denoising
        self.denoising_weight = denoising_weight
        self.denoising_after_topdown = denoising_after_topdown

        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusion_steps = used_backbone_levels - 1

        if dcn_levels is None or dcn_levels == "all":
            self.dcn_levels = list(range(num_fusion_steps))
        else:
            self.dcn_levels = list(dcn_levels)

        if denoising_levels is None:
            self.denoising_levels = [0, 1]
        else:
            self.denoising_levels = list(denoising_levels)

        if self.use_denoising:
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels, num_scales=len(self.lateral_convs)
            )

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
        """返回当前 batch 的平均激活比例（用于日志观测，不参与反传）。"""
        keep_ratio = None
        count = 0
        for m in self.align_modules:
            if hasattr(m, "last_keep_ratio"):
                cur = m.last_keep_ratio
                keep_ratio = cur if keep_ratio is None else (keep_ratio + cur)
                count += 1
        if count > 0:
            return keep_ratio / count
        return self.align_modules[0].offset_and_mask.weight.new_tensor(0.0)

    def _build_laterals_with_denoising(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.use_denoising and not self.denoising_after_topdown:
            denoised_laterals = self.position_denoising_module(laterals)
            laterals = [
                orig + self.denoising_weight * (denoised - orig)
                for orig, denoised in zip(laterals, denoised_laterals)
            ]
        return laterals

    def _build_topdown_and_outputs(self, laterals, inputs):
        pre_topdown_laterals = [l.clone() for l in laterals]
        used_backbone_levels = len(laterals)

        # 1. Top-down 融合与对齐
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        # 2. 融合后的位置去噪
        if self.denoising_after_topdown and self.use_denoising:
            denoised_laterals = self.position_denoising_module(
                laterals, levels_to_denoise=self.denoising_levels
            )
            for i in self.denoising_levels:
                if i < len(laterals):
                    orig = laterals[i]
                    laterals[i] = orig + self.denoising_weight * (
                        denoised_laterals[i] - orig
                    )

        # 3. 生成输出层 P3-P5
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 4. 生成额外的高层输出 P6, P7...
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

        return tuple(outs), pre_topdown_laterals

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_and_outputs(laterals, inputs)
        return fpn_outs