"""DDA + FE attention ablation neck without detail memory.

Same FE ablation switches as Reorder_DC_DNGeo_SharedMemory_AttnAblation,
but built on Reorder_DC_DNGeo_Lite so no memory write/read is present.
"""
from ..builder import NECKS
from .reorder_dc_dngeo_light import Reorder_DC_DNGeo_Lite
from .reorder_dc_dngeo_shared_memory_attn_ablation import (
    PositionDenoisingModuleAblation,
)


@NECKS.register_module()
class Reorder_DC_DNGeo_AttnAblation_NoMemory(Reorder_DC_DNGeo_Lite):
    """DDA neck with FE attention ablations and no detail memory."""

    def __init__(
        self,
        *args,
        use_spatial_attn: bool = True,
        use_channel_attn: bool = True,
        **kwargs,
    ):
        disable_fe = (not use_spatial_attn) and (not use_channel_attn)
        if disable_fe:
            kwargs['use_denoising'] = False

        super().__init__(*args, **kwargs)

        self.use_spatial_attn = use_spatial_attn
        self.use_channel_attn = use_channel_attn

        if self.use_denoising:
            self.position_denoising_module = PositionDenoisingModuleAblation(
                in_channels=self.out_channels,
                num_scales=len(self.lateral_convs),
                use_spatial_attn=use_spatial_attn,
                use_channel_attn=use_channel_attn,
            )
