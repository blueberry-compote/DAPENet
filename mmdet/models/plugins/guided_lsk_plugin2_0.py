import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

@HEADS.register_module()
class GuidedLSK2_0(nn.Module):
    """
    LSK-style 双分支 + 高层语义引导（适配 FPN P3/P4）

    参数:
      in_channels: 输入通道
      reduction: 降维比例
      guide_channels: 若提供且 != in_channels 则做 1x1 投影
      use_channel_guide: 是否使用通道引导（默认 True）
      use_spatial_guide: 是否使用空间引导（默认 True）
      act_layer: 激活

    forward(x, guide): x: [B,C,H,W], guide: [B,Cg,Hg,Wg] or None
    """
    def __init__(self,
                 in_channels,
                 reduction=2,
                 guide_channels=None,
                 use_channel_guide=True,
                 use_spatial_guide=True,
                 act_layer=nn.ReLU,
                 level='p4'):
        super().__init__()

        self.in_channels = in_channels
        self.mid = max(1, in_channels // reduction)
        self.level = level.lower()

        # guide 投影（如果需要）
        self.guide_proj = None
        if guide_channels is not None and guide_channels != in_channels:
            self.guide_proj = nn.Conv2d(guide_channels, in_channels, 1) # 将通道变为一致

        # branch1: 5x5 depthwise
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2,
                             groups=in_channels, bias=False)
       
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)

        # 5×5空洞卷积                     
        self.dw_large_k5_d2 = nn.Conv2d(in_channels, in_channels, kernel_size=5,
                                        padding=4, dilation=2, groups=in_channels, bias=False) 
        self.dw_large_k7_d3 = nn.Conv2d(in_channels,in_channels,kernel_size=7,
                                        padding=9,dilation=3,groups=in_channels,bias=False)                     

        # reductions
        self.reduce1 = nn.Conv2d(in_channels, self.mid, 1, bias=False)
        self.reduce2 = nn.Conv2d(in_channels, self.mid, 1, bias=False)

        # flags
        self.use_channel_guide = use_channel_guide
        self.use_spatial_guide = use_spatial_guide

        # channel gate (optional)
        if self.use_channel_guide:
            self.channel_gate = nn.Sequential(
                nn.Conv2d(in_channels*4, self.mid, 1, bias=False),
                act_layer(inplace=True),
                nn.Conv2d(self.mid, self.mid, 1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.channel_gate = None

        
        if self.use_spatial_guide:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(in_channels*2, self.mid, kernel_size=1, bias=False),   # 压缩通道
                act_layer(inplace=True),                                       # 激活
                nn.Conv2d(self.mid, 1, kernel_size=3, padding=1, bias=False), # 恢复空间信息
                nn.Sigmoid()                                                  # 空间权重（注意力）
            )
        else:
            self.spatial_gate = None

        # fuse back
        self.fuse = nn.Conv2d(self.mid, in_channels, 1, bias=True)
        self.act = act_layer(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
    

    def forward(self, x, guide=None):
        B, C, H, W = x.shape

        # guide 处理
        if guide is not None:
            if self.guide_proj is not None:
                guide = self.guide_proj(guide)
            guide = F.interpolate(guide, size=(H, W), mode='bilinear', align_corners=False)

            c1=self.dw3(x)
            c2=self.dw3(guide)

            # 这里b3等于b1与b2在通道维度拼接
            b3 = torch.cat([c1, c2], dim=1)
            
            if self.use_channel_guide:
                avg_pool = F.adaptive_avg_pool2d(b3, (1, 1))
                max_pool = F.adaptive_max_pool2d(b3, (1, 1))
                chan_gate = torch.cat([avg_pool, max_pool], dim=1)
                chan_gate = self.channel_gate(chan_gate)  # [B, mid, 1, 1]
            else:
                # 当不使用channel引导时，chan_gate为全1张量，实现等价的无门控效果
                chan_gate = x.new_ones(B, self.mid, 1, 1) 

            if self.use_spatial_guide:
                spa_gate = self.spatial_gate(b3)   # [B, 1, H, W]
            else:
                spa_gate =  x.new_ones(B, 1, H, W)
        else:
            return x

        # branch1 (local)
        dw5_out =self.dw5(x)
        b1 = self.reduce1(dw5_out)  # [B, mid, H, W]

        # branch2 (large, built on top of dw5)
        # 正确的方式通常是让更大核更大膨胀感受野用于大尺寸feat，不然小特征用大卷积会过度池化
        if self.level =='p3':
            large_conv = self.dw_large_k7_d3 
        else:
            large_conv = self.dw_large_k5_d2

        b2 = large_conv(dw5_out)
        b2 = self.reduce2(b2)

        # fuse with guide gates
        fuse = chan_gate * b1 + (1.0 - chan_gate) * b2
        fuse = fuse * spa_gate  # spatial gating broadcast

        out = self.fuse(fuse)
        out = x + self.act(out) 
        
        return out
