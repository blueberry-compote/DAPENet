import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

@HEADS.register_module()
class LSK(nn.Module):
    """
    LSK-style 双分支（适配 FPN P3/P4）

    参数:
      in_channels: 输入通道
      reduction: 降维比例
      guide_channels: 若提供且 != in_channels 则做 1x1 投影
    
      act_layer: 激活

    forward(x, guide): x: [B,C,H,W], guide: [B,Cg,Hg,Wg] or None
    """
    def __init__(self,
                 in_channels,
                 reduction=2,
                 act_layer=nn.ReLU,
                 level='p4'):
        super().__init__()

        self.in_channels = in_channels
        self.mid = max(1, in_channels // reduction)
        self.level = level.lower()

        

        # branch1: 5x5 depthwise
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2,
                             groups=in_channels, bias=False)
        # 5×5空洞卷积                     
        self.dw_large_k5_d2 = nn.Conv2d(in_channels, in_channels, kernel_size=5,
                                        padding=4, dilation=2, groups=in_channels, bias=False) 
        self.dw_large_k7_d3 = nn.Conv2d(in_channels,in_channels,kernel_size=7,
                                        padding=9,dilation=3,groups=in_channels,bias=False)                     

        # reductions
        self.reduce1 = nn.Conv2d(in_channels, self.mid, 1, bias=False)
        self.reduce2 = nn.Conv2d(in_channels, self.mid, 1, bias=False)

        

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
        fuse = b1 + b2

        out = self.fuse(fuse)
        out = x + self.act(out)
        return out
