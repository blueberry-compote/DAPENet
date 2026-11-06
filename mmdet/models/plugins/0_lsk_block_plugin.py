import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS

@HEADS.register_module()
class LSKPluginModule(nn.Module):
    """Plug-and-Play Large Selective Kernel Attention Module
    从 LSKNet 抽取而来，可插入任意 backbone/neck/head。
    
    输入输出: [B, C, H, W] → [B, C, H, W]
    """

    def __init__(self, in_channels, reduction=2, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        mid_channels = in_channels // reduction

        # 原 LSKblock 的核心
        self.conv0 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        # 空洞卷积 减小计算量
        self.conv_spatial = nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=9, groups=in_channels, dilation=3)
        # 两个1*1卷积:特征降维+并行拆分
        # mid_channles是卷积核的个数,意在降低通道数（mid<in)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, 1)
        # 全局信息提取 + 注意力权重生成
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3) # in_channle和out_channle都是2
        # 特征融合与通道还原
        self.conv_fuse = nn.Conv2d(mid_channels, in_channels, 1)# 还原维度+ 两路特征与权重相乘
        # 激活
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 两个分支：局部（5x5）与大感受野（dilated 7x7）
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)# 1*1卷积——通道降维（减少计算量，为融合做准备）
        attn2 = self.conv2(attn2)

        # 通道融合
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)# 平均池化能反映特征的全局整体强度
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)# 捕捉特征中的局部突出响应（如边缘、纹理的强信号）
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # 动态空间权重
        sig = self.conv_squeeze(agg).sigmoid()

        # 两个分支融合
        attn = attn1 * sig[:, 0:1] + attn2 * sig[:, 1:2]
        attn = self.conv_fuse(attn)
        out = x * attn
        return out
