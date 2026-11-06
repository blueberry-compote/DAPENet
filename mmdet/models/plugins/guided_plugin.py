import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

@HEADS.register_module()
class Guided(nn.Module):
    """
    🔹轻量化 + 语义引导版 LSK 模块（Guided LSK for FPN）

    主要思路：
    - 在 FPN 的低层特征（如 P3、P4）上使用；
    - 用上层语义特征（如 P5 或 P4）作为引导，指导低层如何自适应选择感受野；
    - 模块输出与输入尺寸相同，可直接插入 FPN；
    - 保留 LSK 的“大核+选择”思想，但移除膨胀卷积以减小计算量；
    - 引入 “跨层引导机制”：包括通道引导（Channel-wise）和空间引导（Spatial-wise）。

    输入：
        x: 当前层特征 [B, C, H, W]（低层特征）
        guide: 上层特征 [B, Cg, Hg, Wg]（高层语义引导特征）
    输出：
        与 x 尺寸相同的特征 [B, C, H, W]
    """

    def __init__(self,
                 in_channels,          # 当前层特征通道数
                 reduction=2,          # 通道压缩比例
                 guide_channels=None,  # 引导特征通道数（可自动推断）
                 use_channel_guide=True,  # 是否启用通道引导
                 use_spatial_guide=True,  # 是否启用空间引导
                 act_layer=nn.ReLU):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.in_channels = in_channels
        self.mid = mid

        # ------------------------- #
        # （1）局部上下文提取分支
        # ------------------------- #
        # 用轻量化 depthwise 5x5 卷积替代原始 LSK 的多分支大核卷积（节省计算）
        self.local_dw = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=True)
        # 降维（压缩通道数）
        self.conv_reduce = nn.Conv2d(in_channels, mid, kernel_size=1)

        # ------------------------- #
        # （2）引导分支：用于高层语义引导
        # ------------------------- #
        self.use_channel_guide = use_channel_guide
        self.use_spatial_guide = use_spatial_guide

        # 如果引导特征的通道数与当前层不同，则通过1x1卷积进行通道对齐
        self.guide_proj = None
        if guide_channels is not None and guide_channels != in_channels:
            self.guide_proj = nn.Conv2d(guide_channels, in_channels, kernel_size=1)

        # （2.1）通道引导：通过全局平均池化提取全局语义 → 两层FC → sigmoid
        if self.use_channel_guide:
            self.chan_fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),        # 全局平均池化 [B, C, 1, 1]
                nn.Conv2d(in_channels, mid, 1), # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, in_channels, 1), # 升维
                nn.Sigmoid()
            )

        # （2.2）空间引导：上采样高层特征 → 生成空间引导权重图
        if self.use_spatial_guide:
            self.spatial_proj = nn.Sequential(
                nn.Conv2d(in_channels, mid, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 1, 3, padding=1),  # 输出单通道空间图
                nn.Sigmoid()
            )

        # ------------------------- #
        # （3）自注意力分支（空间注意）
        # ------------------------- #
        # 对经过引导后的特征做 avg/max pooling，然后用3x3卷积生成空间注意图
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        # 将中间特征映射回原通道
        self.conv_fuse = nn.Conv2d(mid, in_channels, 1)

        # 激活函数
        self.act = act_layer(inplace=True)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        # 统一初始化卷积层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, guide=None):
        """
        前向传播：
        x: 当前层特征 [B, C, H, W]
        guide: 上层语义特征 [B, Cg, Hg, Wg]
        """
        B, C, H, W = x.shape

        # （1）先提取当前层的局部上下文
        local = self.local_dw(x)
        reduced = self.conv_reduce(local)  # [B, mid, H, W]

        # （2）生成引导权重
        channel_gate = None
        spatial_gate = None

        if guide is not None:
            g = guide
            if self.guide_proj is not None:
                g = self.guide_proj(g)  # 通道对齐

            # （2.1）通道引导：全局语义控制每个通道的重要性
            if self.use_channel_guide:
                channel_gate = self.chan_fc(g)  # [B, C, 1, 1]

            # （2.2）空间引导：上层特征上采样后生成空间权重图
            if self.use_spatial_guide:
                g_up = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
                spatial_gate = self.spatial_proj(g_up)  # [B, 1, H, W]

        # （3）应用引导权重
        guided = reduced
        if channel_gate is not None:
            # 将通道权重调整为 mid 通道
            if C % self.mid == 0:
                group = C // self.mid
                gate = channel_gate.view(B, self.mid, group).mean(-1).view(B, self.mid, 1, 1)
            else:
                gate = channel_gate[:, :self.mid, :, :]
            guided = guided * gate  # 通道引导

        if spatial_gate is not None:
            guided = guided * spatial_gate  # 空间引导

        # （4）生成自注意力图
        avg = torch.mean(guided, dim=1, keepdim=True)
        maxv, _ = torch.max(guided, dim=1, keepdim=True)
        attn = torch.cat([avg, maxv], dim=1)
        att_map = torch.sigmoid(self.spatial_attn(attn))  # [B, 1, H, W]

        # 加上这两行保存可视化用的特征
        self._last_att_map = att_map.detach()
        self._last_guided = guided.detach()

        # （5）融合并残差连接
        out = self.conv_fuse(guided * att_map)
        return x + self.act(out)
