# mmdet/models/detectors/gfl_geo.py

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module()
class GFLGeo(SingleStageDetector):
    """
    本文件定义了一个名为 GFLGeo 的检测器，继承自 SingleStageDetector。
    相比标准 GFL 检测器，这里支持从 neck（如 DNSem_FPN）中返回附加的 geo_loss（几何/对比损失），
    并将其合并到最终的总 loss 里参与训练。

    主要实现内容如下：

    1. extract_feat:
        从 backbone 提取特征后，如果有 neck，则将特征进一步送入 neck。
        这里 neck 接口允许返回 (feats, neck_losses)（例如带损失的语义/几何对齐 FPN），
        只取第一个作为后续 head 的输入特征。
        这样下游 bbox_head 的推理部分逻辑与原始保持一致。

    2. forward_train:
        训练阶段，首先保留父类对 batch_input_shape 的写入逻辑（用于多尺度训练）。
        从 backbone 得到主干特征 x。
        如果定义了 neck，则 neck 输入 x，同样允许对应 tuple 返回（多尺度特征与 neck 内部 loss）。
        继续将特征送入 head 做常规目标损失。
        最后如果 neck 有额外损失（如对比/geo loss），则合并到 det_losses 里供总体 loss 反向传播。

    3. simple_test:
        推理阶段只需用 extract_feat 提特征，再用 bbox_head.simple_test 得到检测结果后格式化返回
        （bbox2result 按类别组装结果，支持 COCO/VisDrone 等评测结构）。

    总结：本 detector 用于 neck 有额外损失需要合并进总 loss（例如语义或几何对比损失）的场景，
    推理时则与普通检测器无异。
    """

    def extract_feat(self, img):
        # 提取 backbone 特征，然后经过 neck（如 DNSem_FPN）。
        # 若 neck 输出为 (feats, neck_losses)，仅取特征部分用于后续。
        # 注意：常规 FPN neck 的输出本身就是一个 tuple/list[Tensor]（多层特征），
        # 不能仅凭 isinstance(tuple) 就当作 (feats, losses) 来解包/取第 0 项。
        x = self.backbone(img)
        if self.with_neck:
            neck_out = self.neck(x, backbone_features=x)
            if (
                isinstance(neck_out, tuple)
                and len(neck_out) == 2
                and isinstance(neck_out[1], dict)
            ):
                # 只取 (feats, neck_losses) 的 feats
                x = neck_out[0]
            else:
                # 常规 neck：直接返回多尺度特征
                x = neck_out
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # 保留父类写入 batch_input_shape 的功能
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # 正常通过 backbone 和 neck 得到特征
        x = self.backbone(img)
        feats = x
        neck_losses = {}

        if self.with_neck:
            neck_out = self.neck(x, backbone_features=x)
            if (
                isinstance(neck_out, tuple)
                and len(neck_out) == 2
                and isinstance(neck_out[1], dict)
            ):
                feats, neck_losses = neck_out   # 支持 neck 同时返回对比损失
            else:
                feats = neck_out

        # 继续普通 bbox_head 训练流程
        det_losses = self.bbox_head.forward_train(
            feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        # 合并 neck 的损失（如果有）
        if neck_losses:
            det_losses.update(neck_losses)

        return det_losses

    def simple_test(self, img, img_metas, rescale=False):
        # 推理阶段：仅用提特征和 head 推理，不管 neck 的附加损失
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results