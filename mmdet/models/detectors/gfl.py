# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class GFL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function for training with optional neck sparsity loss."""
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        if self.with_neck and hasattr(self.neck, 'get_sparsity_loss'):
            loss_sparse = self.neck.get_sparsity_loss()
            if 'loss_sparse' in losses:
                losses['loss_sparse'] = losses['loss_sparse'] + loss_sparse
            else:
                losses['loss_sparse'] = loss_sparse

        return losses
