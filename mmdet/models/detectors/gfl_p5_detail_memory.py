"""GFL detector that supplies ground-truth boxes to a memory-aware neck."""
from ..builder import DETECTORS
from .gfl import GFL
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class GFLP5DetailMemory(GFL):
    """GFL with GT-assisted memory writing during training only."""

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        # Keep BaseDetector's batch_input_shape setup used by standard GFL.
        super(SingleStageDetector, self).forward_train(img, img_metas)

        backbone_features = self.backbone(img)
        if self.with_neck:
            features = self.neck(
                backbone_features, gt_bboxes=gt_bboxes)
        else:
            features = backbone_features

        losses = self.bbox_head.forward_train(
            features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore)

        # Preserve the optional behavior of the existing GFL implementation.
        if self.with_neck and hasattr(self.neck, 'get_sparsity_loss'):
            loss_sparse = self.neck.get_sparsity_loss()
            if 'loss_sparse' in losses:
                losses['loss_sparse'] = (
                    losses['loss_sparse'] + loss_sparse)
            else:
                losses['loss_sparse'] = loss_sparse
        return losses
