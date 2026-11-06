# Copyright (c) OpenMMLab. All rights reserved.
"""渐进式训练 Hook：阶段二时固定 DC+DNGeo，仅训练 HFP."""
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class FreezeNonHFPHook(Hook):
    """在训练开始时固定 backbone、DC、DNGeo、bbox_head，仅训练 neck 中的 HFP 模块。

    用于渐进式训练的**阶段二**：从 DC+DNGeo checkpoint 加载后，固定其余参数，
    只让 HFP 适应 DC/DNGeo 的输出分布。

    Args:
        hfp_prefix (str): HFP 模块在 model 中的路径前缀，默认为 'neck.hfp_modules'。
    """

    def __init__(self, hfp_prefix='neck.hfp_modules'):
        self.hfp_prefix = hfp_prefix

    def before_train_epoch(self, runner):
        """每个 epoch 开始前确保冻结状态正确（仅第一个 epoch 需要实际操作）."""
        if runner.epoch > 0:
            return
        self._set_freeze_state(runner, freeze_non_hfp=True)

    def _set_freeze_state(self, runner, freeze_non_hfp=True):
        """设置冻结状态：freeze_non_hfp=True 时冻结除 HFP 外的所有参数."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        hfp_param_names = set()
        for name, _ in model.named_parameters():
            if self.hfp_prefix in name:
                hfp_param_names.add(name)

        for name, param in model.named_parameters():
            if freeze_non_hfp:
                param.requires_grad = name in hfp_param_names
            else:
                param.requires_grad = True

        if freeze_non_hfp:
            num_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            num_train = sum(1 for p in model.parameters() if p.requires_grad)
            runner.logger.info(
                f'[FreezeNonHFPHook] Stage 2: freeze DC+DNGeo+backbone+head. '
                f'Training {num_train} param groups (HFP only), frozen {num_frozen}'
            )
