# Copyright (c) OpenMMLab. All rights reserved.
"""自定义加载 checkpoint 的 Hook，支持 strict=False。

用于渐进式训练阶段二：从 Reorder_DC_DNGeo（无 HFP）加载到 HFP_DC_DNGeo，
HFP 模块在 checkpoint 中不存在，需 strict=False 以跳过缺失的 hfp_modules。
"""
from mmcv.parallel import is_module_wrapper
from mmcv.runner import load_checkpoint
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class LoadCheckpointHook(Hook):
    """在训练开始前加载 checkpoint，支持 strict 等参数。

    Args:
        path (str): Checkpoint 路径。
        strict (bool): 是否严格匹配 key，默认 False（允许缺失或多余 key）。
        map_location (str): 加载到设备，默认 'cpu'。
    """

    def __init__(self, path, strict=False, map_location='cpu'):
        self.path = path
        self.strict = strict
        self.map_location = map_location

    def before_train_epoch(self, runner):
        """仅在第一个 epoch 前加载一次."""
        if runner.epoch > 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        load_checkpoint(
            model,
            self.path,
            map_location=self.map_location,
            strict=self.strict,
            logger=runner.logger)
        runner.logger.info(
            f'[LoadCheckpointHook] Loaded from {self.path} (strict={self.strict})'
        )
