# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class LambdaSparseSchedulerHook(Hook):
    """Schedule neck.lambda_sparse by epoch.

    Schedule (inclusive):
        - warmup_end_epoch: lambda = 0
        - ramp_end_epoch: linearly ramp to target_lambda
        - after ramp_end_epoch: lambda = target_lambda
    """

    def __init__(self,
                 warmup_end_epoch=3,
                 ramp_end_epoch=8,
                 target_lambda=3e-5):
        if ramp_end_epoch <= warmup_end_epoch:
            raise ValueError('ramp_end_epoch must be greater than warmup_end_epoch')
        self.warmup_end_epoch = int(warmup_end_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.target_lambda = float(target_lambda)

    def _compute_lambda(self, epoch):
        if epoch <= self.warmup_end_epoch:
            return 0.0
        if epoch <= self.ramp_end_epoch:
            ratio = (epoch - self.warmup_end_epoch) / (
                self.ramp_end_epoch - self.warmup_end_epoch)
            return self.target_lambda * ratio
        return self.target_lambda

    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        neck = getattr(model, 'neck', None)
        if neck is None or not hasattr(neck, 'lambda_sparse'):
            return

        cur_lambda = self._compute_lambda(runner.epoch)
        neck.lambda_sparse = cur_lambda

        runner.log_buffer.output['lambda_sparse'] = cur_lambda
        runner.log_buffer.ready = True
