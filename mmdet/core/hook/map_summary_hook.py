# Copyright (c) OpenMMLab. All rights reserved.
"""Hook to print mAP50, mAP75, mAP50:95 after evaluation."""
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MAPSummaryHook(Hook):
    """在验证结束后打印 mAP50、mAP75、mAP50:95 一行汇总。"""

    def __init__(self, prefix='bbox'):
        self.prefix = prefix

    def after_train_epoch(self, runner):
        out = getattr(runner, 'log_buffer', None) and getattr(
            runner.log_buffer, 'output', None
        )
        if not out:
            return
        m = self.prefix
        m_50 = out.get(f'{m}_mAP_50')
        m_75 = out.get(f'{m}_mAP_75')
        m_main = out.get(f'{m}_mAP')
        if m_50 is None and m_75 is None and m_main is None:
            return
        m_50 = m_50 if m_50 is not None else float('nan')
        m_75 = m_75 if m_75 is not None else float('nan')
        m_main = m_main if m_main is not None else float('nan')
        msg = (
            f'mAP50: {m_50:.4f}  |  mAP75: {m_75:.4f}  |  '
            f'mAP50:95: {m_main:.4f}'
        )
        if runner.rank == 0:
            print(msg)
        if hasattr(runner, 'logger') and runner.logger is not None:
            runner.logger.info(msg)
