import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class HFPParamLoggerHook(Hook):
    """Log learnable HFP scalars during training.

    It supports both single GPU and DDP (runner.model may be wrapped).

    It will try to log (per HFP module index):
    - raw_alpha and mapped alpha via _alpha() (HFP-A)
    - raw_sigma and mapped sigma via _sigma() (HFP-B)

    Logs are written to:
    - runner.logger (optional)
    - runner.log_buffer (optional) so Text/Tensorboard logger hooks can record.
    """

    def __init__(
        self,
        interval: int = 50,
        only_rank0: bool = True,
        log_to_text: bool = True,
        log_to_buffer: bool = True,
        prefix: str = 'hfp',
    ):
        self.interval = int(interval)
        self.only_rank0 = bool(only_rank0)
        self.log_to_text = bool(log_to_text)
        self.log_to_buffer = bool(log_to_buffer)
        self.prefix = str(prefix)

    def _unwrap_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def _is_rank0(self, runner) -> bool:
        # mmcv runner usually has "rank"; if not, treat as rank0 (non-dist).
        return int(getattr(runner, 'rank', 0)) == 0

    @torch.no_grad()
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        if self.only_rank0 and (not self._is_rank0(runner)):
            return

        model = self._unwrap_model(runner)
        neck = getattr(model, 'neck', None)
        if neck is None or not hasattr(neck, 'hfp_modules'):
            return

        logs = {}
        parts = []

        for i, m in enumerate(neck.hfp_modules):
            # HFP-A
            if hasattr(m, 'raw_alpha'):
                raw = float(m.raw_alpha.detach().cpu().item())
                logs[f'{self.prefix}/raw_alpha_{i}'] = raw
                parts.append(f'raw_alpha[{i}]={raw:.6f}')
                if hasattr(m, '_alpha'):
                    a = float(m._alpha().detach().cpu().item())
                    logs[f'{self.prefix}/alpha_{i}'] = a
                    parts.append(f'alpha[{i}]={a:.6f}')

            # HFP-B
            if hasattr(m, 'raw_sigma'):
                raw = float(m.raw_sigma.detach().cpu().item())
                logs[f'{self.prefix}/raw_sigma_{i}'] = raw
                parts.append(f'raw_sigma[{i}]={raw:.6f}')
                if hasattr(m, '_sigma'):
                    s = float(m._sigma().detach().cpu().item())
                    logs[f'{self.prefix}/sigma_{i}'] = s
                    parts.append(f'sigma[{i}]={s:.6f}')

        if not logs:
            return

        if self.log_to_text:
            runner.logger.info(
                f'[HFPParamLoggerHook] iter={runner.iter} ' + ', '.join(parts)
            )

        if self.log_to_buffer:
            num_samples = 1
            outputs = getattr(runner, 'outputs', None)
            if isinstance(outputs, dict):
                num_samples = outputs.get('num_samples', 1) or 1
            runner.log_buffer.update(logs, num_samples)

