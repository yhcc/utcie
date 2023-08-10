__all__ = [
    'FitlogCallback'
]

import os

from fastNLP.core.callbacks.has_monitor_callback import HasMonitorCallback
from fastNLP.envs import _module_available
from fastNLP.envs import get_global_rank
from fastNLP.core.log import logger

if _module_available('fitlog'):
    import fitlog


class FitlogCallback(HasMonitorCallback):
    """
    自动记录 ``evaluation`` 结果到 ``fitlog`` 中。会自动记录每一次 ``evaluate`` 后的结果；同时会根据
    ``monitor`` 记录最好的结果。另外，会自动将非 ``rank 0`` 上的 ``fitlog`` 设置为 ``debug`` 状态。同时还会在 ``fitlog`` 的
    ``other`` 列中记录一个 ``launch_time`` ，可以通过这个数值找到当前这个脚本的在 save_folder （如果有使用其它需要保存模型的
    ``Callback`` ，例如 :class:`~fastNLP.core.callbacks.CheckpointCallback` ）下的文件夹名称。

    :param monitor: 监控的 metric 值。

        * 为 ``None``
          将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
          尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
          使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
          接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
          的 ``monitor`` 值请返回 ``None`` 。

    :param larger_better: 是否是越大越好。
    :param log_exception: 是否记录 ``exception`` 。
    :param log_loss_every: 多少个 ``batch`` 记录一次 loss 到 ``fitlog`` 中。
    """

    def __init__(self, monitor=None, larger_better: bool = True, log_exception: bool = True, log_loss_every: int = 0):
        assert _module_available('fitlog'), "fitlog is not installed."

        super().__init__(monitor=monitor, larger_better=larger_better)
        self.log_exception = log_exception
        self.log_loss_every = log_loss_every
        self.avg_loss = 0
        self.catch_exception = False
        self.results = []

    def on_after_trainer_initialized(self, trainer, driver):
        if get_global_rank() != 0:  # 如果不是 global rank 为 0 ，需要关闭 fitlog
            fitlog.debug()
        super().on_after_trainer_initialized(trainer, driver)
        fitlog.add_other(name='launch_time', value=os.environ['FASTNLP_LAUNCH_TIME'])

    def on_train_begin(self, trainer):
        fitlog.add_progress(total_steps=trainer.n_batches)

    def on_sanity_check_end(self, trainer, sanity_check_res):
        super(FitlogCallback, self).on_sanity_check_end(trainer, sanity_check_res)
        if self.monitor is None:
            logger.rank_zero_warning(f"No monitor set for {self.log_name}. Therefore, no best metric will "
                                     f"be logged.")

    def on_evaluate_end(self, trainer, results):
        results = self.itemize_results(results)
        fitlog.add_metric(results, step=trainer.global_forward_batches, epoch=trainer.cur_epoch_idx)
        results['step'] = trainer.global_forward_batches
        results['epoch'] = trainer.cur_epoch_idx
        if self.is_better_results(results, keep_if_better=True):
            fitlog.add_best_metric(results)
        better_best = False
        if hasattr(self, 'result_0'):
            for key in list(self.result_0.keys()):
                if results[key] > self.result_0[key]['value']:
                    self.result_0[key]['value'] = results[key]
                    self.result_0[key]['epoch'] = results['epoch']
                    better_best = True
        else:
            better_best = True
            self.result_0 = {}
            for key in list(results.keys()):
                if key in ('f#f#test', 'r_f#f#test', 's_f#f#test', 'e_f#f#test', 'rel_f#f#test'):
                    self.result_0[key] = {'value': results[key], 'epoch': results['epoch']}
        if better_best:
            fitlog.add_other(self.result_0, name='best')

    def on_before_backward(self, trainer, outputs):
        if self.log_loss_every > 0:
            loss = trainer.extract_loss_from_outputs(outputs)
            self.avg_loss += loss.item()
            if trainer.global_forward_batches % self.log_loss_every == 0:
                fitlog.add_loss(self.avg_loss / self.log_loss_every * trainer.accumulation_steps, name='loss',
                                step=trainer.global_forward_batches,
                                epoch=trainer.cur_epoch_idx)
                self.avg_loss = 0

    def on_train_end(self, trainer):
        if not self.catch_exception:
            fitlog.finish()

    def on_exception(self, trainer, exception):
        self.catch_exception = True
        fitlog.finish(status=1)
        if self.log_exception:
            fitlog.add_other(repr(exception), name='except_info')


__all__ = [
    'TorchWarmupCallback'
]

import math
from typing import Union

from fastNLP import Callback


class TorchWarmupCallback(Callback):
    r"""
    调整学习率的 **callback** 。

    :param warmup: 如果 ``warmup`` 为整数，则在该 step 之前，学习率根据 ``schedule`` 的策略变化; 如果 ``warmup`` 为 ``float``，
        如 0.1, 则前 10% 的 step 是按照 ``schedule`` 策略调整。
    :param schedule: 对学习率进行调整的策略：

        1. *linear* -- 前 ``warmup`` 的 step 上升到指定的学习率（从 Trainer 中 optimizer 处获取）, 在剩下的 step 中下降到 0；
        2. *constant* -- 前 ``warmup`` 的 step 上升到指定的学习率，余下的 step 保持不变。
    """

    def __init__(self, warmup: Union[int, float] = 0.1, schedule: str = 'linear'):
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_inverse_square_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def _get_t_steps(self, trainer):
        self.n_batches = trainer.n_batches
        self.num_batches_per_epoch = trainer.num_batches_per_epoch
        if self.warmup > 1:
            self.warmup = self.warmup / trainer.n_batches
        self.t_steps = max(2, trainer.n_batches)  # 不能小于2
        # 防止 t_steps 不能整除 accumulation_steps
        self.t_steps = math.ceil(self.t_steps / trainer.accumulation_steps) * trainer.accumulation_steps

    def on_fetch_data_begin(self, trainer):
        # 因为每个 epoch 可能有变化，这里重新获取一下
        if trainer.num_batches_per_epoch != len(trainer.dataloader):
            trainer.num_batches_per_epoch = len(trainer.dataloader)
            # print(f"Change num_batches_per_epoch:{self.num_batches_per_epoch} to {trainer.num_batches_per_epoch}")
            trainer.num_batches_per_epoch = len(trainer.dataloader)
            trainer.n_batches = trainer.num_batches_per_epoch * trainer.n_epochs
            trainer.global_forward_batches = trainer.num_batches_per_epoch * trainer.cur_epoch_idx

    def on_train_begin(self, trainer):
        self._get_t_steps(trainer)
        # 获取param_group的初始learning rate
        for optimizer in trainer.driver.optimizers:
            for group in optimizer.param_groups:
                self.initial_lrs.append(group['lr'])

    def on_before_optimizers_step(self, trainer, optimizers):
        if self.n_batches != trainer.n_batches:
            self._get_t_steps(trainer)
        # 这里需要加 accumulation_steps 是防止 lr 从 0 开始
        progress = (trainer.global_forward_batches + trainer.accumulation_steps) / self.t_steps
        for optimizer in trainer.driver.optimizers:
            for lr, group in zip(self.initial_lrs, optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)
