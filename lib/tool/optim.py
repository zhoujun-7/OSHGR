import math


class LinearWarmupLearningRateScheduler:
    def __init__(self, optimizer, num_epoch, num_warmup, min_ratio=0):
        self.optimizer = optimizer
        self.num_warmup = num_warmup
        self.min_ratio = min_ratio
        self.num_epoch = num_epoch + 1
        self.num_step = 0
        self.init_gourp_lr_ls = [group["lr"] for group in optimizer.param_groups]
        self.step()

    def _get_lr_ratio(self):
        if self.num_step <= self.num_warmup:
            ratio = self.num_step / self.num_warmup
        else:
            ratio = self.min_ratio + (1 - self.min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * (self.num_step - self.num_warmup) / (self.num_epoch - self.num_warmup))
            )
        return ratio

    def step(self):
        self.num_step += 1
        ratio = self._get_lr_ratio()
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = ratio * self.init_gourp_lr_ls[i]
        return ratio


class SinWarmupLearningRateScheduler(LinearWarmupLearningRateScheduler):
    def __init__(self, optimizer, num_epoch, num_warmup, min_ratio=0):
        super().__init__(optimizer, num_epoch, num_warmup, min_ratio)

    def _get_lr_ratio(self):
        if self.num_step <= self.num_warmup:
            ratio = 0.5 + 0.5 * math.sin(math.pi / 2 * (self.num_step - self.num_warmup / 2) / (self.num_warmup / 2))
        else:
            ratio = self.min_ratio + (1 - self.min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * (self.num_step - self.num_warmup) / (self.num_epoch - self.num_warmup))
            )
        return ratio


class PowerWarmupLearningRateScheduler(LinearWarmupLearningRateScheduler):
    def __init__(self, optimizer, num_epoch, num_warmup, min_ratio=0):
        super().__init__(optimizer, num_epoch, num_warmup, min_ratio)

    def _get_lr_ratio(self):
        if self.num_step <= self.num_warmup:
            ratio = (self.num_step / self.num_warmup) ** 2

        else:
            ratio = self.min_ratio + (1 - self.min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * (self.num_step - self.num_warmup) / (self.num_epoch - self.num_warmup))
            )
        return ratio
