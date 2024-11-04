import torch
import torch.nn as nn
import math
from copy import deepcopy
from util.misc import dist_utils

__all__ = ['ModelEMA']


class ModelEMA(object):
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000, ):
        super().__init__()

        self.module = deepcopy(dist_utils.de_parallel(model)).eval()
        self.decay = decay
        self.warmups = warmups
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = dist_utils.de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self, ):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state['module'], strict=strict)
        if 'updates' in state:
            self.updates = state['updates']

    def forwad(self, ):
        raise RuntimeError('ema...')

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu", use_buffers=True):
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)


def build_ema(model):
    return ModelEMA(model)
