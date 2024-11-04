import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
import random


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(self,  scales=None,  stop_epoch=None, ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def __call__(self, items):
        images = torch.cat([x[0][0][None] for x in items], dim=0)
        events = torch.cat([x[0][1][None] for x in items], dim=0)
        targets = [x[0][2] for x in items]
        index = [x[1] for x in items]
        index = [[t[0] for t in index], [t[1] for t in index]]
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            events = F.interpolate(events, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return (images, events, targets), index




