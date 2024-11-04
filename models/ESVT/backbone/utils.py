from collections import OrderedDict
from typing import Dict, List
import torch.nn as nn


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 3

    def __init__(self, model: nn.Module, return_layers: List[str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model. {}".format(
                [name for name, _ in model.named_children()]))
        orig_return_layers = return_layers
        return_layers = {str(k): str(k) for k in return_layers}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)

        return outputs