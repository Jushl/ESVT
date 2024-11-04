import importlib.metadata
from torch import Tensor
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
from torchvision.transforms.v2._utils import is_pure_tensor
from contextlib import suppress
import collections.abc


if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()
    _boxes_keys = ['format', 'spatial_size']
elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()
    _boxes_keys = ['format', 'canvas_size']
elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.tv_tensors import (BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')


def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    assert key in ('boxes', 'masks',), "Only support 'boxes' and 'masks'"

    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
        return Mask(tensor)


def _find_labels_default_heuristic(inputs: Any) -> torch.Tensor:
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[-1]
    if is_pure_tensor(inputs):
        return inputs
    if not isinstance(inputs, collections.abc.Mapping):
        raise ValueError(
            f"When using the default labels_getter, the input passed to forward must be a dictionary or a two-tuple "
            f"whose second item is a dictionary or a tensor, but got {inputs} instead."
        )
    candidate_key = None
    with suppress(StopIteration):
        candidate_key = next(key for key in inputs.keys() if key.lower() == "labels")
    if candidate_key is None:
        with suppress(StopIteration):
            candidate_key = next(key for key in inputs.keys() if "label" in key.lower())
    if candidate_key is None:
        raise ValueError(
            "Could not infer where the labels are in the sample. Try passing a callable as the labels_getter parameter?"
            "If there are no labels in the sample by design, pass labels_getter=None."
        )
    return inputs[candidate_key]


def _parse_labels_getter(labels_getter: Union[str, Callable[[Any], Optional[torch.Tensor]], None]) -> Callable[[Any], Optional[torch.Tensor]]:
    if labels_getter == "default":
        return _find_labels_default_heuristic
    elif callable(labels_getter):
        return labels_getter
    elif labels_getter is None:
        return lambda _: None
    else:
        raise ValueError(f"labels_getter should either be 'default', a callable, or None, but got {labels_getter}.")

