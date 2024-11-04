import PIL
import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast, Sequence
from dataset.UAV_EOD.function import convert_to_tv_tensor, _boxes_keys, BoundingBoxes, _parse_labels_getter
import torchvision
from torchvision import tv_tensors
from torchvision.ops.boxes import box_iou
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import _get_kernel
from torchvision.transforms.v2._utils import (query_chw,
                                              check_type,
                                              _check_sequence_input,
                                              _get_fill,
                                              _setup_fill_arg,
                                              has_all,
                                              query_size,
                                              get_bounding_boxes,
                                              has_any,
                                              is_pure_tensor
                                              )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, event, target):
        for t in self.transforms:
            image, event, target = t(image, event, target)
        return image, event, target


class Transform(nn.Module):
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor, PIL.Image.Image)

    def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
        needs_transform_list = []
        transform_pure_tensor = not has_any(flat_inputs, tv_tensors.Image, tv_tensors.Video, PIL.Image.Image)
        for inpt in flat_inputs:
            needs_transform = True
            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)


class RandomPhotometricDistort(Transform):
    def __init__(self, brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def _generate_value(self, left: float, right: float) -> float:
        return torch.empty(1).uniform_(left, right).item()

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_channels, *_ = query_chw(flat_inputs)
        params: Dict[str, Any] = {
            key: self._generate_value(range[0], range[1]) if torch.rand(1) < self.p else None
            for key, range in [
                ("brightness_factor", self.brightness),
                ("contrast_factor", self.contrast),
                ("saturation_factor", self.saturation),
                ("hue_factor", self.hue),
            ]
        }
        params["contrast_before"] = bool(torch.rand(()) < 0.5)
        params["channel_permutation"] = torch.randperm(num_channels) if torch.rand(1) < self.p else None
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["brightness_factor"] is not None:
            inpt = self._call_kernel(F.adjust_brightness, inpt, brightness_factor=params["brightness_factor"])
        if params["contrast_factor"] is not None and params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["saturation_factor"] is not None:
            inpt = self._call_kernel(F.adjust_saturation, inpt, saturation_factor=params["saturation_factor"])
        if params["hue_factor"] is not None:
            inpt = self._call_kernel(F.adjust_hue, inpt, hue_factor=params["hue_factor"])
        if params["contrast_factor"] is not None and not params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["channel_permutation"] is not None:
            inpt = self._call_kernel(F.permute_channels, inpt, permutation=params["channel_permutation"])
        return inpt

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )
        flat_outputs = [self._transform(inpt, params) if needs_transform else inpt
                        for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)]
        return tree_unflatten(flat_outputs, spec)


class RandomZoomOut(Transform):
    def __init__(self, fill=0, side_range=(1.0, 2.0), p=0.5,) -> None:
        super().__init__()
        self.p = p
        self.fill = fill
        self._fill = _setup_fill_arg(fill)
        _check_sequence_input(side_range, "side_range", req_sizes=(2,))
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid side range provided {side_range}.")

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)
        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        padding = [left, top, right, bottom]

        return dict(padding=padding)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(F.pad, inpt, **params, fill=fill)

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class RandomIoUCrop(Transform):
    def __init__(self, min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                 sampler_options=None, trials=40, p=0.8):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials
        self.p = p

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if not (
            has_all(flat_inputs, tv_tensors.BoundingBoxes)
            and has_any(flat_inputs, PIL.Image.Image, tv_tensors.Image, is_pure_tensor)
        ):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain tensor or PIL images "
                "and bounding boxes. Sample can also contain masks."
            )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        bboxes = get_bounding_boxes(flat_inputs)
        while True:
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:
                return dict()
            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                xyxy_bboxes = F.convert_bounding_box_format(bboxes.as_subclass(torch.Tensor),
                                                            bboxes.format,
                                                            tv_tensors.BoundingBoxFormat.XYXY,)
                cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(xyxy_bboxes,
                               torch.tensor([[left, top, right, bottom]],
                                            dtype=xyxy_bboxes.dtype,
                                            device=xyxy_bboxes.device),)
                if ious.max() < min_jaccard_overlap:
                    continue
                return dict(top=top, left=left, height=new_h, width=new_w, is_within_crop_area=is_within_crop_area)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt
        output = self._call_kernel(
            F.crop, inpt, top=params["top"], left=params["left"], height=params["height"], width=params["width"]
        )
        if isinstance(output, tv_tensors.BoundingBoxes):
            output[~params["is_within_crop_area"]] = 0
        return output

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p or not inputs[-1]:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class SanitizeBoundingBoxes(Transform):
    def __init__(self, min_size=1.0, labels_getter="default",) -> None:
        super().__init__()

        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}.")
        self.min_size = min_size

        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = inpt is not None and inpt is params["labels"]
        is_bounding_boxes_or_mask = isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask))
        if not (is_label or is_bounding_boxes_or_mask):
            return inpt
        output = inpt[params["valid"]]
        if is_label:
            return output
        return tv_tensors.wrap(output, like=inpt)

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]
        if not inputs[-1]:
            return inputs
        labels = self._labels_getter(inputs)
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"The labels in the input to forward() must be a tensor or None, got {type(labels)} instead."
            )

        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_bounding_boxes(flat_inputs)

        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match."
            )

        boxes = cast(
            tv_tensors.BoundingBoxes,
            F.convert_bounding_box_format(
                boxes,
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            ),
        )
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)
        image_h, image_w = boxes.canvas_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)
        params = dict(valid=valid.as_subclass(torch.Tensor), labels=labels)
        flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]

        return tree_unflatten(flat_outputs, spec)


class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        super().__init__()
        self.p = p

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.horizontal_flip, inpt)

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        if torch.rand(1) >= self.p:
            return inputs
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class Resize(Transform):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=True,) -> None:
        super().__init__()

        if isinstance(size, int):
            size = [size]
        elif isinstance(size, Sequence) and len(size) in {1, 2}:
            size = list(size)
        else:
            raise ValueError(
                f"size can either be an integer or a sequence of one or two integers, but got {size} instead."
            )
        self.size = size

        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, self.size,
                                 interpolation=self.interpolation,
                                 max_size=self.max_size,
                                 antialias=self.antialias,
                                 )

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class ConvertPILImage(Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.
        inpt = tv_tensors.Image(inpt)
        return inpt

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class ConvertBoxes(Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)
