import torch
import torchvision
from torch import Tensor


def xy_to_area(boxes):
    x1, y1, x2, y2 = boxes.unbind(1)
    return (x2 - x1) * (y2 - y1)


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


# elementwise
def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = torchvision.ops.box_area(boxes1) # [N, ]
    area2 = torchvision.ops.box_area(boxes2) # [N, ]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N, ]
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2]) # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:]) # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area


def check_point_inside_box(points: Tensor, boxes: Tensor, eps=1e-9) -> Tensor:
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    x1, y1, x2, y2 = [x.unsqueeze(0) for x in boxes.unbind(-1)]

    l = x - x1
    t = y - y1 
    r = x2 - x
    b = y2 - y
    
    ltrb = torch.stack([l, t, r, b], dim=-1)
    mask = ltrb.min(dim=-1).values > eps

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)
    lt = points - x1y1
    rb = x2y2 - points
    return torch.concat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = -lt + points
    x2y2 = rb + points
    boxes = torch.concat([x1y1, x2y2], dim=-1)
    return boxes
