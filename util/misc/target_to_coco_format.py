import sys
import copy
from util.misc.box_ops import box_cxcywh_to_xyxy, xy_to_area, convert_to_xywh
import torch
from pycocotools.coco import COCO
from dataset import uaveod_category2name


def get_categories_ids(uaveod_category2name):
    return [{"id": id, "name": class_name, "supercategory": "none"} for id, class_name in uaveod_category2name.items()]


def target_to_coco_format(dataLoader):
    print('Loading annotations into memory')
    anno_id = 1
    annotations = []
    images = []
    height = 260
    width = 346

    categories = get_categories_ids(uaveod_category2name)
    pro = 1
    subdataset_num = len(dataLoader)
    for inputs, indexes in dataLoader:
        print('\r', end='')
        print('Loading Progress: {:.2%}'.format(pro / subdataset_num), '▋' * (pro * 50 // subdataset_num), end='')
        sys.stdout.flush()
        pro += 1
        _, _, targets = inputs
        _, img_ids = indexes
        targets = copy.deepcopy(targets)
        for target in targets:
            if target:
                img_w, img_h = target['orig_size'][None].unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                boxes = box_cxcywh_to_xyxy(target['boxes'])
                target['boxes'] = boxes * scale_fct

        for (gt, img_id) in zip(targets, img_ids):
            if gt:
                images.append(
                    {
                        "date_captured": "2024",
                        "file_name": "UAV-EOD",
                        "id": img_id,
                        "license": 1,
                        "url": "",
                        "height": height,
                        "width": width
                    }
                )

                gt_boxes = gt['boxes']
                gt_labels = gt['labels'].tolist()
                gt_areas = xy_to_area(gt_boxes).tolist()
                gt_boxes = convert_to_xywh(gt_boxes).tolist()
                for k, box in enumerate(gt_boxes):
                    annotations.append(
                        {
                            "area": float(gt_areas[k]),
                            "iscrowd": False,
                            "image_id": img_id,
                            "bbox": box,
                            "category_id": gt_labels[k],
                            "id": anno_id
                        }
                    )
                    anno_id += 1

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    print('\n')
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt
