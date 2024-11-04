from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image
from dataset.event_repre.VoxelGrid import VoxelGrid
import json
import dataset
from dataset.UAV_EOD.function import convert_to_tv_tensor


class DVSDetection(Dataset):
    def __init__(self, aps_dir, ann_folder, npy_dir, event_repre, transforms, drop_last, scene, batch_size):
        self._transforms = transforms
        self.aps_dir = aps_dir
        self.anno_dir = ann_folder
        self.npy_dir = npy_dir
        self.drop_last = drop_last
        self.scene = scene
        self.frame_len = len(os.listdir(self.aps_dir))
        self.event_repre = event_repre
        self.aps_files = [f for f in os.listdir(self.aps_dir) if f.endswith('.png')]
        self.npy_files = [f for f in os.listdir(self.npy_dir) if f.endswith('.npy')]
        self.anno_files = [f for f in os.listdir(self.anno_dir) if f.endswith('.json')]

    def __getitem__(self, idx):
        index = idx % 100
        image_path = os.path.join(self.aps_dir, self.aps_files[index])
        image = Image.open(image_path).convert('RGB')
        event = np.load(os.path.join(self.npy_dir, self.npy_files[index]))
        W, H = image.size
        if self.event_repre.lower() == 'voxel':
            event = VoxelGrid(event, 3, H, W)
        else:
            assert self.event_repre.low() in ['voxel']
        event = Image.fromarray(event.astype(np.uint8))
        anno_path = image_path.replace('images', 'labels').replace('png', 'json')
        if os.path.exists(anno_path):
            anno = self.get_json_boxes(anno_path)
            target = {'image_id': idx, 'boxes': anno['boxes'], 'labels': anno['labels']}
            target = self.prepare(image, event, target)
        else:
            target = {}
        image, event, target = self._transforms(image, event, target)
        return image, event, target

    def __len__(self):
        return self.frame_len

    def prepare(self, image, event, target):
        if image:
            w, h = image.size
        else:
            w, h = event.size
        gt = {}
        gt["orig_size"] = torch.as_tensor([int(w), int(h)])
        # gt["size"] = torch.as_tensor([int(h), int(w)])
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        boxes = target['boxes']
        classes = target['labels']
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        gt["boxes"] = boxes
        gt["labels"] = classes
        gt["image_id"] = image_id
        if 'boxes' in gt:
            gt['boxes'] = convert_to_tv_tensor(gt['boxes'], key='boxes', spatial_size=image.size[::-1])
        return gt

    def get_json_boxes(self, label_filename):
        with open(label_filename, 'r') as json_file:
            data = json.load(json_file)
            objects = data['shapes']
            class_indexes = []
            bounding_boxes = []
            for i in range(len(objects)):
                bounding_boxes_points = objects[i]['points']
                if 'label' in objects[i]:
                    bounding_boxes_class = objects[i]['label']
                else:
                    bounding_boxes_class = objects[i]['lable']

                class_index = int(dataset.uaveod_name2category[bounding_boxes_class])
                bounding_box = [int(bounding_boxes_points[0][0]), int(bounding_boxes_points[0][1]),
                                int(bounding_boxes_points[2][0]), int(bounding_boxes_points[2][1])]

                class_indexes.append(class_index)
                bounding_boxes.append(bounding_box)

        return {'labels': class_indexes, 'boxes': bounding_boxes}


