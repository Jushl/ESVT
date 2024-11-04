from dataset.UAV_EOD.transforms import *


def make_transforms(image_set):
    if image_set == 'train':
        return Compose(
            [RandomPhotometricDistort(p=0.5),
             RandomZoomOut(p=0.5, fill=0),
             RandomIoUCrop(p=0.8),
             SanitizeBoundingBoxes(min_size=1),
             RandomHorizontalFlip(p=0.5),
             Resize([640, 640]),
             SanitizeBoundingBoxes(min_size=1),
             ConvertPILImage(dtype='float32', scale=True),
             ConvertBoxes(fmt='cxcywh', normalize=True)
             ]
        )
    if image_set == 'val' or image_set == 'test':
        return Compose(
            [Resize([640, 640]),
             ConvertPILImage(dtype='float32', scale=True),
             ConvertBoxes(fmt='cxcywh', normalize=True)
             ]
        )
    raise ValueError(f'unknown {image_set}')
