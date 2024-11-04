from dataset.UAV_EOD.build_dataset import concatenate_dataset


def build_dataset(mode, args):
    if args.dataset == 'UAV-EOD':
        return concatenate_dataset(mode, args)


uaveod_category2name = {
    0: 'car',
    1: 'two-wheel',
    2: 'pedestrian',
    3: 'bus',
    4: 'truck',
}

uaveod_name2category = {
    'car': 0,
    'two-wheel': 1,
    'pedestrian': 2,
    'bus': 3,
    'truck': 4,
}

uaveod_category2label = {k: i for i, k in enumerate(uaveod_category2name.keys())}
uaveod_label2category = {v: k for k, v in uaveod_category2label.items()}
