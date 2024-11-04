import random
import sys
from torch.utils.data import ConcatDataset
import bisect
import os
from pathlib import Path
from dataset.UAV_EOD.DVSDetection import DVSDetection
from dataset.UAV_EOD.build_transforms import make_transforms


class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        return self.datasets[dataset_idx][idx], (dataset_idx, idx)
        # if dataset_idx == 0:
        #     sample_idx = idx
        # else:
        #     sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # print('数据集: ', dataset_idx, 'index: ', idx)
        # return self.datasets[dataset_idx][sample_idx], (dataset_idx, idx)


def build_dataset(mode, args, seq_scene, seq_id):
    png_root = Path(os.path.join(args.dataset_path, f'{mode}', 'images'))
    npy_root = Path(os.path.join(args.dataset_path, f'{mode}', 'events'))
    ann_root = Path(os.path.join(args.dataset_path, f'{mode}', 'labels'))
    PATHS_dvs = {"train": (png_root / "{}/{}".format(seq_scene, seq_id),
                           ann_root / "{}/{}".format(seq_scene, seq_id),
                           npy_root / "{}/{}".format(seq_scene, seq_id)),
                 "val": (png_root / "{}/{}".format(seq_scene, seq_id),
                         ann_root / "{}/{}".format(seq_scene, seq_id),
                         npy_root / "{}/{}".format(seq_scene, seq_id)),
                 "test": (png_root / "{}/{}".format(seq_scene, seq_id),
                          ann_root / "{}/{}".format(seq_scene, seq_id),
                          npy_root / "{}/{}".format(seq_scene, seq_id))}
    aps_dir, ann_folder, npy_dir = PATHS_dvs[mode]
    drop_last = True
    dataset = DVSDetection(aps_dir, ann_folder, npy_dir,
                           event_repre=args.event_rep, transforms=make_transforms(mode), drop_last=drop_last,
                           scene=seq_scene, batch_size=args.batch_size)
    return dataset


def concatenate_dataset(mode, args):
    if mode == 'train':
        batch_size = args.batch_size
    elif mode == 'val':
        batch_size = args.batch_size_val
    else:
        batch_size = 1
        assert mode in ['train', 'val'], '模式错误，没有train和val以外的模式'

    set_dict = {'train': 86, 'val': 22}
    entire_dataset = []
    subdataset_num = set_dict[mode]
    print('Loading {} sub-datasets'.format(mode))
    frame_path = os.path.join(args.dataset_path, f'{mode}', 'images')

    scene_list = os.listdir(frame_path)
    pro = 1
    for scene in scene_list:
        scene_folder = os.path.join(frame_path, scene)
        sub_datasets = os.listdir(scene_folder)
        for dataset_name in sub_datasets:
            print('\r', end='')
            print('Loading Progress: {:.2%}'.format(pro / subdataset_num), '▋' * (pro * 50 // subdataset_num), end='')
            sys.stdout.flush()
            entire_dataset.append(build_dataset(mode, args, scene, dataset_name))
            pro += 1

    random.shuffle(entire_dataset)  # 打乱数据集

    if batch_size != 1:
        for _ in range(batch_size - len(entire_dataset) % batch_size):
            entire_dataset.append(random.choice(entire_dataset))
            # del entire_dataset[random.randint(0, len(entire_dataset) - 1)]

    print('\n')
    return ConcatDatasetCustom(entire_dataset)



