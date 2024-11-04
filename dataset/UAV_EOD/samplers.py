import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Iterator, Sized


class DistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batch = batch_size
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        volume = len(self.dataset) // (self.batch * 100)
        indices = []
        for v in range(volume):
            for i in range(100):
                for j in range(self.batch):
                    indices.append(self.batch * v * 100 + j * 100 + i)
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class StreamingSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized, batch) -> None:
        self.dataset = data_source
        self.batch = batch
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        volume = len(self.dataset) // (self.batch * 100)
        dataset_index = []
        for v in range(volume):
            for i in range(100):
                for j in range(self.batch):
                    dataset_index.append(self.batch * v * 100 + j * 100 + i)

        return iter(dataset_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

