import itertools

import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset


class DownsampleSampler(Sampler[int]):
    def __init__(self, data_source: Dataset, n: int, seed: int = 0):
        self.data_source = data_source
        self.n = n
        # self.seed = seed
        # self._epoch_counter = itertools.count()  # 자동 증가 카운터
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        # epoch = next(self._epoch_counter)
        # g = torch.Generator()
        # g.manual_seed(self.seed + epoch)
        N = len(self.data_source)
        k = min(self.n, N)
        idx = torch.randperm(N, generator=self.generator)[:k].tolist()
        return iter(idx)

    def __len__(self):
        return min(self.n, len(self.data_source))
