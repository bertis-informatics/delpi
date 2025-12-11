from typing import List

import polars as pl
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler

SAMPLE_ID_COL = "_sample_index"


class SeqDataBatchSampler(BatchSampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_grouping_column: str = "n_exp_tokens",
        shuffle: bool = True,
        seed: int = None,
        indices: List[int] = None,
        batch_count: int = None,
    ):

        super().__init__(sampler=None, batch_size=batch_size, drop_last=False)

        self.label_df = dataset.label_df.with_row_index(SAMPLE_ID_COL)
        self.batch_grouping_column = batch_grouping_column
        self.random_state = np.random.RandomState(seed)
        self.shuffle = shuffle
        self.batch_count = batch_count
        self.indices = indices

        if indices is None and batch_count is None:
            self.batch_count = self.count_num_of_batches()
        else:
            self.batch_count = batch_count

    def _get_shuffled_index(self):

        batch_size = self.batch_size
        label_df = (
            self.label_df if self.indices is None else self.label_df[self.indices]
        )

        batch_keys = list()
        for _, sub_df in label_df.select(
            pl.col(SAMPLE_ID_COL, self.batch_grouping_column)
        ).group_by(self.batch_grouping_column, maintain_order=True):
            indexes = sub_df[SAMPLE_ID_COL].to_list()
            if self.shuffle:
                indexes = self.random_state.permutation(indexes)
            batch_keys.extend(
                [
                    indexes[i : i + batch_size]
                    for i in range(0, len(indexes), batch_size)
                ]
            )

        # shuffle list of batches, each of which contains the same length samples
        if self.shuffle:
            self.random_state.shuffle(batch_keys)

        return batch_keys

    def __iter__(self):
        for batch in self._get_shuffled_index():
            if self.drop_last and len(batch) != self.batch_size:
                continue
            yield batch

    def count_num_of_batches(self):

        label_df = (
            self.label_df if self.indices is None else self.label_df[self.indices]
        )
        sample_count = label_df.group_by(self.batch_grouping_column).len()
        if self.drop_last:
            return sample_count.select(pl.col("len") // self.batch_size)["len"].sum()
        else:
            return sample_count.select(
                (pl.col("len") + self.batch_size - 1) // self.batch_size
            )["len"].sum()

    def __len__(self):
        # https://pytorch.org/docs/stable/data.html
        # The __len__() method isn’t strictly required by DataLoader,
        # but is expected in any calculation involving the length of a DataLoader.
        # return (self.label_df.shape[0] + self.batch_size - 1) // self.batch_size
        if self.batch_count is not None:
            return self.batch_count

        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def get_batch_sampler_for_seq_data(
    dataset: Dataset,
    batch_grouping_column: str,
    world_size: int,
    shuffle: bool,
    rand_seed: int,
    batch_size: int,
    local_rank: int = 0,
):
    """Create batch sampler that supports multi-GPU training."""

    if world_size > 1:
        batch_sampler = None
        batch_count_list = []
        for rank in range(world_size):
            dist_sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=rand_seed,
                num_replicas=world_size,
                rank=rank,
            )
            batch_sampler_ = SeqDataBatchSampler(
                dataset,
                batch_size=batch_size,
                batch_grouping_column=batch_grouping_column,
                shuffle=shuffle,
                seed=rand_seed,
                indices=list(dist_sampler),
            )
            batch_count_list.append(batch_sampler_.count_num_of_batches())
            if rank == local_rank:
                batch_sampler = batch_sampler_
        # reset batch_count
        batch_sampler.batch_count = min(batch_count_list)
    else:
        batch_sampler = SeqDataBatchSampler(
            dataset,
            batch_size=batch_size,
            batch_grouping_column=batch_grouping_column,
            shuffle=shuffle,
        )

    return batch_sampler
