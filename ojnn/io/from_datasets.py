from __future__ import annotations
from typing import cast
from collections.abc import Iterator
import jax
from jax import ShapeDtypeStruct
from jax.tree import map as tree_map
from jax.random import bits as jrbits
import datasets  # type: ignore
from ..utils import KeyArray
from .dataset import HostDataset, HostDataStream
from .utils import get_positive_index, get_shape_dtype


def configure_hf_datasets(
    in_memory_max_size: int = 0,
    offline_mode: bool = False,
) -> None:
    datasets.config.IN_MEMORY_MAX_SIZE = in_memory_max_size
    datasets.config.HF_DATASETS_OFFLINE = offline_mode


def _bachify_shape_dtype(sd_tree, batch_size: int):
    return tree_map(
        lambda t: ShapeDtypeStruct(
            shape=(batch_size,) + tuple(t.shape), dtype=t.dtype
        ),
        sd_tree,
    )


class Dataset(HostDataset):

    def __init__(self, host_dataset: datasets.Dataset):
        if not isinstance(host_dataset, datasets.Dataset):
            raise ValueError(
                f"host_dataset should have type datasets.Dataset, received "
                f"{type(host_dataset)}"
            )
        self.host_dataset = host_dataset

    def __getitem__(self, item):
        return self.host_dataset[item]

    def __len__(self) -> int:
        return len(self.host_dataset)

    @property
    def item_shape_dtypes(self):
        return get_shape_dtype(self[0])

    def split(self, at_index: int) -> tuple[Dataset, Dataset]:
        at_index = get_positive_index(at_index, len(self))
        datadict = self.host_dataset.train_test_split(
            train_size=at_index, shuffle=False
        )
        trainset, testset = datadict["train"], datadict["test"]
        return Dataset(trainset), Dataset(testset)

    def shard(self, num_shards: int, shard_index: int) -> Dataset:
        host_dataset = self.host_dataset.shard(num_shards, shard_index)
        return Dataset(host_dataset)

    def shuffle(self, rngkey: KeyArray) -> Dataset:
        ri = jrbits(jax.device_put(rngkey, jax.devices("cpu")[0])).item()
        host_dataset = self.host_dataset.shuffle(ri)
        return Dataset(host_dataset)

    def batchify(self, batch_size: int) -> Dataset:
        host_dataset = self.host_dataset.batch(
            batch_size=batch_size, drop_last_batch=True
        )
        return Dataset(host_dataset)


class IterableDataset(HostDataStream):
    def __init__(
        self,
        host_datastream: datasets.IterableDataset,
        item_shape_dtypes,
        num_items: int,
    ):
        if not isinstance(host_datastream, datasets.IterableDataset):
            raise ValueError(
                f"host_dataset should have type datasets.IterableDataset, "
                f"received {type(host_datastream)}"
            )
        self.host_datastream = host_datastream
        self._item_shape_dtypes = item_shape_dtypes
        self._num_items = num_items

    def __iter__(self) -> Iterator:
        return iter(self.host_datastream)

    def __len__(self) -> int:
        return self._num_items

    @property
    def item_shape_dtypes(self):
        return self._item_shape_dtypes

    def shuffle(self, rngkey: KeyArray, buffer_size: int) -> IterableDataset:
        ri = jrbits(jax.device_put(rngkey, jax.devices("cpu")[0])).item()
        host_stream = self.host_datastream.shuffle(
            seed=ri, buffer_size=buffer_size
        )
        return IterableDataset(host_stream, self._item_shape_dtypes, len(self))

    def shard(self, shard_count: int, shard_index: int) -> IterableDataset:
        # assuming each shard has the same data size
        total_shards = self.host_datastream.num_shards
        shard_rest = total_shards % shard_count
        if shard_index < shard_rest:
            num_shards = total_shards // shard_count + 1
        else:
            num_shards = total_shards // shard_count
        host_stream = self.host_datastream.shard(shard_count, shard_index)
        # come on Huggingface, do the typing properly ...
        host_stream = cast(datasets.IterableDataset, host_stream)
        return IterableDataset(
            host_stream,
            self._item_shape_dtypes,
            len(self) // total_shards * num_shards,
        )

    def batchify(self, batch_size: int) -> IterableDataset:
        host_stream = self.host_datastream.batch(
            batch_size=batch_size, drop_last_batch=True
        )
        return IterableDataset(
            host_stream,
            _bachify_shape_dtype(self._item_shape_dtypes, batch_size),
            len(self) // batch_size,
        )
