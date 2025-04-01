from __future__ import annotations
from typing import Generic, TypeVar, cast
import abc
from collections.abc import Iterator, Iterable, Sequence, Callable
from itertools import islice
import numpy as np
import jax
from jax.lax import fori_loop
from jax.random import split as jrsplit, permutation as jrpermutation
from jax.sharding import SingleDeviceSharding
from ..utils import KeyArray
from .utils import host_callback, get_positive_index, get_shape_dtype


ItemType = TypeVar("ItemType", covariant=True)


class HostDataStream(Iterable, Generic[ItemType], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[ItemType]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def item_shape_dtypes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def shuffle(
        self, rngkey: KeyArray, buffer_size: int
    ) -> HostDataStream[ItemType]:
        raise NotImplementedError

    @abc.abstractmethod
    def shard(
        self, shard_count: int, shard_index: int
    ) -> HostDataStream[ItemType]:
        # shard over batch dim
        raise NotImplementedError

    @abc.abstractmethod
    def batchify(self, batch_size: int) -> HostDataStream[ItemType]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class HostDataset(Sequence, Generic[ItemType], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def item_shape_dtypes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def split(
        self, at_index: int
    ) -> tuple[HostDataset[ItemType], HostDataset[ItemType]]:
        raise NotImplementedError

    @abc.abstractmethod
    def shuffle(self, rngkey: KeyArray) -> HostDataset[ItemType]:
        raise NotImplementedError

    @abc.abstractmethod
    def batchify(self, batch_size: int) -> HostDataset[ItemType]:
        raise NotImplementedError

    def shard(
        self, shard_count: int, shard_index: int
    ) -> HostDataset[ItemType]:
        total = len(self)
        if shard_count > total:
            raise ValueError(f"shard_count too large: {shard_count} > {total}")
        if shard_index < -shard_count or shard_index >= shard_count:
            raise ValueError(f"invalid shard index {shard_index}")
        elif shard_index < 0:
            shard_index = shard_index + shard_count
        # shard over batch dim
        start_idx = len(self) * shard_index // shard_count
        end_idx = len(self) * (shard_index + 1) // shard_count
        return self.split(end_idx)[0].split(start_idx)[1]


class NumpyDataset(HostDataset[dict[str, np.ndarray]]):
    def __init__(self, *, _index: np.ndarray | None = None, **arrays):
        if len(arrays) == 0:
            raise ValueError("no array given")
        if _index is None:
            length = len(next(iter(arrays.values())))
            if not all(len(a) == length for a in arrays.values()):
                errmsg = "\n".join(
                    (
                        "Inconsistent lengths:",
                        *[f"- {k}: {len(v)}" for k, v in arrays.items()],
                    )
                )
                raise ValueError(errmsg)
        else:
            length = None
        self._index = _index
        self._init_length = length
        self._data = arrays

    def flatten_indices(self) -> NumpyDataset:
        data = {k: v[self._index] for k, v in self._data.items()}
        return NumpyDataset(**data)

    def __getitem__(self, item):
        index = self._index
        if index is None:
            return {k: v[item] for k, v in self._data.items()}
        else:
            return {k: v[index[item]] for k, v in self._data.items()}

    def __len__(self) -> int:
        index = self._index
        return self._init_length if index is None else len(index)

    @property
    def item_shape_dtypes(self) -> dict[str, jax.ShapeDtypeStruct]:
        return get_shape_dtype(self[0])

    def split(self, at_index: int) -> tuple[NumpyDataset, NumpyDataset]:
        data, index = self._data, self._index
        index = np.arange(self._init_length) if index is None else index
        at_index = get_positive_index(at_index, len(self))
        return NumpyDataset(_index=index[:at_index], **data), NumpyDataset(
            _index=index[at_index:], **data
        )

    def shuffle(self, rngkey: KeyArray) -> NumpyDataset:
        rngkey = jax.device_put(rngkey, jax.devices("cpu")[0])
        pindices = np.asarray(jrpermutation(rngkey, len(self)))
        data, index = self._data, self._index
        index = pindices if index is None else index[pindices]
        return NumpyDataset(_index=index, **data)

    def batchify(self, batch_size: int) -> NumpyDataset:
        data, index = self._data, self._index
        index = np.arange(self._init_length) if index is None else index
        total = len(index)
        if batch_size > total:
            raise ValueError(f"batch_size too large: {batch_size} > {total}")
        _last = total % batch_size
        if _last:
            index = index[:-_last]

        return NumpyDataset(
            _index=index.reshape(
                [total // batch_size, batch_size, *index.shape[1:]]
            ),
            **data,
        )


StateType = TypeVar("StateType")


def foreach_loop(
    host_data: HostDataStream[ItemType] | HostDataset[ItemType],
    fn: Callable[[ItemType, StateType], StateType],
    init_state: StateType,
    *slice_args,
    shuffle_key: KeyArray | None = None,
    shuffle_stream_buffer_batches: int = 100,
    shard_count: int | None = None,
    shard_index: int | None = None,
    output_sharding: SingleDeviceSharding | None = None,
    unroll: int | bool | None = None,
) -> StateType:
    if not isinstance(host_data, (HostDataStream, HostDataset)):
        raise ValueError(f"invalid host_data with type {type(host_data)}")

    if (shard_count is None) and (shard_index is None):
        _need_shard = False
    elif (shard_count is not None) and (shard_index is not None):
        _need_shard = True
    else:
        raise ValueError("only one of {shard_count, shard_index} specified")

    if slice_args:
        total = len(range(*slice(*slice_args).indices(len(host_data))))
    else:
        total = len(host_data)

    dataiter = None

    def _process_host_data(rngkey):
        nonlocal host_data, dataiter
        if rngkey is None:
            if _need_shard:
                host_data = host_data.shard(
                    cast(int, shard_count), cast(int, shard_index)
                )
        else:
            if _need_shard:
                shard_key, shuffle_key = jrsplit(rngkey)
                rand_shard_indices = jrpermutation(
                    shard_key, cast(int, shard_count)
                )
                host_data = host_data.shard(
                    cast(int, shard_count),
                    rand_shard_indices[shard_index].item(),
                )
            else:
                shuffle_key = rngkey
            if isinstance(host_data, HostDataStream):
                host_data = host_data.shuffle(
                    shuffle_key,
                    buffer_size=shuffle_stream_buffer_batches,
                )
            else:
                host_data = host_data.shuffle(shuffle_key)
        if slice_args:
            dataiter = islice(host_data, *slice_args)
        else:
            dataiter = iter(host_data)

    def _get_next():
        return next(dataiter)

    def _body_fn(_, state):
        data_item = host_callback(
            _get_next,
            debug_impl=False,
            result_shape_dtypes=host_data.item_shape_dtypes,
            sharding=output_sharding,
        )()
        return fn(data_item, state)

    host_callback(_process_host_data, debug_impl=False)(shuffle_key)
    return fori_loop(0, total, _body_fn, init_state, unroll=unroll)
