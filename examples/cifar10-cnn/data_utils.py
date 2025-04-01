from collections.abc import Sequence
import abc
from pathlib import Path
import numpy as np
from jax import Array
from datasets import load_dataset, Features, Array3D
from jax.numpy import (
    asarray as asjaxarray,
    expand_dims,
    flip,
    pad,
)
from jax.random import randint, uniform
from jax.lax import dynamic_slice, cond
from ojnn import maybe_split
from ojnn.io.from_datasets import Dataset


CIFAR10_MEAN_STD = (
    (0.49139965, 0.4821584, 0.4465309),
    (0.24703221, 0.24348511, 0.2615878),
)


def _convert_image_to_array(example):
    array_image = np.array(example["img"], dtype=np.int32)
    return {"arr": array_image.transpose(2, 0, 1)}


# https://github.com/huggingface/datasets/issues/5517#issuecomment-1429950390
def _as_numpy_int32(batch):
    return {key: np.asarray(val, dtype=np.int32) for key, val in batch.items()}


def get_cifar10() -> tuple[Dataset, Dataset]:
    datadir = str(Path(__file__).parent.parent.joinpath("data").resolve())
    # dsdict = load_dataset("cifar10", cache_dir=datadir, keep_in_memory=True)
    dsdict = load_dataset("cifar10", cache_dir=datadir)
    trainset, testset = dsdict["train"], dsdict["test"]
    target_features = Features(
        {
            "arr": Array3D(shape=(3, 32, 32), dtype="int32"),
            "label": trainset.features["label"],
        }
    )
    trainset = (
        trainset.map(_convert_image_to_array, remove_columns=["img"])
        .cast(target_features)
        .with_transform(_as_numpy_int32)
    )
    testset = (
        testset.map(_convert_image_to_array, remove_columns=["img"])
        .cast(target_features)
        .with_transform(_as_numpy_int32)
    )
    return Dataset(trainset), Dataset(testset)


# ========== #
# transforms #
# ========== #


# array transform functions for 2D image batches with shape ...*C*H*W


IntPair = tuple[int, int]
MaybeIntPair = None | int | IntPair


def _parse_pair(p: MaybeIntPair) -> IntPair:
    if hasattr(p, "__len__") and len(p) == 2:
        return p
    else:
        return p, p


class Transform(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rngkey_count(self) -> int:
        raise NotImplementedError


class ComposeTransform(Transform):
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = tuple(transforms)

    def __call__(self, array: Array, rngkey: Array | None = None) -> Array:
        offset = 0
        if rngkey is not None:
            maybe_split(rngkey, self.rngkey_count)
        for transform in self.transforms:
            c = transform.rngkey_count
            if c > 0:
                key = rngkey[offset : offset + c]
                offset += c
            else:
                key = None
            array = transform(array, key)
        return array

    @property
    def rngkey_count(self) -> int:
        return sum(t.rngkey_count for t in self.transforms)


# c.f.: jax.numpy.pad
class Pad(Transform):
    def __init__(
        self,
        padding: int | tuple[MaybeIntPair, MaybeIntPair],
        mode="constant",
        **kwargs,
    ):
        self.padding = self._get_padding(padding)
        self.mode = mode
        self.kwargs = kwargs

    @staticmethod
    def _get_padding(padding) -> tuple[IntPair, IntPair]:
        ph, pw = _parse_pair(padding)
        return _parse_pair(ph), _parse_pair(pw)

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        assert array.ndim >= 2
        pad_width = ((0, 0),) * (array.ndim - 2) + self.padding
        return pad(array, pad_width, self.mode, **self.kwargs)

    @property
    def rngkey_count(self) -> int:
        return 0


class ImageToArray(Transform):
    def __init__(self, max_value: int = 255):
        self.max_value = max_value

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        return array / self.max_value

    @property
    def rngkey_count(self) -> int:
        return 0


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = asjaxarray(mean)
        self.std = asjaxarray(std)

    @staticmethod
    def _align_shape(array: Array, reference: Array) -> Array:
        if array.ndim == 1 and reference.ndim >= 3:
            return expand_dims(array, (array.ndim, array.ndim + 1))
        else:
            return array

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        mean = self._align_shape(self.mean, array)
        std = self._align_shape(self.std, array)
        return (array - mean) / std

    @property
    def rngkey_count(self) -> int:
        return 0


class RandomHorizontalFlip(Transform):
    def __init__(self, flip_rate: float = 0.5):
        self.flip_rate = flip_rate

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        (rngkey,) = rngkey
        return cond(
            uniform(rngkey) < self.flip_rate,
            lambda _: flip(array, axis=-1),
            lambda _: array,
            operand=None,
        )

    @property
    def rngkey_count(self) -> int:
        return 1


class RandomVerticalFlip(Transform):
    def __init__(self, flip_rate: float = 0.5):
        self.flip_rate = flip_rate

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        (rngkey,) = rngkey
        return cond(
            uniform(rngkey) < self.flip_rate,
            lambda _: flip(array, axis=-2),
            lambda _: array,
            operand=None,
        )

    @property
    def rngkey_count(self) -> int:
        return 1


class RandomCrop(Transform):
    def __init__(self, output_size: MaybeIntPair):
        self.output_size = _parse_pair(output_size)

    def __call__(self, array: Array, rngkey: Sequence[Array] = ()) -> Array:
        (rngkey,) = rngkey
        h_out, w_out = self.output_size
        h_in, w_in = array.shape[-2:]
        h_shrink, w_shrink = h_in - h_out, w_in - w_out
        assert h_shrink >= 0, w_shrink >= 0
        top_crop, left_crop = randint(
            rngkey,
            (2,),
            asjaxarray((0, 0)),
            asjaxarray((h_shrink + 1, w_shrink + 1)),
        )
        return dynamic_slice(
            array,
            (0,) * (array.ndim - 2) + (top_crop, left_crop),
            (array.shape[:-2]) + (h_out, w_out),
        )

    @property
    def rngkey_count(self) -> int:
        return 1
