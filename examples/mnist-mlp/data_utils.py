from typing import Sequence
from os.path import join as opjoin, exists as opexists
import abc
import gzip
import pathlib
import urllib.request
import numpy as np
from numpy import ndarray
from jax import Array
from jax.numpy import (
    asarray as asjaxarray,
    expand_dims,
    flip,
    pad,
)
from jax.random import randint, uniform
from jax.lax import dynamic_slice, cond
from ojnn import maybe_split
from ojnn.io import NumpyDataset


def url_download(url: str, save_to: str, overwrites: bool = False) -> None:
    if opexists(save_to) and not overwrites:
        print(f"Download skipped: {save_to} already exists.")
        return
    print(f"Downloading from {url} to {save_to} ... ", end="")
    urllib.request.urlretrieve(url, save_to)
    print("done.")


def mkdir(dirpath: str, parents: bool = True, exist_ok: bool = True) -> None:
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


# ======== #
# datasets #
# ======== #


class MNIST(NumpyDataset):
    # URL = "http://yann.lecun.com/exdb/mnist/"  # no longer works
    URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    TRAIN_IMAGE = "train-images-idx3-ubyte"
    TRAIN_LABEL = "train-labels-idx1-ubyte"
    TEST_IMAGE = "t10k-images-idx3-ubyte"
    TEST_LABEL = "t10k-labels-idx1-ubyte"
    EXT_COMPRESS = ".gz"
    EXT_NDARRAY = ".npy"
    DATA_FOLDER = "mnist"
    MEAN = 0.13066046
    STD = 0.30810785

    def __init__(self, data_dir: str, train: bool, download: bool = False):
        self.data_dir = opjoin(data_dir, self.DATA_FOLDER)
        self.train = train
        self.download = download

        if download:
            mkdir(self.data_dir)
        imgs, labels = self._get_data()
        super().__init__(image=imgs, label=labels)

    def _get_data(self) -> tuple[ndarray, ndarray]:
        if self.train:
            imgname, lblname = self.TRAIN_IMAGE, self.TRAIN_LABEL
        else:
            imgname, lblname = self.TEST_IMAGE, self.TEST_LABEL
        return np.expand_dims(
            self._get_array(imgname), axis=1
        ), self._get_array(lblname)

    def _get_array(self, name) -> ndarray:
        array_path = opjoin(self.data_dir, name + self.EXT_NDARRAY)
        if not opexists(array_path):
            compress_path = opjoin(self.data_dir, name + self.EXT_COMPRESS)
            if not opexists(compress_path):
                if self.download:
                    url = opjoin(self.URL, name + self.EXT_COMPRESS)
                    url_download(url, compress_path)
                else:
                    raise FileNotFoundError(f"{compress_path} is missing.")
            else:
                pass
            np.save(array_path, self._gz2npy(compress_path))
        else:
            pass
        return np.load(array_path)

    @staticmethod
    def _as_int(b: bytes) -> int:
        return int.from_bytes(b, "big")

    @staticmethod
    def _gz2npy(gzfile: str) -> ndarray:
        with gzip.open(gzfile, "rb") as gzipfp:
            data = gzipfp.read()
            magic = MNIST._as_int(data[0:4])
            ndim = magic % 256
            shape = [
                MNIST._as_int(data[4 * (i + 1) : 4 * (i + 2)])
                for i in range(ndim)
            ]
        return np.frombuffer(data, np.uint8, offset=4 * (ndim + 1)).reshape(
            *shape
        )


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
