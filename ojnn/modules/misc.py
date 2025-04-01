from collections.abc import Callable, Sequence
from typing import overload, Literal, Generic
from typing_extensions import Self
from math import prod
from jax import Array
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
import jax.nn as jnn
from ..ftypes import parameter, config, buffer, const
from ..struct import new
from ..utils import Axis
from .module import Module, DType
from .utils import _assert_no_key
from .init import zeros, ones


# regurgitate jax.nn other functions to make pickling work ... LOL


@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis,
    b: ArrayLike | None,
    keepdims: bool,
    return_sign: Literal[False],
    where: ArrayLike | None,
) -> Array: ...


@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis,
    b: ArrayLike | None,
    keepdims: bool,
    return_sign: Literal[True],
    where: ArrayLike | None,
) -> tuple[Array, Array]: ...


def logsumexp(
    a, axis=None, b=None, keepdims=False, return_sign=False, where=None
):
    return jnn.logsumexp(
        a,
        axis=axis,
        b=b,
        keepdims=keepdims,
        return_sign=return_sign,
        where=where,
    )


def standardize(
    x: ArrayLike,
    axis: Axis = -1,
    mean: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    epsilon: ArrayLike = 1e-5,
    where: ArrayLike | None = None,
) -> Array:
    return jnn.standardize(
        x,
        axis=axis,
        mean=mean,
        variance=variance,
        epsilon=epsilon,
        where=where,
    )


def one_hot(
    x: ArrayLike,
    num_classes: int,
    *,
    dtype: DTypeLike | None = None,
    axis: int | tuple[int, ...] = -1,
) -> Array:
    return jnn.one_hot(x, num_classes=num_classes, dtype=dtype, axis=axis)


def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: ArrayLike | None = None,
    key_value_seq_lengths: ArrayLike | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    implementation: Literal["xla", "cudnn"] | None = None,
) -> Array:
    return jnn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=key_value_seq_lengths,
        local_window_size=local_window_size,
        implementation=implementation,
    )


# some utility modules


class Identity(Module, Generic[DType]):
    def forward(self: Self, x: DType, _=None) -> tuple[Self, DType]:
        _assert_no_key(_)
        return self, x


def _id_shape(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(shape)


ForwardFn = Callable[[ArrayLike], Array]
ShapeFn = Callable[[Sequence[int]], tuple[int, ...]]


class Lambda(Module):
    forward_fn: ForwardFn = config()
    shape_fn: ShapeFn = config()

    def __init__(self, forward_fn: ForwardFn, shape_fn: ShapeFn = _id_shape):
        self.assign_(forward_fn=forward_fn, shape_fn=shape_fn)

    def reset(
        self: Self, input_shape: Sequence[int], _=None
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        return self, staticmethod(self.shape_fn)(input_shape)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, staticmethod(self.forward_fn)(x)


class Flatten(Module):
    flatten_dims: int | None = config()

    def __init__(self, flatten_dims: int | None = None):
        self.assign_(flatten_dims=flatten_dims)

    def reset(
        self: Self, input_shape: Sequence[int], _=None
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        fd = self.flatten_dims
        if fd is None:  # flatten all
            return self, (prod(input_shape),)
        else:
            if fd < 2:  # no need to flatten
                return self, tuple(input_shape)
            else:  # partial flatten
                return self, tuple(input_shape[:-fd]) + (
                    prod(input_shape[-fd:]),
                )

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        x = jnp.asarray(x)
        fd = self.flatten_dims
        if fd is None:  # flatten all
            return self, x.flatten()
        else:
            if fd < 2:  # no need to flatten
                return self, x
            else:  # partial flatten
                return self, x.reshape(*x.shape[:-fd], -1)


class Flatten1d(Flatten):
    def __init__(self):
        super().__init__(flatten_dims=2)


class Flatten2d(Flatten):
    def __init__(self):
        super().__init__(flatten_dims=3)


class Flatten3d(Flatten):
    def __init__(self):
        super().__init__(flatten_dims=4)


class Standardize(Module):
    axis: tuple[int, ...] = config()
    epsilon: float = config()

    def __init__(self, axis: Sequence[int], epsilon: float = 1e-5):
        assert epsilon >= 0.0, "epsilon need to be positive"
        self.assign_(axis=tuple(axis), epsilon=epsilon)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, standardize(x, axis=self.axis, epsilon=self.epsilon)


class ScaleShift(Module):
    axis: tuple[int, ...] = config()
    dtype: DTypeLike | None = config()
    weight: Array = parameter()
    bias: Array | None = parameter()

    def __init__(
        self,
        axis: Sequence[int],
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        if not with_bias:
            self.assign_(bias=None)
        self.assign_(axis=tuple(axis), dtype=dtype)

    def reset(
        self: Self, input_shape: Sequence[int], _=None
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        out_shape = tuple(input_shape)
        dim = len(out_shape)
        axis = [i - dim if i >= 0 else i for i in self.axis]
        first_i = min(axis)
        param_shape = [1] * (-first_i)
        for i in axis:
            param_shape[i - first_i] = out_shape[i]
        weight = ones(param_shape, self.dtype)
        if self.bias is None:
            bias = None
        else:
            bias = zeros(param_shape, self.dtype)
        return new(self, weight=weight, bias=bias), out_shape

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        bias = self.bias
        if bias is None:
            return self, self.weight * x
        else:
            return self, self.weight * x + bias


class TrackAverage(Module):
    dtype: DTypeLike | None = config()
    count: int = buffer(default=0)
    average: Array = buffer()

    def __init__(self, dtype: DTypeLike | None = None):
        self.assign_(dtype=dtype)

    def reset(
        self: Self,
        input_shape: Sequence[int],
        _=None,
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        output_shape = tuple(input_shape)
        return (
            new(
                self,
                count=0,
                average=jnp.zeros(output_shape, dtype=self.dtype),
            ),
            output_shape,
        )

    def _welford_update(self: Self, val: Array) -> Self:
        count, avg = self.count, self.average
        count = count + 1
        avg = avg + (val - avg) / count
        return self.update(count=count, average=avg)

    def forward(
        self: Self,
        input_value: ArrayLike,
        _=None,
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        v = jnp.asarray(input_value)
        return self._welford_update(v), v


class TrackEMA(Module):
    momentum: float = config()
    dtype: DTypeLike | None = config()
    init_value: ArrayLike = const()
    average: Array = buffer()

    def __init__(
        self,
        momentum: float,
        init_value: ArrayLike = 0.0,
        dtype: DTypeLike | None = None,
    ):
        self.assign_(momentum=momentum, init_value=init_value, dtype=dtype)

    def reset(
        self: Self,
        input_shape: Sequence[int],
        _=None,
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        output_shape = tuple(input_shape)
        return (
            new(
                self,
                average=jnp.zeros(output_shape, dtype=self.dtype)
                + self.init_value,
            ),
            output_shape,
        )

    def _ema_update(self: Self, x: Array) -> Self:
        m, v = self.momentum, self.average
        v = m * v + (1.0 - m) * x
        return self.update(average=v)

    def forward(
        self: Self,
        input_value: ArrayLike,
        _=None,
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        v = jnp.asarray(input_value)
        return self._ema_update(v), v


class TrackMeanStd(Module):
    dtype: DTypeLike | None = config()
    count: int = buffer(default=0)
    mean: Array = buffer()
    _sumsqdiff: Array = buffer()

    def __init__(self, dtype: DTypeLike | None = None):
        self.assign_(dtype=dtype)

    def reset(
        self: Self,
        input_shape: Sequence[int],
        _=None,
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        output_shape = tuple(input_shape)
        return (
            new(
                self,
                count=0,
                mean=jnp.zeros(output_shape, dtype=self.dtype),
                _sumsqdiff=jnp.zeros(output_shape, dtype=self.dtype),
            ),
            output_shape,
        )

    def _welford_update(self: Self, val: Array) -> Self:
        count, mean, sumsqdiff = self.count, self.mean, self._sumsqdiff
        count = count + 1
        diff_old = val - mean
        mean = mean + diff_old / count
        diff_new = val - mean
        sumsqdiff = sumsqdiff + diff_old * diff_new
        return self.update(
            count=count,
            mean=mean,
            _sumsqdiff=sumsqdiff,
        )

    def forward(
        self: Self,
        input_value: ArrayLike,
        _=None,
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        v = jnp.asarray(input_value)
        return self._welford_update(v), v

    @property
    def variance(self) -> Array:
        return self._sumsqdiff / self.count

    @property
    def sample_variance(self) -> Array:
        return self._sumsqdiff / jnp.maximum(self.count, 0)

    @property
    def std(self) -> Array:
        return jnp.sqrt(self.variance)

    @property
    def sample_std(self) -> Array:
        return jnp.sqrt(self.sample_variance)
