from collections.abc import Callable, Sequence
from typing import Optional, TypeVar, TypeAlias
from typing_extensions import ClassVar, Self
from math import prod
from jax import Array
from jax.lax import reduce_window, add, min, max
from jax.numpy import (
    asarray as jnp_asarray,
    mean as jnp_mean,
    sum as jnp_sum,
    prod as jnp_prod,
    amin as jnp_amin,
    amax as jnp_amax,
)
from jax.typing import ArrayLike
from ..ftypes import config
from .module import Module
from .utils import (
    IntShape,
    PaddingLike,
    get_dim_and_window_shape,
    get_padding,
    as_tuple,
    conv_output_shape,
    effective_kernel_size,
    _assert_no_key,
)


# currently supported pooling ops with full jit + ad support
# cf.: https://github.com/google/jax/issues/7815


_T = TypeVar("_T")
_IntTuple: TypeAlias = tuple[int, ...]
PoolingFn: TypeAlias = Callable[
    [
        ArrayLike,
        _IntTuple,
        _IntTuple,
        str | tuple[tuple[int, int], ...],
        _IntTuple,
    ],
    Array,
]


def _make_pool(
    op: Callable[[ArrayLike, ArrayLike], Array], zero: ArrayLike
) -> PoolingFn:
    def _pool(
        x: ArrayLike,
        window_size: _IntTuple,
        stride: _IntTuple,
        padding: str | tuple[tuple[int, int], ...],
        dilation: _IntTuple,
    ) -> Array:
        x = jnp_asarray(x)
        extra_ndim = x.ndim - len(window_size)
        window_size = (1,) * extra_ndim + window_size
        stride = (1,) * extra_ndim + stride
        dilation = (1,) * extra_ndim + dilation
        if isinstance(padding, str):
            padding = padding
        else:
            padding = ((0, 0),) * extra_ndim + padding
        return reduce_window(
            x, zero, op, window_size, stride, padding, window_dilation=dilation
        )

    return _pool


def _sum_pool(
    x: ArrayLike,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    dilation: tuple[int, ...],
) -> Array:
    return _make_pool(add, 0)(x, window_size, stride, padding, dilation)


def _avg_pool(
    x: ArrayLike,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    dilation: tuple[int, ...],
) -> Array:
    window_numel = prod(window_size)
    return (
        _make_pool(add, 0)(x, window_size, stride, padding, dilation)
        / window_numel
    )


def _min_pool(
    x: ArrayLike,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    dilation: tuple[int, ...],
) -> Array:
    return _make_pool(min, float("inf"))(
        x, window_size, stride, padding, dilation
    )


def _max_pool(
    x: ArrayLike,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    dilation: tuple[int, ...],
) -> Array:
    return _make_pool(max, float("-inf"))(
        x, window_size, stride, padding, dilation
    )


class Pool(Module):
    pooling_ops: ClassVar[dict[str, PoolingFn]] = {
        "sum": _sum_pool,
        "avg": _avg_pool,
        "min": _min_pool,
        "max": _max_pool,
    }
    pooling_op: str = config()
    window_size: tuple[int, ...] = config()
    stride: tuple[int, ...] = config()
    padding: str | tuple[tuple[int, int], ...] = config()
    dilation: tuple[int, ...] = config()
    dim: int = config()

    def __init__(
        self,
        pooling_op: str,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
        dim: Optional[int] = None,
    ):
        assert pooling_op in self.pooling_ops
        dim, window_size = get_dim_and_window_shape(dim, window_size)
        padding = get_padding(padding, dim)
        dilation = as_tuple(dilation, dim)
        if stride is None:
            stride = effective_kernel_size(window_size, dilation)
        else:
            stride = as_tuple(stride, dim)
        self.assign_(
            window_size=window_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pooling_op=pooling_op,
            dim=dim,
        )

    def reset(
        self: Self,
        input_shape: Sequence[int],
        _=None,
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        dim = self.dim

        input_shape = tuple(input_shape)
        window_size = effective_kernel_size(self.window_size, self.dilation)

        output_shape = conv_output_shape(
            input_shape,
            input_shape[-dim - 1],
            dim,
            window_size,
            self.stride,
            self.padding,
        )
        return self, output_shape

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, self.pooling_ops[self.pooling_op](
            x,
            self.window_size,
            self.stride,
            self.padding,
            self.dilation,
        )


class AvgPool1d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("avg", window_size, stride, padding, dilation, 1)


class AvgPool2d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("avg", window_size, stride, padding, dilation, 2)


class AvgPool3d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("avg", window_size, stride, padding, dilation, 3)


class MaxPool1d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("max", window_size, stride, padding, dilation, 1)


class MaxPool2d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("max", window_size, stride, padding, dilation, 2)


class MaxPool3d(Pool):
    def __init__(
        self,
        window_size: IntShape,
        stride: IntShape | None = None,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
    ):
        super().__init__("max", window_size, stride, padding, dilation, 3)


class GlobalPool(Module):
    _reduce_fns: ClassVar[dict[str, Callable]] = {
        "sum": jnp_sum,
        "avg": jnp_mean,
        "prod": jnp_prod,
        "min": jnp_amin,
        "max": jnp_amax,
    }
    reduce_fn: Callable = config()
    axis: tuple[int, ...] = config()
    keepdims: bool = config()

    def __init__(
        self,
        reduce_fn: str | Callable,
        axis: tuple[int, ...] | int,
        keepdims: bool = False,
    ):
        if isinstance(axis, int):
            axis = (axis,)
        assert all(i < 0 for i in axis), "only negative indices are allowed"
        if isinstance(reduce_fn, str):
            fn = self._reduce_fns[reduce_fn]
        else:
            fn = reduce_fn
        self.assign_(reduce_fn=fn, axis=axis, keepdims=keepdims)

    def reset(
        self: Self,
        input_shape: Sequence[int],
        _=None,
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        keepdims = self.keepdims
        ndim = len(input_shape)
        axis = [ndim + i for i in self.axis]
        out_shape = []
        for i, s in enumerate(input_shape):
            if i in axis:
                if keepdims:
                    out_shape.append(1)
                else:
                    pass
            else:
                out_shape.append(s)
        return self, tuple(out_shape)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, staticmethod(self.reduce_fn)(
            x, self.axis, keepdims=self.keepdims
        )


class GlobalAvgPool1d(GlobalPool):
    def __init__(self, keepdims: bool = False):
        super().__init__(reduce_fn="avg", axis=(-1), keepdims=keepdims)


class GlobalAvgPool2d(GlobalPool):
    def __init__(self, keepdims: bool = False):
        super().__init__(reduce_fn="avg", axis=(-2, -1), keepdims=keepdims)


class GlobalAvgPool3d(GlobalPool):
    def __init__(self, keepdims: bool = False):
        super().__init__(reduce_fn="avg", axis=(-3, -2, -1), keepdims=keepdims)
