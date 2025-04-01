from collections.abc import Sequence
from typing_extensions import Self
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike, ArrayLike
from jax.numpy import zeros, asarray
from jax.lax import conv_general_dilated
from ..struct import new
from ..ftypes import parameter, config
from ..utils import KeyArg, maybe_split
from .module import Module
from .init import identity_or_orthogonal, as_grouped
from .utils import (
    IntShape,
    PaddingLike,
    as_tuple,
    get_padding,
    get_dim_and_window_shape,
    effective_kernel_size,
    conv_output_shape,
    _assert_no_key,
)


def dense(x: ArrayLike, weight: Array, bias: Array | None = None) -> Array:
    if bias is None:
        return jnp.inner(x, weight)
    else:
        return jnp.inner(x, weight) + bias


def conv(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    stride: IntShape = 1,
    padding: PaddingLike = 0,
    dilation: IntShape = 1,
    groups: int = 1,
) -> Array:
    x, weight = asarray(x), asarray(weight)
    dim = weight.ndim - 2
    stride = as_tuple(stride, dim)
    dilation = as_tuple(dilation, dim)
    padding = get_padding(padding, dim)
    batch_shape, sample_shape = x.shape[: -dim - 1], x.shape[-dim - 1 :]
    conv_out = conv_general_dilated(
        x.reshape(-1, *sample_shape),
        weight,
        stride,
        padding,
        rhs_dilation=dilation,
        feature_group_count=groups,
    )
    conv_out = conv_out.reshape(*batch_shape, *conv_out.shape[1:])
    if bias is None:
        return conv_out
    else:
        b = jnp.expand_dims(bias, range(1, dim + 1))
        return conv_out + b


class Dense(Module):
    out_features: int = config()
    dtype: DTypeLike = config()
    weight: Array = parameter()
    bias: Array | None = parameter()

    def __init__(
        self,
        out_features: int,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        if not with_bias:
            self.assign_(bias=None)
        self.assign_(
            out_features=out_features,
            dtype=dtype,
        )
        super().__init__()

    @property
    def reset_rngkey_count(self) -> int:
        return 1

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        dtype = self.dtype
        in_features, out_features = input_shape[-1], self.out_features
        output_shape = tuple(input_shape[:-1]) + (out_features,)
        rngkey = maybe_split(rngkey, 1)[0]
        weight = identity_or_orthogonal(
            (out_features, in_features), rngkey=rngkey, dtype=dtype
        )
        if self.bias is None:
            bias = None
        else:
            bias = zeros((out_features,), dtype=dtype)
        return new(self, weight=weight, bias=bias), output_shape

    def forward(
        self: Self,
        x: ArrayLike,
        _: KeyArg = None,
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, dense(x, self.weight, self.bias)


# shape = (*batch_dims, channel, *spatial_dims)
class Conv(Module):
    out_channels: int = config()
    kernel_size: tuple[int, ...] = config()
    stride: tuple[int, ...] = config()
    padding: str | tuple[tuple[int, int], ...] = config()
    dilation: tuple[int, ...] = config()
    groups: int = config()
    dtype: DTypeLike | None = config()
    conv_dim: int = config()
    weight: Array = parameter()
    bias: Array | None = parameter()

    def __init__(
        self,
        out_channels: int,
        kernel_size: IntShape,
        stride: IntShape = 1,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
        groups: int = 1,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
        conv_dim: int | None = None,
    ):
        conv_dim, kernel = get_dim_and_window_shape(conv_dim, kernel_size)
        padding = get_padding(padding, conv_dim)
        stride = as_tuple(stride, conv_dim)
        dilation = as_tuple(dilation, conv_dim)
        if not with_bias:
            self.assign_(bias=None)
        self.assign_(
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            conv_dim=conv_dim,
        )

    @property
    def reset_rngkey_count(self) -> int:
        return self.groups

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        conv_dim, c_out, dtype = self.conv_dim, self.out_channels, self.dtype
        assert len(input_shape) >= conv_dim + 1
        input_shape = tuple(input_shape)
        kernel_size = effective_kernel_size(self.kernel_size, self.dilation)
        c_in = input_shape[-conv_dim - 1]

        output_shape = conv_output_shape(
            input_shape,
            c_out,
            conv_dim,
            kernel_size,
            self.stride,
            self.padding,
        )
        rngkey = maybe_split(rngkey, 1)[0]
        weight = as_grouped(identity_or_orthogonal, self.groups)(
            (c_out, c_in, *kernel_size), rngkey, dtype
        )
        if self.bias is None:
            bias = None
        else:
            bias = jnp.zeros(c_out, dtype=dtype)
        return new(self, weight=weight, bias=bias), output_shape

    def forward(
        self: Self,
        x: ArrayLike,
        _: KeyArg = None,
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, conv(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv1d(Conv):
    def __init__(
        self,
        out_channels: int,
        kernel_size: IntShape,
        stride: IntShape = 1,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
        groups: int = 1,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            with_bias=with_bias,
            dtype=dtype,
            conv_dim=1,
        )


class Conv2d(Conv):
    def __init__(
        self,
        out_channels: int,
        kernel_size: IntShape,
        stride: IntShape = 1,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
        groups: int = 1,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            with_bias=with_bias,
            dtype=dtype,
            conv_dim=2,
        )


class Conv3d(Conv):
    def __init__(
        self,
        out_channels: int,
        kernel_size: IntShape,
        stride: IntShape = 1,
        padding: PaddingLike = 0,
        dilation: IntShape = 1,
        groups: int = 1,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            with_bias=with_bias,
            dtype=dtype,
            conv_dim=3,
        )
