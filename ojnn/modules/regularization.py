from collections.abc import Sequence
from typing_extensions import Self
import abc
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jax.lax import rsqrt
from jax.numpy import zeros, ones_like, ones, mean, square, asarray
import jax.random as jrand
from ..utils import KeyArg, maybe_split
from ..ftypes import buffer, config
from .utils import _assert_negative_axis, _assert_no_key
from .module import Module, Sequential
from .misc import ScaleShift, Standardize, standardize


def dropout(
    x: ArrayLike,
    p: float = 0.5,
    feature_dims: int | None = 1,
    rngkey: KeyArg = None,
) -> Array:
    x = asarray(x)
    rngkey = maybe_split(rngkey, 1)[0]
    keep_prob = 1 - p
    if feature_dims is None:
        mask_shape = x.shape
    elif feature_dims > 0:
        mask_shape = x.shape[-feature_dims:]
    else:
        raise ValueError("feature_dims should be None or larger than 1")
    keep_mask = jrand.bernoulli(rngkey, keep_prob, shape=mask_shape)
    return x * keep_mask / keep_prob


class Dropout(Module):
    p: float = config()
    feature_dims: int | None = config()

    def __init__(self, p: float = 0.5, feature_dims: int | None = 1):
        assert 0 <= p < 1, "drop rate should be between 0 and 1"
        self.assign_(p=p, feature_dims=feature_dims)

    @property
    def forward_rngkey_count(self) -> int:
        if self.mode == "train":
            return 1
        else:
            return 0

    def forward(
        self: Self, x: Array, rngkey: KeyArg = None
    ) -> tuple[Self, Array]:
        if self.mode == "train":
            return self, dropout(
                x,
                self.p,
                self.feature_dims,
                maybe_split(rngkey, self.forward_rngkey_count),
            )
        else:
            return self, x


class Dropout1d(Dropout):
    def __init__(self, p: float = 0.5):
        super().__init__(p, feature_dims=2)


class Dropout2d(Dropout):
    def __init__(self, p: float = 0.5):
        super().__init__(p, feature_dims=3)


class Dropout3d(Dropout):
    def __init__(self, p: float = 0.5):
        super().__init__(p, feature_dims=4)


class _NormBase(Module, metaclass=abc.ABCMeta):
    channel_dim: int = config()
    epsilon: float = config()
    affine: bool = config()
    dtype: DTypeLike | None = config()


class LayerNorm(_NormBase, Sequential):
    channel_dim: int = config()
    epsilon: float = config()
    affine: bool = config()
    dtype: DTypeLike = config()

    def __init__(
        self,
        channel_dim: int = -1,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        _assert_negative_axis(channel_dim)
        self.assign_(
            channel_dim=channel_dim,
            epsilon=epsilon,
            affine=affine,
            dtype=dtype,
        )
        if affine:
            Sequential.__init__(
                self,
                Standardize(tuple(range(channel_dim, 0)), epsilon=epsilon),
                ScaleShift(axis=(channel_dim,), dtype=dtype),
            )
        else:
            Sequential.__init__(
                self,
                Standardize(tuple(range(channel_dim, 0)), epsilon=epsilon),
            )


class _GroupStandardize(Module):
    channel_dim: int = config()
    num_groups: int = config()
    epsilon: float = config()

    def __init__(self, channel_dim: int, num_groups: int, epsilon: float):
        assert epsilon >= 0.0, "epsilon need to be positive"
        self.assign_(
            channel_dim=channel_dim,
            num_groups=num_groups,
            epsilon=epsilon,
        )

    def reset(
        self: Self, input_shape: Sequence[int], _=None
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        out_shape = tuple(input_shape)
        c, g = out_shape[self.channel_dim], self.num_groups
        msg = f"channel size {c} not divisible by group size {g}"
        assert c % g == 0, msg
        return self, out_shape

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        x = asarray(x)
        _assert_no_key(_)
        cdim, g = self.channel_dim, self.num_groups
        shp = x.shape
        hshp = shp[:cdim]
        csz = shp[cdim]
        tshp = () if cdim == -1 else shp[cdim + 1 :]
        return self, standardize(
            x.reshape(*hshp, g, csz // g, *tshp),
            axis=tuple(range(-len(tshp) - 1, 0)),
            epsilon=self.epsilon,
        ).reshape(*hshp, csz, *tshp)


class GroupNorm(_NormBase, Sequential):
    channel_dim: int = config()
    num_groups: int = config()
    epsilon: float = config()
    affine: bool = config()
    dtype: DTypeLike = config()

    def __init__(
        self,
        channel_dim: int = -1,
        num_groups: int = 32,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        _assert_negative_axis(channel_dim)
        self.assign_(
            channel_dim=channel_dim,
            num_groups=num_groups,
            epsilon=epsilon,
            affine=affine,
            dtype=dtype,
        )
        if affine:
            Sequential.__init__(
                self,
                _GroupStandardize(channel_dim, num_groups, epsilon=epsilon),
                ScaleShift(axis=(channel_dim,), dtype=dtype),
            )
        else:
            Sequential.__init__(
                self,
                _GroupStandardize(channel_dim, num_groups, epsilon=epsilon),
            )


class GroupNorm1d(GroupNorm):
    def __init__(
        self,
        num_groups: int = 32,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-2, num_groups, epsilon, affine, dtype)


class GroupNorm2d(GroupNorm):
    def __init__(
        self,
        num_groups: int = 32,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-3, num_groups, epsilon, affine, dtype)


class GroupNorm3d(GroupNorm):
    def __init__(
        self,
        num_groups: int = 32,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-4, num_groups, epsilon, affine, dtype)


class InstanceNorm(_NormBase, Sequential):
    channel_dim: int = config()
    epsilon: float = config()
    affine: bool = config()
    dtype: DTypeLike | None = config()

    def __init__(
        self,
        channel_dim: int,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        assert channel_dim + 1 < 0, "channel_dim should be smaller than -1"
        self.assign_(
            channel_dim=channel_dim,
            epsilon=epsilon,
            affine=affine,
            dtype=dtype,
        )
        if affine:
            Sequential.__init__(
                self,
                Standardize(tuple(range(channel_dim + 1, 0)), epsilon=epsilon),
                ScaleShift(axis=(channel_dim,), dtype=dtype),
            )
        else:
            Sequential.__init__(
                self,
                Standardize(tuple(range(channel_dim, 0)), epsilon=epsilon),
            )


class InstanceNorm1d(InstanceNorm):
    def __init__(
        self,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-2, epsilon, affine, dtype)


class InstanceNorm2d(InstanceNorm):
    def __init__(
        self,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-3, epsilon, affine, dtype)


class InstanceNorm3d(InstanceNorm):
    def __init__(
        self,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-4, epsilon, affine, dtype)


def _avg_var_over_axis(x: Array, axis: int) -> tuple[Array, Array]:
    axis = x.ndim + axis if axis < 0 else axis
    reduce_axes = tuple(range(axis)) + tuple(range(axis + 1, x.ndim))
    avg = mean(x, reduce_axes, keepdims=True)
    total, dof = x.size, x.size - avg.size
    if dof != 0:
        var = mean(square(x - avg), reduce_axes, keepdims=True) * total / dof
    else:
        var = ones_like(avg)  # no info about var, maintain unit scale
    return avg, var


def _avg_var_ema(
    avg: Array, var: Array, ravg: Array, rvar: Array, momentum: float
) -> tuple[Array, Array, Array, Array]:
    ravg = momentum * ravg + (1 - momentum) * avg.flatten()
    rvar = momentum * rvar + (1 - momentum) * var.flatten()
    return avg, var, ravg, rvar


class _BatchNormStandardize(Module):
    momentum: float | None = config()
    epsilon: float = config()
    channel_dim: int = config()
    dtype: DTypeLike | None = config()
    running_avg: Array = buffer()
    running_var: Array = buffer()

    def __init__(
        self,
        momentum: float | None = 0.9,
        epsilon: float = 1e-5,
        channel_dim: int = -1,
        dtype: DTypeLike | None = None,
    ):
        assert epsilon >= 0.0, "epsilon need to be positive"
        if momentum is None:
            self.assign_(
                momentum=momentum,
                epsilon=epsilon,
                channel_dim=channel_dim,
                dtype=dtype,
                running_avg=None,
                running_var=None,
            )
        else:
            self.assign_(
                momentum=momentum,
                epsilon=epsilon,
                channel_dim=channel_dim,
                dtype=dtype,
            )

    def reset(
        self: Self, input_shape: Sequence[int], _=None
    ) -> tuple[Self, tuple[int, ...]]:
        _assert_no_key(_)
        if self.momentum is None:
            return self, tuple(input_shape)
        dtype = self.dtype
        channels = input_shape[self.channel_dim]
        running_avg = zeros([channels], dtype=dtype)
        running_var = ones([channels], dtype=dtype)
        return self.update(
            running_avg=running_avg, running_var=running_var
        ), tuple(input_shape)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        x = asarray(x)
        momentum, axis = self.momentum, self.channel_dim
        shape = list((1,) * x.ndim)
        shape[axis] = -1
        if momentum is None:
            avg, var = _avg_var_over_axis(x, axis)
            return self, (x - avg) * rsqrt(var + self.epsilon)
        else:
            ravg, rvar = self.running_avg, self.running_var
            if self.mode == "train":
                avg, var, ravg, rvar = _avg_var_ema(
                    *_avg_var_over_axis(x, axis), ravg, rvar, momentum=momentum
                )
            else:
                avg, var, ravg, rvar = (
                    ravg.reshape(*shape),
                    rvar.reshape(*shape),
                    ravg,
                    rvar,
                )
            outputs = (x - avg) * rsqrt(var + self.epsilon)
            return self.update(running_avg=ravg, running_var=rvar), outputs


class BatchNorm(_NormBase, Sequential):
    channel_dim: int = config()
    momentum: float | None = config()
    epsilon: float = config()
    affine: bool = config()
    dtype: DTypeLike | None = config()

    def __init__(
        self,
        channel_dim: int = -1,
        momentum: float | None = 0.9,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        _assert_negative_axis(channel_dim)
        self.assign_(
            channel_dim=channel_dim,
            momentum=momentum,
            epsilon=epsilon,
            affine=affine,
            dtype=dtype,
        )
        if affine:
            Sequential.__init__(
                self,
                _BatchNormStandardize(
                    momentum=momentum,
                    epsilon=epsilon,
                    channel_dim=channel_dim,
                    dtype=dtype,
                ),
                ScaleShift(axis=(channel_dim,), dtype=dtype),
            )
        else:
            Sequential.__init__(
                self,
                _BatchNormStandardize(
                    momentum=momentum,
                    epsilon=epsilon,
                    channel_dim=channel_dim,
                    dtype=dtype,
                ),
            )


class BatchNorm1d(BatchNorm):
    def __init__(
        self,
        momentum: float | None = 0.9,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-2, momentum, epsilon, affine, dtype)


class BatchNorm2d(BatchNorm):
    def __init__(
        self,
        momentum: float | None = 0.9,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-3, momentum, epsilon, affine, dtype)


class BatchNorm3d(BatchNorm):
    def __init__(
        self,
        momentum: float | None = 0.9,
        epsilon: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ):
        super().__init__(-4, momentum, epsilon, affine, dtype)
