from typing import TypeAlias
from collections.abc import Sequence, Callable
from math import sqrt, prod
from jax.typing import ArrayLike, DTypeLike
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
import jax.nn.initializers as jnninit
from ..utils import KeyArray, maybe_split


InitFn: TypeAlias = Callable[..., Array]


_activfn_gains: dict[str, float | Callable[..., float]] = {
    "sigmoid": 1.0,
    "tanh": 5.0 / 3,
    "relu": sqrt(2),
    "leaky_relu": lambda negative_slope: sqrt(2.0 / (1 + negative_slope**2)),
}


def calculate_gain(activfn: str, args=None) -> float:
    if activfn not in _activfn_gains:
        raise NotImplementedError(f"Unsupported activation {activfn}")
    else:
        gain = _activfn_gains[activfn]
        if callable(gain):
            return gain(args)
        else:
            return gain


# 2d init related, for shape (out_channels, in_channels)


def eye(c_out: int, c_in: int, dtype: DTypeLike | None = None) -> Array:
    return jnp.eye(c_out, c_in, dtype=dtype)


# references:
# - "Exact solutions to the nonlinear dynamics of learning in deep linear
#   neural networks", Saxe et al., ICLR 2014
def orthogonal(
    c_out: int, c_in: int, rngkey: KeyArray, dtype: DTypeLike | None = None
) -> Array:
    return jnninit.orthogonal(column_axis=-2, dtype=dtype)(
        rngkey, (c_out, c_in)
    )


# extends to work with shape (out_channels, in_channels, *spatial_shape)
def _dirac_extended(init_fn_2d: InitFn):
    def _dirac_extended_fn(
        shape: Sequence[int], *args, dtype=None, **kwargs
    ) -> Array:
        c_out, c_in = shape[:2]
        spatial_shape = shape[2:]
        init2d = init_fn_2d(c_out, c_in, *args, **kwargs).reshape(
            c_out, c_in, *((1,) * len(spatial_shape))
        )
        kernel_center = tuple((s - 1) // 2 for s in spatial_shape)
        delta_kernel = (
            jnp.zeros(spatial_shape, dtype=dtype).at[kernel_center].set(1)
        )
        return jnp.kron(init2d, delta_kernel)

    return _dirac_extended_fn


# assumes shape = (out_channels, in_channels, *spatial_shape)


def as_grouped(init_fn: InitFn, groups: int) -> InitFn:
    def _grouped_fn(
        shape: Sequence[int],
        rngkey: KeyArray,
        *args,
        **kwargs,
    ) -> Array:
        c_out, c_in = shape[:2]
        spatial_shape = shape[2:]
        assert c_out % groups == 0, "output channels not divisible by groups"
        assert c_in % groups == 0, "input channels not divisible by groups"
        assert all(s > 0 for s in spatial_shape)
        keys = maybe_split(rngkey, groups)
        pergroup_shape = (c_out // groups, c_in // groups, *spatial_shape)
        return jnp.concatenate(
            [init_fn(pergroup_shape, k, *args, **kwargs) for k in keys],
            axis=0,
        )

    return _grouped_fn


# references:
# - "Gradient Descent with Identity Initialization Efficiently Learns
#   Positive-Definite Linear Transformations by Deep Residual Networks",
#   Barlett et al., Neural Comput. 2019
# - "Dynamical Isometry and a Mean Field Theory of CNNs: How to Train
#   10,000-Layer Vanilla Convolutional Neural Networks", Xiao et al., ICML 2018
def identity(
    shape: Sequence[int] | int, dtype: DTypeLike | None = None
) -> Array:
    return _dirac_extended(eye)(shape, dtype)


def _normal_fan_in(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return jnninit.variance_scaling(
        1.0, "fan_in", "normal", in_axis=1, out_axis=0, dtype=dtype
    )(rngkey, shape)


def _normal_fan_avg(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return jnninit.variance_scaling(
        1.0, "fan_avg", "normal", in_axis=1, out_axis=0, dtype=dtype
    )(rngkey, shape)


def _uniform_fan_in(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return jnninit.variance_scaling(
        1.0, "fan_in", "uniform", in_axis=1, out_axis=0, dtype=dtype
    )(rngkey, shape)


def _uniform_fan_avg(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return jnninit.variance_scaling(
        1.0, "fan_avg", "uniform", in_axis=1, out_axis=0, dtype=dtype
    )(rngkey, shape)


def lecun_normal(
    shape: Sequence[int],
    rngkey: KeyArray,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return gain * _normal_fan_in(shape, rngkey, dtype)


def lecun_uniform(
    shape: Sequence[int],
    rngkey: KeyArray,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return gain * _uniform_fan_in(shape, rngkey, dtype)


def glorot_normal(
    shape: Sequence[int],
    rngkey: KeyArray,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return gain * _normal_fan_avg(shape, rngkey, dtype)


def glorot_uniform(
    shape: Sequence[int],
    rngkey: KeyArray,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return gain * _uniform_fan_avg(shape, rngkey, dtype)


def he_normal(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return sqrt(2) * _normal_fan_in(shape, rngkey, dtype)


def he_uniform(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    return sqrt(2) * _uniform_fan_in(shape, rngkey, dtype)


def _identity_or_widen(
    widen_init_fn: InitFn,
    shape: Sequence[int],
    *args,
    dtype: DTypeLike | None = None,
    **kwargs,
) -> Array:
    c_out, c_in = shape[:2]
    if c_out <= c_in:
        return identity(shape, dtype)
    else:
        return widen_init_fn(shape, *args, dtype=dtype, **kwargs)


def identity_or_orthogonal(
    shape: Sequence[int],
    rngkey: KeyArray,
    dtype: DTypeLike | None = None,
) -> Array:
    def orthogonal_dirac(shp, key, *, dtype):
        return _dirac_extended(orthogonal)(shp, key, dtype)

    return _identity_or_widen(orthogonal_dirac, shape, rngkey, dtype=dtype)


# generic shape init


def zeros(shape: Sequence[int], dtype: DTypeLike | None = None) -> Array:
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: DTypeLike | None = None) -> Array:
    return jnp.ones(shape, dtype=dtype)


def constant(
    shape: Sequence[int], fill_value: ArrayLike, dtype: DTypeLike | None = None
) -> Array:
    return jnp.full(shape, fill_value, dtype=dtype)


def uniform(
    shape: Sequence[int],
    rngkey: KeyArray,
    minval: ArrayLike = 0.0,
    maxval: ArrayLike = 1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return jrand.uniform(rngkey, shape, jnp.dtype(dtype), minval, maxval)


def normal(
    shape: Sequence[int],
    rngkey: KeyArray,
    std=1.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return jrand.normal(rngkey, shape, jnp.dtype(dtype)) * jnp.array(
        std, dtype
    )


def truncated_normal(
    shape: Sequence[int],
    rngkey: KeyArray,
    std=1.0,
    minval: ArrayLike = -2.0,
    maxval: ArrayLike = 2.0,
    dtype: DTypeLike | None = None,
) -> Array:
    return jnninit.truncated_normal(std, dtype, minval, maxval)(rngkey, shape)
