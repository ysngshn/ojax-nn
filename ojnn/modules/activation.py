from typing_extensions import Self
from jax import Array
from jax.typing import ArrayLike
import jax.nn as jnn
from ..ftypes import config
from ..utils import Axis
from .utils import _assert_no_key, _assert_negative_axis
from .module import Module
from .misc import Lambda


# regurgitate jax.nn activation functions to make pickling work ... LOL


def relu(x: ArrayLike) -> Array:
    return jnn.relu(x)


def relu6(x: ArrayLike) -> Array:
    return jnn.relu6(x)


def sigmoid(x: ArrayLike) -> Array:
    return jnn.sigmoid(x)


def softplus(x: ArrayLike) -> Array:
    return jnn.softplus(x)


def sparse_plus(x: ArrayLike) -> Array:
    return jnn.sparse_plus(x)


def sparse_sigmoid(x: ArrayLike) -> Array:
    return jnn.softplus(x)


def soft_sign(x: ArrayLike) -> Array:
    return jnn.softplus(x)


def silu(x: ArrayLike) -> Array:
    return jnn.silu(x)


def log_sigmoid(x: ArrayLike) -> Array:
    return jnn.log_sigmoid(x)


def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 0.01) -> Array:
    return jnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x: ArrayLike) -> Array:
    return jnn.hard_sigmoid(x)


def hard_silu(x: ArrayLike) -> Array:
    return jnn.hard_silu(x)


def hard_tanh(x: ArrayLike) -> Array:
    return jnn.hard_tanh(x)


def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array:
    return jnn.elu(x, alpha=alpha)


def celu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array:
    return jnn.celu(x, alpha=alpha)


def selu(x: ArrayLike) -> Array:
    return jnn.selu(x)


def gelu(x: ArrayLike, approximate: bool = True) -> Array:
    return jnn.gelu(x, approximate=approximate)


def glu(x: ArrayLike, axis: int = -1) -> Array:
    return jnn.glu(x, axis=axis)


def squareplus(x: ArrayLike, b: ArrayLike = 4) -> Array:
    return jnn.squareplus(x, b=b)


def mish(x: ArrayLike) -> Array:
    return jnn.mish(x)


def tanh(x: ArrayLike) -> Array:
    return jnn.tanh(x)


def softmax(
    x: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None,
) -> Array:
    return jnn.softmax(x, axis=axis, where=where)


def log_softmax(
    x: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None,
) -> Array:
    return jnn.log_softmax(x, axis=axis, where=where)


# Module wrappers for activation functions


class ReLU(Lambda):
    def __init__(self):
        super().__init__(relu)


class ReLU6(Lambda):
    def __init__(self):
        super().__init__(relu6)


class Sigmoid(Lambda):
    def __init__(self):
        super().__init__(sigmoid)


class Softplus(Lambda):
    def __init__(self):
        super().__init__(softplus)


class SparsePlus(Lambda):
    def __init__(self):
        super().__init__(sparse_plus)


class SparseSigmoid(Lambda):
    def __init__(self):
        super().__init__(sparse_sigmoid)


class SoftSign(Lambda):
    def __init__(self):
        super().__init__(soft_sign)


class SiLU(Lambda):
    def __init__(self):
        super().__init__(silu)


class LogSigmoid(Lambda):
    def __init__(self):
        super().__init__(log_sigmoid)


class LeakyReLU(Module):
    negative_slope: float = config()

    def __init__(self, negative_slope: float = 0.01):
        self.assign_(negative_slope=negative_slope)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, leaky_relu(x, negative_slope=self.negative_slope)


class HardSigmoid(Lambda):
    def __init__(self):
        super().__init__(hard_sigmoid)


class HardSiLU(Lambda):
    def __init__(self):
        super().__init__(hard_silu)


class HardTanh(Lambda):
    def __init__(self):
        super().__init__(hard_tanh)


class ELU(Module):
    alpha: float = config()

    def __init__(self, alpha: float = 1.0):
        self.assign_(alpha=alpha)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, elu(x, alpha=self.alpha)


class CELU(Module):
    alpha: float = config()

    def __init__(self, alpha: float = 1.0):
        self.assign_(alpha=alpha)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, celu(x, alpha=self.alpha)


class SELU(Lambda):
    def __init__(self):
        super().__init__(selu)


class GELU(Module):
    approximate: bool

    def __init__(self, approximate: bool = True):
        self.assign_(approximate=approximate)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, gelu(x, approximate=self.approximate)


class GLU(Module):
    axis: int

    def __init__(self, axis: int = -1):
        self.assign_(axis=axis)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        _assert_negative_axis(self.axis)
        return self, glu(x, axis=self.axis)


class SquarePlus(Module):
    b: float

    def __init__(self, b: float = 4):
        self.assign_(b=b)

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self, squareplus(x, b=self.b)


class Mish(Lambda):
    def __init__(self):
        super().__init__(mish)


class Tanh(Lambda):
    def __init__(self):
        super().__init__(tanh)


class Softmax(Module):
    axis: Axis = config()

    def __init__(self, axis: Axis = -1):
        self.assign_(axis=axis)

    def forward(self: Self, x: Array, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        _assert_negative_axis(self.axis)
        return self, softmax(x, axis=self.axis)


class LogSoftmax(Module):
    axis: Axis = config()

    def __init__(self, axis: Axis = -1):
        self.assign_(axis=axis)

    def forward(self: Self, x: Array, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        _assert_negative_axis(self.axis)
        return self, log_softmax(x, axis=self.axis)
