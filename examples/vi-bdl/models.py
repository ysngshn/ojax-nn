# model for CIFAR-10 / SVHN
# cf.:
# - Yuesong et al., "Variational Learning is Effective for Large Deep
# Networks", ICML 2024
# - https://github.com/team-approx-bayes/ivon-experiments
from typing import Sequence
from typing_extensions import Self
import math
from jax import Array
from jax.numpy import zeros, ones, square, mean, maximum
from jax.random import uniform
from jax.lax import rsqrt
from jax.typing import ArrayLike
from ojnn import (
    config,
    parameter,
    Module,
    Sequential,
    MapReduce,
    GlobalAvgPool2d,
    Conv2d,
    Identity,
    Dense,
    maybe_split,
)
from ojnn.utils import KeyArg


__all__ = ("resnet20",)


# filter response norm with TLU activation
# cf.:
# - Singh and Krishnan, "Filter Response Normalization Layer: Eliminating Batch
# Dependence in the Training of Deep Neural Networks", CVPR 2020
class FRNormTLU(Module):
    channel_dim: int = config()
    eps: float = config()
    weight: Array = parameter()
    bias: Array = parameter()
    threshold: Array = parameter()

    def __init__(self, channel_dim: int = -3, eps: float = 1e-6):
        assert channel_dim < 0
        self.assign_(channel_dim=channel_dim, eps=eps)

    def reset(self, input_shape, _=None):
        channels = input_shape[self.channel_dim]
        shape = [channels] + [1 for _ in range(self.channel_dim + 1, 0)]
        weight = ones(shape)
        bias = zeros(shape)
        threshold = zeros(shape)
        return self.update(
            weight=weight, bias=bias, threshold=threshold
        ), tuple(input_shape)

    def forward(self, x: ArrayLike, _=None):
        nu2 = mean(
            square(x),
            axis=tuple(range(self.channel_dim + 1, 0)),
            keepdims=True,
        )
        y = self.weight * x * rsqrt(nu2 + self.eps) + self.bias
        return self, maximum(y, self.threshold)


# conv 2d with default PyTorch init
class TorchInitConv(Conv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "same",
        with_bias: bool = True,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            with_bias=with_bias,
        )

    @property
    def reset_rngkey_count(self) -> int:
        if self.bias is None:
            return 1
        else:
            return 2

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        if self.bias is None:
            rngkey = maybe_split(rngkey, 1)[0]
            key2 = None
        else:
            rngkey, key2 = maybe_split(rngkey, 2)
        newself, out_shapes = super().reset(input_shape, rngkey)
        w, b = newself.weight, newself.bias
        fan_in = math.prod(w.shape[1:])
        bound = 1.0 / math.sqrt(fan_in)
        weight = uniform(rngkey, w.shape, w.dtype, minval=-bound, maxval=bound)
        if b is None:
            bias = None
        else:
            bias = uniform(key2, b.shape, b.dtype, minval=-bound, maxval=bound)
        return newself.update(weight=weight, bias=bias), out_shapes


# Dense layer with default PyTorch init
class LecunInitDense(Dense):

    def __init__(
        self,
        out_features: int,
        with_bias: bool = True,
    ):
        super().__init__(out_features, with_bias=with_bias)

    @property
    def reset_rngkey_count(self) -> int:
        if self.bias is None:
            return 1
        else:
            return 2

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        if self.bias is None:
            rngkey = maybe_split(rngkey, 1)[0]
            key2 = None
        else:
            rngkey, key2 = maybe_split(rngkey, 2)
        newself, out_shapes = super().reset(input_shape, rngkey)
        w, b = newself.weight, newself.bias
        fan_in = w.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        weight = uniform(rngkey, w.shape, w.dtype, minval=-bound, maxval=bound)
        if b is None:
            bias = None
        else:
            bias = uniform(key2, b.shape, b.dtype, minval=-bound, maxval=bound)
        return newself.update(weight=weight, bias=bias), out_shapes


def _conv2d_frn_tlu(
    channels: int,
    kernel_size: int,
    stride: int = 1,
) -> list[Module]:
    return [
        TorchInitConv(channels, kernel_size, stride, with_bias=True),
        FRNormTLU(),
    ]


def _make_basic_resblock(
    channels: int,
    stride: int = 1,
    conv_shortcut: bool = False,
) -> list[Module]:
    layers = [
        *_conv2d_frn_tlu(channels, 3, stride),
        *_conv2d_frn_tlu(channels, 3),
    ]
    residual = Sequential(*layers)
    if conv_shortcut:
        return [
            MapReduce(
                Sequential(*_conv2d_frn_tlu(channels, 1, stride)), residual
            ),
        ]
    else:
        return [MapReduce(Identity(), residual)]


def _make_basic_restrunk(
    channels: int,
    nblocks: int,
    stride: int,
) -> list[Module]:
    layers = _make_basic_resblock(channels, stride, stride != 1)
    for _ in range(nblocks - 1):
        layers += _make_basic_resblock(channels)
    return layers


# resnet20_frn model from https://github.com/team-approx-bayes/ivon-experiments
def resnet20(nclasses: int = 10) -> Module:
    c_sizes = [16, 32, 64]
    return Sequential(
        *_conv2d_frn_tlu(c_sizes[0], 3),
        *_make_basic_restrunk(c_sizes[0], 3, 1),
        *_make_basic_restrunk(c_sizes[1], 3, 2),
        *_make_basic_restrunk(c_sizes[2], 3, 2),
        GlobalAvgPool2d(),
        LecunInitDense(nclasses),
    )
