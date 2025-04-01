# DenseNet architectures
# cf.:
# - Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
# - https://github.com/bamos/densenet.pytorch
from typing import Sequence
from typing_extensions import Self
from ojnn import (
    Module,
    Sequential,
    MapConcat,
    AvgPool2d,
    GlobalAvgPool2d,
    Conv2d,
    Identity,
    ReLU,
    Dense,
    BatchNorm2d,
    maybe_split,
    he_normal,
)
from ojnn.utils import KeyArg


# conv 2d with Kaiming He initialization
class HeInitConv(Conv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        with_bias: bool = True,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            with_bias=with_bias,
        )

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        newself, out_shapes = super().reset(input_shape, rngkey)
        w = newself.weight
        return (
            newself.update(
                weight=he_normal(w.shape, maybe_split(rngkey, 1)[0], w.dtype)
            ),
            out_shapes,
        )


class HeInitDense(Dense):

    def __init__(
        self,
        out_features: int,
        with_bias: bool = True,
    ):
        super().__init__(out_features, with_bias=with_bias)

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        newself, out_shapes = super().reset(input_shape, rngkey)
        w = newself.weight
        return (
            newself.update(
                weight=he_normal(w.shape, maybe_split(rngkey, 1)[0], w.dtype)
            ),
            out_shapes,
        )


def _bn_relu_conv(
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int | str = 0,
) -> list[Module]:
    return [
        BatchNorm2d(),
        ReLU(),
        HeInitConv(
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            with_bias=False,
        ),
    ]


def _make_basic_layer(growth_rate: int) -> Module:
    return MapConcat(
        Sequential(
            *_bn_relu_conv(growth_rate, 3, padding="same"),
        ),
        Identity(),
        axis=-3,
    )


def _make_bottleneck_layer(growth_rate: int) -> Module:
    return MapConcat(
        Sequential(
            *_bn_relu_conv(4 * growth_rate, 1),
            *_bn_relu_conv(growth_rate, 3, padding="same"),
        ),
        Identity(),
        axis=-3,
    )


def _make_dense_block(_layer_fn, growth_rate: int, count: int) -> list[Module]:
    return [_layer_fn(growth_rate) for _ in range(count)]


def _make_transition(out_channels: int) -> list[Module]:
    return [
        *_bn_relu_conv(out_channels, 1),
        AvgPool2d(2),
    ]


def _make_basic_densenet(depth: int, k: int, nclasses: int = 10) -> Module:
    assert (depth - 4) % 3 == 0, "depth for basic densenet has to be 3*n+4"
    layers_per_block = (depth - 4) // 3
    return Sequential(
        HeInitConv(16, 3, padding="same", with_bias=False),
        *_make_dense_block(_make_basic_layer, k, layers_per_block),
        *_make_transition(16 + layers_per_block * k),
        *_make_dense_block(_make_basic_layer, k, layers_per_block),
        *_make_transition(16 + 2 * layers_per_block * k),
        *_make_dense_block(_make_basic_layer, k, layers_per_block),
        *_make_transition(16 + 3 * layers_per_block * k),
        BatchNorm2d(),
        ReLU(),
        GlobalAvgPool2d(),
        HeInitDense(nclasses),
    )


def _make_densenet_bc(depth: int, k: int, nclasses: int = 10) -> Module:
    assert (depth - 4) % 6 == 0, "depth for basic densenet has to be 6*n+4"
    layers_per_block = (depth - 4) // 6
    return Sequential(
        HeInitConv(c := 2 * k, 3, padding="same", with_bias=False),
        *_make_dense_block(_make_bottleneck_layer, k, layers_per_block),
        *_make_transition(c := (c + layers_per_block * k) // 2),
        *_make_dense_block(_make_bottleneck_layer, k, layers_per_block),
        *_make_transition(c := (c + layers_per_block * k) // 2),
        *_make_dense_block(_make_bottleneck_layer, k, layers_per_block),
        *_make_transition((c + layers_per_block * k) // 2),
        BatchNorm2d(),
        ReLU(),
        GlobalAvgPool2d(),
        HeInitDense(nclasses),
    )


def densenet40k12(nclasses: int = 10) -> Module:
    return _make_basic_densenet(40, 12, nclasses)


def densenet100k12(nclasses: int = 10) -> Module:
    return _make_basic_densenet(100, 12, nclasses)


def densenet100k24(nclasses: int = 10) -> Module:
    return _make_basic_densenet(100, 24, nclasses)


def densenetbc100k12(nclasses: int = 10) -> Module:
    return _make_densenet_bc(100, 12, nclasses)


def densenetbc250k24(nclasses: int = 10) -> Module:
    return _make_densenet_bc(250, 24, nclasses)


def densenetbc190k40(nclasses: int = 10) -> Module:
    return _make_densenet_bc(190, 40, nclasses)
