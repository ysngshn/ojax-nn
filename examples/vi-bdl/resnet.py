# pre-activation variant of the ResNet models
# cf.:
# - He et al. "Identity mappings in deep residual networks" ECCV 2016
# - github.com/KaimingHe/resnet-1k-layers
# - catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
from typing import Sequence
from typing_extensions import Self
from ojnn import (
    Module,
    Sequential,
    MapReduce,
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


def _bn_relu() -> list[Module]:
    return [BatchNorm2d(), ReLU()]


# residual block with trainable scalar for rezero init
def make_block(
    channels: int,
    bottleneck_scale: int | None = None,
    stride: int = 1,
    conv_shortcut: bool = False,
) -> Module:
    if bottleneck_scale is None:
        layers = [
            *([] if conv_shortcut else _bn_relu()),
            HeInitConv(
                channels, 3, stride=stride, padding="same", with_bias=False
            ),
            *_bn_relu(),
            HeInitConv(channels, 3, padding="same", with_bias=False),
        ]
    else:
        bottleneck_channels = channels // bottleneck_scale
        layers = [
            *([] if conv_shortcut else _bn_relu()),
            HeInitConv(bottleneck_channels, 1, with_bias=False),
            *_bn_relu(),
            HeInitConv(
                bottleneck_channels,
                3,
                stride=stride,
                padding="same",
                with_bias=False,
            ),
            *_bn_relu(),
            HeInitConv(channels, 1, with_bias=False),
        ]
    residual = Sequential(*layers)
    if conv_shortcut:
        return Sequential(
            *_bn_relu(),
            MapReduce(
                HeInitConv(channels, 1, stride=stride, with_bias=False),
                residual,
            ),
        )
    else:
        return MapReduce(Identity(), residual)


def make_trunk(
    blocks: int,
    channels: int,
    stride: int = 1,
    bottleneck_scale: int | None = None,
) -> list[Module]:
    resblocks = [
        make_block(channels, bottleneck_scale, stride, conv_shortcut=True),
    ]
    for _ in range(1, blocks):
        resblocks.append(make_block(channels, bottleneck_scale))
    return resblocks


_resnet_stages = (16, 64, 128, 256)
_bottleneck_scale = 4


def make_basic_resnet(depth: int, nclass: int = 10) -> Module:
    assert (depth - 2) % 6 == 0, "depth for basic resnet has to be 6*n+2"
    blocks = (depth - 2) // 6
    assert blocks > 0, "basic resnet requires at least depth 8"
    return Sequential(
        HeInitConv(_resnet_stages[0], 3, padding="same"),
        *make_trunk(blocks, _resnet_stages[1], 1, None),
        *make_trunk(blocks, _resnet_stages[2], 2, None),
        *make_trunk(blocks, _resnet_stages[3], 2, None),
        *_bn_relu(),
        GlobalAvgPool2d(),
        HeInitDense(nclass),
    )


def make_bottleneck_resnet(depth: int, nclass: int = 10) -> Module:
    assert (depth - 2) % 9 == 0, "depth for bottleneck resnet has to be 9*n+2"
    blocks = (depth - 2) // 9
    assert blocks > 0, "bottleneck resnet requires at least depth 11"

    return Sequential(
        HeInitConv(_resnet_stages[0], 3, padding="same"),
        *make_trunk(blocks, _resnet_stages[1], 1, _bottleneck_scale),
        *make_trunk(blocks, _resnet_stages[2], 2, _bottleneck_scale),
        *make_trunk(blocks, _resnet_stages[3], 2, _bottleneck_scale),
        *_bn_relu(),
        GlobalAvgPool2d(),
        HeInitDense(nclass),
    )


def resnet20(nclass: int = 10) -> Module:
    return make_basic_resnet(20, nclass)


def resnet110(nclass: int = 10) -> Module:
    return make_bottleneck_resnet(110, nclass)


def resnet164(nclass: int = 10) -> Module:
    return make_bottleneck_resnet(164, nclass)


def resnet1001(nclass: int = 10) -> Module:
    return make_bottleneck_resnet(1001, nclass)
