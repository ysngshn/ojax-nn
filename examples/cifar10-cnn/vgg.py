# VGG architectures adapted for CIFAR-10
# cf.:
# - Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale
# Image Recognition", ICLR 2015
#
# adjustments for CIFAR-10:
# - reduce all channel size by 4
# - remove the max pooling at stages 1. 2.and 4
#
# we also uses the OJNN default identity / orthogonal weight init
from ojnn import (
    Module,
    Sequential,
    MaxPool2d,
    Flatten2d,
    Conv2d,
    ReLU,
    Dense,
    Dropout,
)


def _conv_relu(out_channels: int) -> list[Module]:
    return [Conv2d(out_channels, 3, padding="same"), ReLU()]


def _dense_relu_dropout(out_channels: int) -> list[Module]:
    return [Dense(out_channels), ReLU(), Dropout(p=0.5)]


def _make_stage(out_channels: int, nlayer: int, maxpool: bool) -> list[Module]:
    layers = []
    for _ in range(nlayer):
        layers += _conv_relu(out_channels)
    if maxpool:
        layers.append(MaxPool2d(2))
    return layers


def _make_classifier_head(nclasses: int = 10) -> list[Module]:
    return [
        Flatten2d(),
        *_dense_relu_dropout(1024),
        *_dense_relu_dropout(1024),
        Dense(nclasses),
    ]


def vgg11(nclasses: int = 10) -> Module:
    return Sequential(
        *_make_stage(16, 1, False),
        *_make_stage(32, 1, False),
        *_make_stage(64, 2, True),
        *_make_stage(128, 2, False),
        *_make_stage(128, 2, True),
        *_make_classifier_head(nclasses),
    )


def vgg13(nclasses: int = 10) -> Module:
    return Sequential(
        *_make_stage(16, 2, False),
        *_make_stage(32, 2, False),
        *_make_stage(64, 2, True),
        *_make_stage(128, 2, False),
        *_make_stage(128, 2, True),
        *_make_classifier_head(nclasses),
    )


def vgg16(nclasses: int = 10) -> Module:
    return Sequential(
        *_make_stage(16, 2, False),
        *_make_stage(32, 2, False),
        *_make_stage(64, 3, True),
        *_make_stage(128, 3, False),
        *_make_stage(128, 3, True),
        *_make_classifier_head(nclasses),
    )


def vgg19(nclasses: int = 10) -> Module:
    return Sequential(
        *_make_stage(16, 2, False),
        *_make_stage(32, 2, False),
        *_make_stage(64, 4, True),
        *_make_stage(128, 4, False),
        *_make_stage(128, 4, True),
        *_make_classifier_head(nclasses),
    )
