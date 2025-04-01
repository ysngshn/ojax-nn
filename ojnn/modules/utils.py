from typing import TypeVar, Union, TypeAlias, cast
from collections.abc import Sequence
from math import prod
from ..utils import Axis, KeyArg

# ==================== #
# Conv / Pooling stuff #
# ==================== #


_T = TypeVar("_T")
MaybeShape: TypeAlias = Union[_T, Sequence[_T]]
IntShape: TypeAlias = MaybeShape[int]
PaddingLike: TypeAlias = str | IntShape | Sequence[tuple[int, int]]


def _assert_negative_axis(axis: Axis) -> None:
    msg = "axis should be negative"
    if axis is None:  # None
        pass
    elif hasattr(axis, "__len__"):  # tuple[int, ...]
        axis = cast(tuple[int, ...], axis)
        assert all(a < 0 for a in axis), msg
    else:  # int
        assert axis < 0, msg


def _assert_no_key(key: KeyArg) -> None:
    assert not key, "random key should not be supplied"


def _as_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return x, x
    else:
        return x


def as_tuple(shape: IntShape, dim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * dim
    else:
        if len(shape) != dim:
            raise ValueError(
                f"wrong dimension: expected {dim}, received {len(shape)}"
            )
        else:
            return tuple(shape)


def get_dim_and_window_shape(
    dim: int | None, kernel: IntShape
) -> tuple[int, tuple[int, ...]]:
    if dim is None:
        if isinstance(kernel, int):
            raise ValueError("Unable to infer dimension of sliding window")
        else:
            dim, kernel = len(kernel), tuple(kernel)
    else:
        if isinstance(kernel, int):
            kernel = (kernel,) * dim
        else:
            kernel = tuple(kernel)
            if len(kernel) != dim:
                raise ValueError(
                    f"inconsistent dimension: window has dimension "
                    f"{len(kernel)}, specified {dim}"
                )
    if prod(kernel) <= 0:
        raise ValueError(
            f"window should have strictly positive size, received window "
            f"{kernel} if size {prod(kernel)}."
        )
    return dim, kernel


def get_padding(
    padding: PaddingLike, dim: int
) -> str | tuple[tuple[int, int], ...]:
    if isinstance(padding, str):
        return padding.upper()
    else:
        if isinstance(padding, int):
            return ((padding, padding),) * dim
        else:
            return tuple(_as_pair(p) for p in padding)


def effective_kernel_size(
    kernel_size: tuple[int, ...], dilation: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple((k - 1) * d + 1 for k, d in zip(kernel_size, dilation))


def conv_output_shape(
    input_shape: tuple[int, ...],
    out_channels: int,
    dimension: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    input_spatials = input_shape[-dimension:]

    if padding in ("SAME", "SAME_LOWER"):
        return (*input_shape[: -dimension - 1], out_channels) + tuple(
            (i - 1) // s + 1 for i, s in zip(input_spatials, stride)
        )
    elif padding == "VALID":
        padding = ((0, 0),) * dimension
    elif isinstance(padding, str):
        raise NotImplementedError(f"unknown padding mode {padding}")
    else:
        pass

    output_spatials = tuple(
        (i + pl + pr - k) // s + 1
        for i, k, s, (pl, pr) in zip(
            input_spatials, kernel_size, stride, padding
        )
    )

    return (*input_shape[: -dimension - 1], out_channels) + output_spatials
