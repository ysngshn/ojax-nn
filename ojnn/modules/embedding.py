from collections.abc import Sequence
from typing_extensions import Self
from jax import Array
from jax.typing import DTypeLike, ArrayLike
from jax.numpy import take as jnp_take
from ..ftypes import parameter, config
from ..utils import KeyArg, maybe_split
from ..struct import new
from .module import Module
from .init import lecun_normal


class Embed(Module):
    num_embeddings: int = config()
    features: int = config()
    dtype: DTypeLike | None = config()
    weight: Array = parameter()

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: DTypeLike | None = None,
    ):
        self.assign_(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
        )

    @property
    def reset_rngkey_count(self) -> int:
        return 1

    def reset(
        self: Self,
        input_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        output_shape = tuple(input_shape) + (self.features,)
        rngkey = maybe_split(rngkey, 1)[0]
        weight = lecun_normal(
            (self.num_embeddings, self.features),
            rngkey=rngkey,
            dtype=self.dtype,
        )
        return new(self, weight=weight), output_shape

    def forward(
        self: Self,
        x: ArrayLike,
        _: KeyArg = None,
    ) -> tuple[Self, Array]:
        return self, jnp_take(self.weight, x, axis=0)
