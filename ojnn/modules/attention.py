from collections.abc import Sequence
from typing_extensions import Self
import abc
from math import sqrt
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from ..utils import KeyArg, maybe_split
from ..ftypes import config, child
from ..struct import new
from .module import Module
from .linear import Dense
from .utils import _assert_no_key
from .misc import dot_product_attention
from .init import lecun_uniform


def _to_attn_shape(x: Array, num_heads: int) -> tuple[Array, tuple[int, ...]]:
    t, c = x.shape[-2:]
    batch_shape = x.shape[:-2]
    return x.reshape(-1, t, num_heads, c // num_heads), batch_shape


def _from_attn_shape(x: Array, batch_shape: tuple[int, ...]) -> Array:
    t, h, c = x.shape[-3:]
    return x.reshape(*batch_shape, t, h * c)


def _check_div_by_head(val: int, val_name: str, num_heads: int):
    assert val % num_heads == 0, f"{val_name} should be divisible by num_heads"


class _MultiHeadAttentionBase(Module, metaclass=abc.ABCMeta):
    # configs
    out_features: int = config()
    num_heads: int = config()
    embed_size: int = config()
    is_causal: bool = config()
    with_bias: bool = config()
    dtype: DTypeLike | None = config()
    # children
    proj_k: Dense = child()
    proj_q: Dense = child()
    proj_v: Dense = child()
    proj_o: Dense = child()

    def __init__(
        self,
        out_features: int,
        num_heads: int,
        embed_size: int | None = None,
        is_causal: bool = False,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        _check_div_by_head(out_features, "out_features", num_heads)
        embed_size = out_features if embed_size is None else embed_size
        _check_div_by_head(embed_size, "embed_size", num_heads)
        self.assign_(
            out_features=out_features,
            num_heads=num_heads,
            embed_size=embed_size,
            is_causal=is_causal,
            with_bias=with_bias,
            dtype=dtype,
            proj_k=Dense(embed_size, with_bias=with_bias),
            proj_q=Dense(embed_size, with_bias=with_bias),
            proj_v=Dense(embed_size, with_bias=with_bias),
            proj_o=Dense(out_features, with_bias=with_bias),
        )

    @property
    def reset_rngkey_count(self) -> int:
        return 6

    def _reset_projections(
        self: Self,
        q_shape: Sequence[int],
        k_shape: Sequence[int],
        v_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        """we use the heuristic init

        Wk = Wq = lecun_uniform, Wv = -Wo.T = lecun_uniform

        which creates prominent diagonal entries that approximate

        Wk @ Wq.T = c1 Id; Wv @ Wo = -c2 Id

        found to be desirable by Asher Trockman and J. Zico Kolter,

        cf.: Mimetic initialization of self-attention layers. ICML 2023
        """
        qkey, kkey, vkey, okey, key_qk, key_vo = maybe_split(rngkey, 6)
        # default inits
        proj_q, attn_shape = self.proj_q.reset(q_shape, qkey)
        proj_k, _ = self.proj_k.reset(k_shape, kkey)
        proj_v, _ = self.proj_v.reset(v_shape, vkey)
        proj_o, out_shape = self.proj_o.reset(attn_shape, okey)
        embed = self.embed_size
        q_in = proj_q.weight.shape[1]
        k_in = proj_k.weight.shape[1]
        v_in = proj_v.weight.shape[1]
        o_out = proj_o.weight.shape[0]
        max_qk = max(q_in, k_in)
        max_vo = max(v_in, o_out)
        w_qk = lecun_uniform((embed, max_qk), key_qk, dtype=self.dtype)
        w_vo = lecun_uniform((embed, max_vo), key_vo, dtype=self.dtype)
        proj_q = new(proj_q, weight=sqrt(max_qk / q_in) * w_qk[:, :q_in])
        proj_k = new(proj_k, weight=sqrt(max_qk / k_in) * w_qk[:, :k_in])
        proj_v = new(proj_v, weight=sqrt(max_vo / v_in) * w_vo[:, :v_in])
        proj_o = new(proj_o, weight=sqrt(max_vo / o_out) * w_vo[:, :o_out].T)

        return (
            new(
                self,
                proj_q=proj_q,
                proj_k=proj_k,
                proj_v=proj_v,
                proj_o=proj_o,
            ),
            out_shape,
        )

    def _compute_attention(
        self: Self, query: ArrayLike, key: ArrayLike, value: ArrayLike
    ) -> tuple[Self, Array]:
        proj_q, q_embed = self.proj_q.forward(query)
        proj_k, k_embed = self.proj_k.forward(key)
        proj_v, v_embed = self.proj_v.forward(value)
        q, q_bshape = _to_attn_shape(q_embed, self.num_heads)
        k = _to_attn_shape(k_embed, self.num_heads)[0]
        v = _to_attn_shape(v_embed, self.num_heads)[0]
        attn = _from_attn_shape(
            dot_product_attention(q, k, v, is_causal=self.is_causal),
            q_bshape,
        )
        proj_o, out = self.proj_o.forward(attn)
        return (
            self.update(
                proj_q=proj_q, proj_k=proj_k, proj_v=proj_v, proj_o=proj_o
            ),
            out,
        )


class MultiHeadAttention(_MultiHeadAttentionBase):
    def reset(
        self: Self,
        input_shape: tuple[Sequence[int], Sequence[int], Sequence[int]],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        q_shape, k_shape, v_shape = input_shape
        return self._reset_projections(q_shape, k_shape, v_shape, rngkey)

    # all of shape (...,T, C)
    def forward(
        self: Self, kqv: tuple[ArrayLike, ArrayLike, ArrayLike], _=None
    ) -> tuple[Self, Array]:
        _assert_no_key(_)
        k_in, q_in, v_in = kqv
        return self._compute_attention(k_in, q_in, v_in)


class MultiHeadSelfAttention(_MultiHeadAttentionBase):
    def reset(
        self: Self,
        input_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        return self._reset_projections(
            input_shape, input_shape, input_shape, rngkey
        )

    # shape (...,T, C)
    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        return self._compute_attention(x, x, x)
