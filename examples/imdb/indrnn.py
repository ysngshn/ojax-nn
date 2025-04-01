from collections.abc import Callable
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import Array, ShapeDtypeStruct as ArraySDS
from jax.lax import fori_loop


def _indrnn_fwd_kernel(
    x_ref,  # [T]
    whh_ref,  # scalar
    h0_ref,  # scalar
    # outputs
    o_ref,  # [T]
    pa_ref,  # [T]
    *,
    activation,
    unroll,
):
    seqlen = x_ref.size
    whh = whh_ref[...]
    h0 = h0_ref[...]

    def _step(i, h_prev):
        preact_i = x_ref[i] + whh * h_prev
        h_next = activation(preact_i)
        pa_ref[i] = preact_i
        o_ref[i] = h_next
        return h_next

    _ = fori_loop(0, seqlen, _step, h0, unroll=unroll)


def _indrnn_fwd(
    activation: Callable[[Array], Array],
    channel_dim: int,
    unroll: bool | int,
    interpret: bool,
    debug: bool,
    x: Array,
    whh: Array,
    h0: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:

    def _index_fn_seq(*indices):
        i = list(indices)
        i.insert(0, 0)
        return tuple(i)

    def _index_fn_whh(*indices):
        i = list(indices)
        i.insert(0, 0)
        return (i[channel_dim],)

    grid = tuple(x.shape[1:])

    h0 = jnp.zeros(grid, dtype=x.dtype) if h0 is None else h0
    seq_block_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    seq_spec = pl.BlockSpec(seq_block_shape, _index_fn_seq)
    whh_spec = pl.BlockSpec((1,), _index_fn_whh)
    h_spec = pl.BlockSpec((1,) * h0.ndim, lambda *idx: idx)

    y, preact = pl.pallas_call(
        partial(_indrnn_fwd_kernel, activation=activation, unroll=unroll),
        out_shape=[ArraySDS(x.shape, x.dtype), ArraySDS(x.shape, x.dtype)],
        grid=grid,
        in_specs=[seq_spec, whh_spec, h_spec],
        out_specs=[seq_spec, seq_spec],
        debug=debug,
        interpret=interpret,
    )(x, whh, h0)
    return y, (whh, preact, h0)


def _indrnn_bwd_kernel(
    dy_ref,  # [T]
    w_ref,
    pa_ref,  # [T]
    h0_ref,
    # outputs
    dx_ref,  # [T]
    dw_ref,
    dh0_ref,
    *,
    activation,
    unroll,
):
    seqlen = dy_ref.shape[0]
    w = w_ref[...]
    h0 = h0_ref[...]
    dh = jnp.zeros(dy_ref.shape[1:], dy_ref.dtype)
    dw = jnp.zeros(w.shape, dy_ref.dtype)

    def _step(i, carry):
        dw, dh = carry
        da = dy_ref[seqlen-i] + dh
        f_vjp = jax.vjp(activation, pa_ref[seqlen-i])[1]
        dpa = f_vjp(da)[0]
        h_prev = activation(pa_ref[seqlen-i-1])
        dw += jnp.sum(h_prev * dpa)
        dx_ref[seqlen-i] = dpa
        return dw, dpa * w

    dw, dh = fori_loop(
        1, seqlen, _step, (dw, dh), unroll=unroll
    )

    da = dy_ref[0] + dh
    f_vjp = jax.vjp(activation, pa_ref[0])[1]
    dpa = f_vjp(da)[0]
    dw_ref[...] = dw + jnp.sum(h0 * dpa)
    dx_ref[0] = dpa
    dh0_ref[...] = dpa * w


def _indrnn_bwd(
    activation: Callable[[Array], Array],
    channel_dim: int,
    unroll: bool | int,
    interpret: bool,
    debug: bool,
    res: tuple[Array, Array, Array],
    dy: Array,
) -> tuple[Array, Array, Array]:
    whh, preact, h0 = res

    def _index_fn_seq(*indices):
        i = list(indices)
        i.insert(0, 0)
        return tuple(i)

    def _index_fn_whh(*indices):
        i = list(indices)
        i.insert(0, 0)
        return (i[channel_dim],)

    grid = tuple(preact.shape[1:])
    seq_block_shape = (dy.shape[0],) + (1,) * (dy.ndim - 1)

    seq_spec = pl.BlockSpec(seq_block_shape, _index_fn_seq)
    whh_spec = pl.BlockSpec((1,), _index_fn_whh)
    h0_spec = pl.BlockSpec((1,) * len(grid), lambda *idx: idx)

    seq_sds = ArraySDS(dy.shape, dy.dtype)
    whh_sds = ArraySDS(whh.shape, dy.dtype)
    h0_sds = ArraySDS(grid, dy.dtype)

    dx, dw, dh0 = pl.pallas_call(
        partial(_indrnn_bwd_kernel, activation=activation, unroll=unroll),
        out_shape=[seq_sds, whh_sds, h0_sds],
        grid=grid,
        in_specs=[seq_spec, whh_spec, seq_spec, h0_spec],
        out_specs=[seq_spec, whh_spec, h0_spec],
        debug=debug,
        interpret=interpret,
    )(dy, whh, preact, h0)
    return dx, dw, dh0


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _indrnn(
    activation: Callable[[Array], Array],
    channel_dim: int,
    unroll: bool | int,
    interpret: bool,
    debug: bool,
    x: Array,
    whh: Array,
    h0: Array,
) -> Array:
    return _indrnn_fwd(
        activation, channel_dim, unroll, interpret, debug, x, whh, h0
    )[0]


_indrnn.defvjp(_indrnn_fwd, _indrnn_bwd)


def indrnn(
    x: Array,
    whh: Array,
    h0: Array | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    seq_dim: int = -2,
    channel_dim: int = -1,
    unroll: bool | int = False,
    interpret: bool = False,
    debug: bool = False,
) -> Array:
    assert whh.ndim == 1
    assert x.shape[channel_dim] == whh.size

    x = jnp.moveaxis(x, seq_dim, 0)
    x = _indrnn(activation, channel_dim, unroll, interpret, debug, x, whh, h0)
    return jnp.moveaxis(x, seq_dim, 0)
    
