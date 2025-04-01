from collections.abc import Callable
from functools import wraps
from jax import ShapeDtypeStruct
import jax.tree as jtree
from jax.sharding import SingleDeviceSharding
from jax.lax import stop_gradient
from jax.experimental import io_callback
from jax.debug import callback as debug_callback


def get_positive_index(index: int, total: int) -> int:
    if index < -total or index >= total:
        raise ValueError(
            f"invalid index {index} outside of range [{-total} .. {total - 1}]"
        )
    elif index < 0:
        index += total
    return index


def get_shape_dtype(pytree):
    return jtree.map(lambda x: ShapeDtypeStruct(x.shape, x.dtype), pytree)


def host_callback(
    fn: Callable,
    debug_impl: bool = False,
    result_shape_dtypes=None,
    sharding: SingleDeviceSharding | None = None,
    ordered: bool = True,
):

    @wraps(fn)
    def jax_callback(*args, **kwargs):
        if debug_impl and sharding is not None:
            raise ValueError('debug host callback has no "sharding".')
        if result_shape_dtypes is None:  # no output, grad-free setup
            if debug_impl:
                return debug_callback(fn, *args, ordered=ordered, **kwargs)
            else:
                args, kwargs = stop_gradient((args, kwargs))
                return stop_gradient(
                    io_callback(
                        fn,
                        None,
                        *args,
                        sharding=sharding,
                        ordered=ordered,
                        **kwargs,
                    )
                )
        else:  # produces output, requires io_callback
            if debug_impl:
                raise ValueError(
                    'debug host callback has no "result_shape_dtypes"'
                )
            else:
                return io_callback(
                    fn,
                    result_shape_dtypes,
                    *args,
                    sharding=sharding,
                    ordered=ordered,
                    **kwargs,
                )

    return jax_callback
