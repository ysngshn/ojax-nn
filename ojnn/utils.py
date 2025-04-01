from typing import TypeAlias
from jax import Array
from jax.random import split as jrsplit
from jax.dtypes import issubdtype, prng_key


KeyArray: TypeAlias = Array
KeyArg: TypeAlias = KeyArray | None
Axis: TypeAlias = int | tuple[int, ...] | None


def maybe_split(key: KeyArg, size: int) -> KeyArray:
    if key is None:
        raise ValueError("Cannot split without a key")
    if issubdtype(key.dtype, prng_key):  # new type
        keyshape = key.shape
    else:
        keyshape = key.shape[:-1]
    if keyshape == ():
        return jrsplit(key, size)
    else:
        if keyshape[0] != size:
            raise ValueError(f"expect {size} keys, found {keyshape[0]}")
        else:
            return key
