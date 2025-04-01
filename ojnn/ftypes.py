from typing import TypeVar
from collections.abc import Callable
from dataclasses import MISSING, field, Field
from jax.numpy import isscalar
from jax.typing import ArrayLike
import ojax


def make_default_array() -> float:
    return float("nan")


# custom subclasses of dataclasses.Field


class StructField(ojax.OTreeField):
    pass


class Config(StructField, ojax.Aux):
    pass


class _StateBase(StructField, ojax.Child):
    pass


class State(_StateBase):
    """A generic numeric PyTree state"""

    pass


class Const(_StateBase):
    """A constant numeric pytree node that shouldn't be updated"""

    pass


class _InternalState(_StateBase):
    pass


class Child(_StateBase):
    """A child container"""

    pass


class Parameter(_StateBase):
    """A numeric array that can be optimized by optimizer"""

    pass


class Buffer(_StateBase):
    """A numeric array that is not a parameter."""

    pass


class Schedulable(_StateBase):
    """A scalar state that can be scheduled."""

    pass


class External(StructField, ojax.Aux):
    """A field that holds "impure" JAX-incompatible external content"""

    pass


# corresponding custom variants of dataclasses.field


def config(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Config(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def state(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return State(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def const(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Const(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def _internal_state(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=False,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return _InternalState(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def child(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Child(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def parameter(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=False,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    if default is MISSING and default_factory is MISSING:
        default = make_default_array()

    return Parameter(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def buffer(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=False,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    if default is MISSING and default_factory is MISSING:
        default = make_default_array()

    return Buffer(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def schedulable(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Schedulable(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def external(
    *,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=MISSING,
):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return External(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


Field_T = TypeVar("Field_T", bound=Field)
_field_fn_map: dict[type[Field], Callable[..., Field]] = {
    Field: field,
    Config: config,
    State: state,
    Const: const,
    _InternalState: _internal_state,
    Child: child,
    Parameter: parameter,
    Buffer: buffer,
    Schedulable: schedulable,
    External: external,
}


# validity checks for field update


def _check_parameter(f, v):
    if (not isinstance(v, ArrayLike)) and (v is not None):
        raise ValueError(
            f"{f.name} should be a JAX Array, received {v} of type {type(v)}."
        )


def _check_buffer(f, v):
    if (not isinstance(v, ArrayLike)) and (v is not None):
        raise ValueError(
            f"{f.name} should be a JAX Array, received {v} of type {type(v)}."
        )


def _check_schedulable(f, v):
    if (not isscalar(v)) and (v is not None):
        raise ValueError(f"{f.name} should have scalar value")
