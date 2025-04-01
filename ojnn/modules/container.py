"""container structures that manage model states for neural networks"""

from typing import (
    TypeVar,
    TypeAlias,
    ClassVar,
    Literal,
    overload,
    Generic,
    cast,
)
from collections.abc import (
    Callable,
    Sequence,
    Mapping,
    Iterator,
)
from typing_extensions import Self
from dataclasses import Field
import re
from jax import Array
from ..ftypes import (
    StructField,
    Child,
    Config,
    Buffer,
    Parameter,
    Const,
    _internal_state,
    config,
)
from ..struct import Struct, new


# dataclasses.fields variant for Container which can filter FieldType


_T = TypeVar("_T")
_CT = TypeVar("_CT")
RecursiveState: TypeAlias = tuple[_T, dict[str, "RecursiveState[_T]"]]
Parametric_T = TypeVar("Parametric_T", bound="Parametric")


def _check_child_container(f, v):
    if (
        isinstance(f, Child)
        and (not isinstance(v, Parametric))
        and (v is not None)
    ):
        raise ValueError(
            f"Invalid container type: {f.name} has type {type(v)}."
        )


class Parametric(Struct, Generic[Parametric_T]):
    admissible_field_types: ClassVar[tuple[type[StructField], ...]] = (
        Config,
        Parameter,
        Buffer,
        Child,
        Const,
    )
    _trainable_parameters: dict[str, bool] = config(init=False)

    def __new__(cls):
        self = super().__new__(cls)
        # automatically initialize the _trainable_parameters internal state
        object.__setattr__(
            self,
            "_trainable_parameters",
            {f.name: True for f in cls.fields(Parameter)},
        )
        return self

    @classmethod
    def infer_field_type(cls, f: Field) -> type[StructField]:
        try:
            ftype = cast(type, f.type)
            if issubclass(ftype, Parametric):
                return Child
            elif issubclass(ftype, Array):
                return Buffer
            else:
                return Config
        except TypeError:  # issubclass fails if f.type is not a proper class
            return Config

    # overwrites assign and update to ensure appropriate fields

    def _check_trainable(self, f: Field) -> None:
        if isinstance(f, Parameter) and (
            not self._trainable_parameters[f.name]
        ):
            raise ValueError(f"cannot modify non trainable parameter {f.name}")

    def update_guard(self, **kwargs) -> None:
        super().update_guard(**kwargs)
        for k, v in kwargs.items():
            f = self.__dataclass_fields__[k]
            _check_child_container(f, v)
            self._check_trainable(f)

    # methods to retrieve the lists of parameters / buffers / children

    def get_parameters(self) -> dict[str, Array]:
        return {f.name: getattr(self, f.name) for f in self.fields(Parameter)}

    def get_buffers(self) -> dict[str, Array]:
        return {f.name: getattr(self, f.name) for f in self.fields(Buffer)}

    def get_children(self: Self) -> dict[str, Parametric_T]:
        return {f.name: getattr(self, f.name) for f in self.fields(Child)}

    # generic function to recursively update current module and all submodules

    def recursive_apply(
        self: Self,
        fn: Callable[[Self, _T | None], tuple[Self, _CT]],
        recursive_args: RecursiveState[_T] | None = None,
    ) -> tuple[Self, RecursiveState[_CT]]:
        """recursively update this module and all its submodules"""
        newself, output = fn(
            self,
            None if recursive_args is None else recursive_args[0],
        )

        if len(newself.get_children()) == 0:
            return newself, (output, {})

        def _apply(k, a, m):
            newm, o = m.recursive_apply(fn, a)
            return (k, newm), (k, o)

        if recursive_args is not None:
            children_args = recursive_args[1]
            assert sorted(children_args.keys()) == sorted(
                [k for k, _ in newself.get_children().items()]
            )
            newch, choutput = zip(
                *[
                    _apply(k, children_args[k], m)
                    for k, m in newself.get_children().items()
                ]
            )
        else:
            newch, choutput = zip(
                *[
                    _apply(nc, None, mc)
                    for nc, mc in newself.get_children().items()
                ]
            )
        return new(newself, **dict(newch)), (output, dict(choutput))

    # methods to handle trainable parameters for NN training

    @overload
    def trainable_parameters(
        self: Self, recursive: Literal[True]
    ) -> RecursiveState[dict[str, Array]]: ...

    @overload
    def trainable_parameters(
        self: Self, recursive: Literal[False]
    ) -> dict[str, Array]: ...

    @overload
    def trainable_parameters(
        self: Self,
    ) -> RecursiveState[dict[str, Array]]: ...

    def trainable_parameters(self, recursive=True):
        def _filter_trainable(
            m: Parametric, params: dict[str, Array]
        ) -> dict[str, Array]:
            return {n: p for n, p in params.items() if m.is_trainable(n)}

        if recursive:
            return self.recursive_apply(
                lambda m, _: (
                    m,
                    m.trainable_parameters(recursive=False),
                ),
                None,
            )[1]
        else:
            return _filter_trainable(self, self.get_parameters())

    def is_trainable(self, param_name: str) -> bool:
        return self._trainable_parameters[param_name]

    def config_trainable(
        self: Self, trainable: bool | None = None, /, **kwargs: bool
    ) -> Self:
        """configure the trainability of current module parameters"""
        assert set(kwargs).issubset(set(self._trainable_parameters.keys()))
        assert set(kwargs.values()).issubset({False, True})
        if trainable is not None:  # set for all parameter in this module
            updated_trainable = {
                n: trainable for n in self._trainable_parameters.keys()
            }
        else:
            if not kwargs:
                return self
            else:
                updated_trainable = dict(self._trainable_parameters)
        updated_trainable.update(**kwargs)
        newself = new(self, _trainable_parameters=updated_trainable)
        return newself

    # methods to handle the array states in general

    def states(self) -> RecursiveState:
        def _get_state(m, _):
            return m, {**self.get_buffers(), **self.get_parameters()}

        return self.recursive_apply(_get_state, None)[1]

    def load_states(self: Self, states: RecursiveState) -> Self:
        """recursively attach detached trainable parameters"""

        def _update_state(model, state):
            return model.update(**state), None

        return self.recursive_apply(_update_state, states)[0]

    def update(self: Self, **kwargs) -> Self:
        return super().update(**kwargs)


# generic sequence and map containers


_idx_pattern = re.compile(r"_(\d+)")


class _SeqContainer(Parametric, Sequence, Generic[_T]):
    _seq: tuple[_T] = _internal_state()

    def __init__(self, *values: _T):
        object.__setattr__(self, "_seq", tuple(values))

    @overload
    def get(self, idx: int) -> _T: ...

    @overload
    def get(self, idx: slice) -> tuple[_T, ...]: ...

    def get(self, idx):
        return self._seq[idx]

    def set(self: Self, idx: int, val: _T) -> Self:
        kwargs = {f"_{idx}": val}
        return self.update(**kwargs)

    def update_guard(self, **kwargs) -> None:
        fnames = {f.name for f in self.fields()}
        super().update_guard(
            **{k: v for k, v in kwargs.items() if k in fnames}
        )

    def assign_(self, **kwargs) -> None:
        if not hasattr(self, "_seq"):  # init assign with _seq
            super().assign_(**kwargs)
            return
        values = list(self._seq)
        fnames = {f.name for f in self.fields()}
        updates = {}
        for k, v in kwargs.items():
            if k in fnames:
                updates[k] = v
            else:
                m = _idx_pattern.fullmatch(k)
                if not m:
                    raise ValueError(
                        f"invalid argument {k}, should be an index number "
                        f"prefixed by a '_'"
                    )
                else:
                    idx = int(m.group(1))
                    values[idx] = v
        super().assign_(_seq=tuple(values), **updates)

    def __getattr__(self, name: str) -> _T:
        m = _idx_pattern.fullmatch(name)
        if m:
            return self.get(int(m.group(1)))
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    @overload
    def __getitem__(self, idx: int) -> _T: ...

    @overload
    def __getitem__(self, idx: slice) -> tuple[_T, ...]: ...

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self) -> int:
        return len(self._seq)


class _MapContainer(Parametric, Mapping, Generic[_T]):
    _map: dict[str, _T] = _internal_state()

    def __init__(self, **items: _T):
        object.__setattr__(self, "_map", items)

    def update_guard(self, **kwargs) -> None:
        fnames = {f.name for f in self.fields()}
        super().update_guard(
            **{k: v for k, v in kwargs.items() if k in fnames}
        )

    def assign_(self, **kwargs) -> None:
        if not hasattr(self, "_map"):  # init assign with _seq
            super().assign_(**kwargs)
            return
        fnames = {f.name for f in self.fields()}
        updates = {}
        newmap = dict(self._map)
        for k, v in kwargs.items():
            if k in fnames:
                updates[k] = v
            elif k in newmap:
                newmap[k] = v
            else:
                raise ValueError(f"key {k} not found")
        super().assign_(_map=newmap, **updates)
        object.__setattr__(self, "_map", newmap)

    def __getattr__(self, name: str) -> _T:
        if name == "_map":
            raise AttributeError
        elif name in self._map:
            return self._map[name]
        else:
            raise AttributeError

    def __getitem__(self, key: str) -> _T:
        return self._map[key]

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[str]:
        return iter(self._map)

    def keys(self):
        yield from self._map.keys()

    def values(self):
        yield from self._map.values()

    def items(self):
        yield from self._map.items()


class _ArrayContainerMixin(Parametric):
    def update(self, **kwargs):
        assert all(isinstance(v, Array) for v in kwargs.values())
        return super().update(**kwargs)


# containers for parameters


class ParamSeq(_ArrayContainerMixin, _SeqContainer):
    def __init__(self, *items: Array):
        object.__setattr__(
            self,
            "_trainable_parameters",
            {f"_{i}": True for i in range(len(items))},
        )
        _SeqContainer.__init__(self, *items)

    def get_parameters(self) -> dict[str, Array]:
        return {f"_{i}": v for i, v in enumerate(self._seq)}

    def get_trainable(self, idx: int) -> bool:
        return self._trainable_parameters[f"_{idx}"]

    def set_trainable(self: Self, idx: int, trainable: bool) -> Self:
        trainable_map = self._trainable_parameters
        key = f"_{idx}"
        if key not in trainable_map:
            raise ValueError(f"invalid index {idx}")
        trainable_map = dict(trainable_map)
        trainable_map[key] = bool(trainable)
        newself = new(self)
        object.__setattr__(newself, "_trainable_parameters", trainable_map)
        return newself


class ParamMap(_ArrayContainerMixin, _MapContainer):
    def __init__(self, **items: Array):
        object.__setattr__(
            self, "_trainable_parameters", {k: True for k in items.keys()}
        )
        _MapContainer.__init__(self, **items)

    def get_parameters(self) -> dict[str, Array]:
        return dict(self._map)


# containers for buffers


class BufferSeq(_ArrayContainerMixin, _SeqContainer):
    def __init__(self, *items: Array):
        _SeqContainer.__init__(self, *items)

    def get_buffers(self) -> dict[str, Array]:
        return {f"_{i}": v for i, v in enumerate(self._seq)}


class BufferMap(_ArrayContainerMixin, _MapContainer):
    def __init__(self, **items: Array):
        _MapContainer.__init__(self, **items)

    def get_buffers(self) -> dict[str, Array]:
        return dict(self._map)


# containers for Modules


Parametric_Contra = TypeVar(
    "Parametric_Contra", bound="Parametric", contravariant=True
)


class _ParametricContainerMixin(Parametric):
    def update(self, **kwargs):
        assert all(isinstance(v, Parametric) for v in kwargs.values())
        return super().update(**kwargs)


class ModuleSeq(
    _ParametricContainerMixin, _SeqContainer, Generic[Parametric_T]
):
    def __init__(self, *items: Parametric_T):
        _SeqContainer.__init__(self, *items)

    def get_children(self) -> dict[str, Parametric_T]:
        return {f"_{i}": v for i, v in enumerate(self._seq)}


class ModuleMap(
    _ParametricContainerMixin, _MapContainer, Generic[Parametric_T]
):
    def __init__(self, **items: Parametric_T):
        _MapContainer.__init__(self, **items)

    def get_children(self) -> dict[str, Parametric_T]:
        return dict(self._map)
