from collections.abc import Callable, Sequence
from typing import ClassVar, TypeVar, TypeAlias, cast, Generic
from typing_extensions import Self
from functools import partial
import abc
import jax.numpy
from ..ftypes import config, Parameter, Buffer
from ..utils import KeyArg, KeyArray, maybe_split
from ..struct import new
from .container import Parametric, ModuleSeq, ModuleMap, RecursiveState
from .utils import _assert_no_key


DShape = TypeVar("DShape", bound=tuple)
DType = TypeVar("DType")
InputShape = TypeVar("InputShape", bound=Sequence, contravariant=True)
OutputShape = TypeVar("OutputShape", bound=tuple, covariant=True)
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)


# a segment of neural network
class Module(
    Parametric,
    Generic[InputType, OutputType, InputShape, OutputShape],
    metaclass=abc.ABCMeta,
):
    modes: ClassVar[tuple[str, ...]] = ("train", "eval")
    _mode: int = config(default=0, init=False)
    _initialized: bool = config(default=False, init=False)

    def __new__(cls, *args, **kwargs):
        # assure cls.modes is a tuple
        cls.modes = tuple(cls.modes)
        self = super().__new__(cls)
        object.__setattr__(self, "_mode", 0)
        # if empty PyTree state, then automatically initialized
        if all(
            (not isinstance(f, (Parameter, Buffer))) for f in self.fields()
        ):
            object.__setattr__(self, "_initialized", True)
        else:
            object.__setattr__(self, "_initialized", False)
        return self

    @property
    def reset_rngkey_count(self) -> int:
        return 0

    def reset(
        self: Self,
        input_shape: InputShape,
        rngkey: KeyArg = None,
    ) -> tuple[Self, OutputShape]:
        """initialize parameters, buffers here"""
        _assert_no_key(rngkey)
        return self, cast(OutputShape, tuple(input_shape))

    @property
    def forward_rngkey_count(self) -> int:
        return 0

    @abc.abstractmethod
    def forward(
        self: Self,
        input_value: InputType,
        rngkey: KeyArg = None,
    ) -> tuple[Self, OutputType]:
        raise NotImplementedError

    def init(
        self: Self,
        input_shape: InputShape,
        rngkey: KeyArg = None,
    ) -> Self:
        newself = new(self, _initialized=True)
        return newself.reset(input_shape, rngkey)[0]

    def __call__(
        self: Self,
        input_value: InputType,
        rngkey: KeyArg = None,
        parameters: RecursiveState | None = None,
    ) -> tuple[Self, OutputType]:
        if not self._initialized:
            raise RuntimeError("need to call .init() method first.")
        if parameters is not None:
            newself = self.load_states(parameters)
        else:
            newself = self
        if rngkey is None:
            return newself.forward(input_value)
        else:
            return newself.forward(input_value, rngkey)

    @property
    def mode(self) -> str:
        return self.modes[self._mode]

    def update_mode(self: Self, mode: str, recursive: bool = True) -> Self:
        if mode not in self.modes:
            raise ValueError(
                f"invalid Module mode '{mode}', expects one of {self.modes}"
            )
        else:
            mode_idx = self.modes.index(mode)

            def _upd_mode(m, _):
                newm = new(m, _mode=mode_idx)
                return newm, None

            if recursive:
                return self.recursive_apply(_upd_mode, None)[0]
            else:
                return _upd_mode(self, None)[0]


# Module compositors


class _ModuleContainerMixin(Parametric):
    @property
    def reset_rngkey_count(self) -> int:
        return sum(m.reset_rngkey_count for m in self.get_children().values())

    @property
    def forward_rngkey_count(self) -> int:
        return sum(
            m.forward_rngkey_count for m in self.get_children().values()
        )


class _SeqMixin(_ModuleContainerMixin, Module):
    def reset(
        self: Self,
        input_shape,
        rngkey: KeyArg = None,
    ):
        new_ms = {}
        shape = tuple(input_shape)
        rkc = self.reset_rngkey_count
        if rkc == 0:
            for k, m in self.get_children().items():
                m, shape = m.reset(shape)
                new_ms[k] = m
        else:
            idx = 0
            rngkey = maybe_split(rngkey, rkc)
            for k, m in self.get_children().items():
                c = m.reset_rngkey_count
                if c == 0:
                    m, shape = m.reset(shape)
                else:
                    rngkey = cast(KeyArray, rngkey)
                    key = rngkey[idx : idx + c]
                    m, shape = m.reset(shape, key)
                new_ms[k] = m
                idx += c
        return self.update(**new_ms), shape

    def forward(
        self: Self,
        input_value,
        rngkey: KeyArg = None,
    ):
        new_ms = {}
        result = input_value
        fkc = self.forward_rngkey_count
        if fkc == 0:
            for k, m in self.get_children().items():
                m, result = m.forward(result)
                new_ms[k] = m
        else:
            rngkey = maybe_split(rngkey, fkc)
            idx = 0
            for k, m in self.get_children().items():
                c = m.forward_rngkey_count
                if c == 0:
                    m, result = m.forward(result)
                else:
                    rngkey = cast(KeyArray, rngkey)
                    key = rngkey[idx : idx + c]
                    m, result = m.forward(result, key)
                new_ms[k] = m
                idx += c
        return self.update(**new_ms), result


class Sequential(_SeqMixin, ModuleSeq):
    def __init__(self, *modules: Module):
        ModuleSeq.__init__(self, *modules)


class NamedSequential(_SeqMixin, ModuleMap):
    def __init__(self, **modules: Module):
        ModuleMap.__init__(self, **modules)


ReduceFn: TypeAlias = Callable[[Sequence[DType]], DType]
ReduceShapeFn: TypeAlias = Callable[[Sequence[DShape]], DShape] | None


class _MapMixin(_ModuleContainerMixin, Module):
    reduce_fn: ReduceFn = config()
    reduce_shape_fn: ReduceShapeFn = config()

    def reset(
        self: Self,
        input_shape,
        rngkey: KeyArg = None,
    ):
        new_ms = {}
        shapes = []
        rkc = self.reset_rngkey_count
        if rkc == 0:
            for k, m in self.get_children().items():
                m, shape = m.reset(input_shape)
                new_ms[k] = m
                shapes.append(shape)
        else:
            rngkey = maybe_split(rngkey, rkc)
            idx = 0
            for k, m in self.get_children().items():
                c = m.reset_rngkey_count
                if c == 0:
                    m, shape = m.reset(input_shape)
                else:
                    rngkey = cast(KeyArray, rngkey)
                    key = rngkey[idx : idx + c]
                    m, shape = m.reset(input_shape, key)
                new_ms[k] = m
                shapes.append(shape)
                idx += c
        if self.reduce_shape_fn is None:
            if len(set(shapes)) > 1:
                raise ValueError(f"different output shapes {set(shapes)}")
            else:
                return self.update(**new_ms), shapes[0]
        else:
            return self.update(**new_ms), self.reduce_shape_fn(shapes)

    def forward(
        self: Self,
        input_value,
        rngkey: KeyArg = None,
    ):
        new_ms = {}
        results = []
        fkc = self.forward_rngkey_count
        if fkc == 0:
            for k, m in self.get_children().items():
                m, result = m.forward(input_value)
                new_ms[k] = m
                results.append(result)
        else:
            rngkey = maybe_split(rngkey, fkc)
            idx = 0
            for k, m in self.get_children().items():
                c = m.forward_rngkey_count
                if c == 0:
                    m, result = m.forward(input_value)
                else:
                    rngkey = cast(KeyArray, rngkey)
                    key = rngkey[idx : idx + c]
                    m, result = m.forward(input_value, key)
                new_ms[k] = m
                results.append(result)
                idx += c
        return self.update(**new_ms), self.reduce_fn(results)


class MapReduce(_MapMixin, ModuleSeq):
    def __init__(
        self,
        *modules: Module,
        reduce_fn: ReduceFn = sum,
        reduce_shape_fn: ReduceShapeFn = None,
    ):
        assert len(modules) > 0
        self.assign_(
            reduce_fn=reduce_fn,
            reduce_shape_fn=reduce_shape_fn,
        )
        ModuleSeq.__init__(self, *modules)


class NamedMapReduce(_MapMixin, ModuleMap):
    def __init__(
        self,
        *,
        reduce_fn: ReduceFn = sum,
        reduce_shape_fn: ReduceShapeFn = None,
        **modules: Module,
    ):
        assert len(modules) > 0
        self.assign_(
            reduce_fn=reduce_fn,
            reduce_shape_fn=reduce_shape_fn,
        )
        ModuleMap.__init__(self, **modules)


def _concat_shapes(
    shapes: Sequence[tuple[int, ...]], axis: int
) -> tuple[int, ...]:
    dims = set([len(s) for s in shapes])
    if len(dims) > 1:
        raise ValueError(f"different output dimensions {dims}")
    dim = next(iter(dims))
    new_shape = []
    if axis >= dim or axis < -dim:
        raise ValueError(f"invalid axis {axis} for {dim}D output")
    axis = axis + dim if axis < 0 else axis
    for d, ss in enumerate(zip(*shapes)):
        if d != axis:
            if len(set(ss)) > 1:
                raise ValueError(f"different sizes {set(ss)} at axis {d}")
            else:
                new_shape.append(ss[0])
        else:
            new_shape.append(sum(ss))
    return tuple(new_shape)


class MapConcat(MapReduce):
    axis: int = config()

    def __init__(self, *modules: Module, axis: int):
        assert len(modules) > 0
        self.assign_(axis=axis)
        super().__init__(
            *modules,
            reduce_fn=partial(jax.numpy.concatenate, axis=axis),
            reduce_shape_fn=partial(_concat_shapes, axis=axis),
        )


class NamedMapConcat(NamedMapReduce):
    axis: int = config()

    def __init__(self, *, axis: int, **modules: Module):
        assert len(modules) > 0
        self.assign_(axis=axis)
        super().__init__(
            **modules,
            reduce_fn=partial(jax.numpy.concatenate, axis=axis),
            reduce_shape_fn=partial(_concat_shapes, axis=axis),
        )
