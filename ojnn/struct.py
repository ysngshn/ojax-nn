from typing import ClassVar, TypeVar
from typing_extensions import Self
import dataclasses
import inspect
import ojax

from .ftypes import (
    StructField,
    _InternalState,
    _field_fn_map,
    _check_schedulable,
    _check_parameter,
    _check_buffer,
    Buffer,
    Parameter,
    Schedulable,
    Const,
    Config,
)


class Struct(ojax.OTree):
    admissible_field_types: ClassVar[tuple[type[StructField], ...]] = (Config,)

    @classmethod
    def infer_field_type(cls, f: dataclasses.Field) -> type[StructField]:
        return Config

    @classmethod
    def field_type(cls, f: dataclasses.Field) -> type[StructField]:
        if isinstance(f, StructField):
            return type(f)
        else:
            return cls.infer_field_type(f)

    def __init_subclass__(cls, **kwargs):
        subcls = super().__init_subclass__(**kwargs)
        # ensure valid admissible field type definition as tuple
        admissible_field_types = tuple(subcls.admissible_field_types)
        for f in admissible_field_types:
            if (
                (not inspect.isclass(f))
                or (not issubclass(f, StructField))
                or (f is StructField)
            ):
                raise ValueError(f"Invalid admissible field type: {f}")
        subcls.admissible_field_types = admissible_field_types
        # update for typing.dataclass_transform
        fs = subcls.admissible_field_types + (_InternalState,)
        ffns = tuple(_field_fn_map[f] for f in fs)
        subcls.__dataclass_transform__["field_specifiers"] = fs + ffns
        # check fields for inadmissible field type
        for f in dataclasses.fields(subcls):
            if not issubclass(
                subcls.field_type(f),
                (subcls.admissible_field_types, _InternalState),
            ):
                raise ValueError(
                    f"field {f.name} of class {subcls.__name__} has "
                    f"inadmissible type {subcls.field_type(f)}, expect one of "
                    f"{subcls.admissible_field_types}"
                )
        return subcls

    @classmethod
    def fields(
        cls,
        f_type: (
            type[StructField] | tuple[type[StructField], ...] | None
        ) = None,
        infer: bool = True,
    ) -> tuple[dataclasses.Field, ...]:
        c_fields = dataclasses.fields(cls)
        if f_type is None:
            f_type = cls.admissible_field_types
        if infer:
            return tuple(
                f for f in c_fields if issubclass(cls.field_type(f), f_type)
            )
        else:
            return tuple(f for f in c_fields if isinstance(f, f_type))

    def update_guard(self, **kwargs) -> None:
        field_names = {
            f.name: f
            for f in self.fields(
                (*self.admissible_field_types, _InternalState)
            )
        }
        arg_names = set(kwargs.keys())
        if not arg_names.issubset(field_names):
            raise ValueError(
                f"Unrecognized fields: {arg_names.difference(field_names)}"
            )
        for k, v in kwargs.items():
            f = self.__dataclass_fields__[k]
            ft = self.field_type(f)
            if issubclass(ft, Parameter):
                _check_parameter(f, v)
            if issubclass(ft, Buffer):
                _check_buffer(f, v)
            elif issubclass(ft, Const):
                if hasattr(self, k):
                    raise ValueError(f"Cannot update constant field {k}")
            elif issubclass(ft, Schedulable):
                _check_schedulable(f, v)

    def update(self: Self, **kwargs) -> Self:
        self.update_guard(**kwargs)
        return super().update(**kwargs)


Struct_T = TypeVar("Struct_T", bound=Struct)


def new(obj: Struct_T, **kwargs) -> Struct_T:
    return ojax.new(obj, **kwargs)
