import abc
from collections.abc import Callable
from typing import (
    TypeVar,
    ClassVar,
    Generic,
    TypeAlias,
    Concatenate,
    ParamSpec,
)
from typing_extensions import Self
from jax import value_and_grad, Array
from jax.tree import leaves as tree_leaves
from ..ftypes import (
    Config,
    State,
    Const,
    Schedulable,
    StructField,
)
from ..struct import Struct
from ..modules.container import RecursiveState
from ..modules.module import Module


_P = ParamSpec("_P")
ModuleParam: TypeAlias = RecursiveState[dict[str, Array]]
Aux_T = TypeVar("Aux_T")
ObjectiveFn: TypeAlias = Callable[
    Concatenate[Module, _P], tuple[Module, Array, Aux_T]
]


class Optimizer(Struct, Generic[Aux_T], metaclass=abc.ABCMeta):
    admissible_field_types: ClassVar[tuple[type[StructField], ...]] = (
        Config,
        State,
        Const,
        Schedulable,
    )

    def __init__(self, params: ModuleParam):
        if not tree_leaves(params):
            raise ValueError("No paramter to optimize")

    @abc.abstractmethod
    def update_params(
        self: Self, params: ModuleParam, grads: ModuleParam
    ) -> tuple[Self, ModuleParam]:
        raise NotImplementedError

    def step(
        self: Self,
        model: Module,
        objective: ObjectiveFn,
        *args,
        **kwargs,
    ) -> tuple[Self, Module, Array, Aux_T]:
        old_params = model.trainable_parameters()

        def _forward(p, m):
            m = m.load_states(p)
            m, l, a = objective(m, *args, **kwargs)
            return l, (m, a)

        (loss, (model, aux)), grads = value_and_grad(_forward, has_aux=True)(
            old_params, model
        )
        newself, new_params = self.update_params(old_params, grads)
        return newself, model.load_states(new_params), loss, aux


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
TargetType = TypeVar("TargetType")


def make_supervised_objective(
    lossfn: Callable[[OutputType, TargetType], Array]
) -> Callable[[Module, InputType, TargetType], tuple[Module, Array, Array]]:
    def objective(
        model: Module, inputs, target, rngkey=None
    ) -> tuple[Module, Array, Array]:
        model, output = model(inputs, rngkey=rngkey)
        loss = lossfn(output, target)
        return model, loss, output

    return objective
