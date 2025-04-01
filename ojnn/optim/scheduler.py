from collections.abc import Callable, Iterable
from typing import ClassVar, TypeVar, TypeAlias, Generic, Any
from typing_extensions import Self
from ..ftypes import StructField, Config, Schedulable, State, state
from ..struct import Struct
from .optimizer import Optimizer
from .from_optax import FromOptax


ScheduleFn: TypeAlias = Callable[[int], Any]


def _check_schedulable(optimizer: Optimizer, names: Iterable[str]) -> None:
    if isinstance(optimizer, FromOptax):
        raise ValueError(
            "Optax implementation has incompatible scheduling. Refer to its "
            "official documentation instead."
        )
    names = set(names)
    invalid_names = names.difference(
        f.name for f in optimizer.fields(Schedulable)
    )
    if invalid_names:
        raise ValueError(f"invalid names for schedule: {tuple(invalid_names)}")


def _check_config(optimizer: Optimizer, names: Iterable[str]) -> None:
    names = set(names)
    invalid_names = names.difference(
        f.name for f in optimizer.fields(Schedulable)
    )
    if invalid_names:
        raise ValueError(
            f"optimizer {optimizer} has no config names {tuple(invalid_names)}"
        )


Optimizer_T = TypeVar("Optimizer_T", bound=Optimizer)


class Scheduler(Struct, Generic[Optimizer_T]):
    admissible_field_types: ClassVar[tuple[type[StructField], ...]] = (
        Config,
        State,
    )
    schedules: dict[str, ScheduleFn]
    original_config: dict[str, Any] | None
    current_step: int = state()

    def __init__(
        self, optimizer: Optimizer_T | None = None, /, **schedules: ScheduleFn
    ):
        if optimizer is None:
            opt_config = None
        else:
            _check_schedulable(optimizer, schedules.keys())
            opt_config = {k: getattr(optimizer, k) for k in schedules.keys()}
        self.assign_(
            schedules=schedules, original_config=opt_config, current_step=0
        )

    def schedule(
        self: Self, optimizer: Optimizer_T
    ) -> tuple[Self, Optimizer_T]:
        schedules = self.schedules
        _check_schedulable(optimizer, schedules.keys())
        current_step = self.current_step
        opt_updates = {k: sch(current_step) for k, sch in schedules.items()}
        return (
            self.update(current_step=current_step + 1),
            optimizer.update(**opt_updates),
        )

    def reset(self: Self) -> Self:
        return self.update(current_step=0)

    def restore(self, optimizer: Optimizer_T) -> Optimizer_T:
        config = self.original_config
        if config is None:
            raise RuntimeError("No config to restore from.")
        else:
            _check_config(optimizer, config.keys())
            return optimizer.update(**config)
