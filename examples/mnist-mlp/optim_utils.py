from typing_extensions import Self
from jax import Array
import jax.numpy as jnp
from jax.nn import log_softmax
from jax.tree import map as tree_map
from jax.numpy import (
    argmax,
    take_along_axis,
    expand_dims,
    clip,
    pi,
    cos,
)
from jax.lax import cond as lax_cond
from ojnn import state, schedulable
from ojnn.optim import Optimizer
from ojnn.optim.optimizer import ModuleParam


def acc(output: Array, target: Array) -> Array:
    preds = argmax(output, axis=-1)
    return (preds == target).mean()


def cross_entropy(logits: Array, labels: Array) -> Array:
    return -take_along_axis(
        log_softmax(logits, axis=-1), expand_dims(labels, axis=-1), axis=-1
    ).mean()


def make_linear_schedule(
    value_start: float, value_end: float, step_start: float, step_end: float
):
    assert step_end > step_start

    def linear_schedule(step):
        step = clip(step, min=step_start, max=step_end)
        return (
            value_end * (step - step_start) + value_start * (step_end - step)
        ) / (step_end - step_start)

    return linear_schedule


def make_warmup_cosine_schedule(
    value_start: float, value_end: float, step_start: float, step_end: float
):
    assert step_end > step_start

    def warmup_cosine_schedule(step):
        step = clip(step, max=step_end)
        return lax_cond(
            step < step_start,
            lambda: (
                value_start * step + 1e-2 * value_start * (step_start - step)
            )
            / step_start,
            lambda: (
                0.5
                * (value_start - value_end)
                * cos(pi * (step - step_start) / (step_end - step_start))
                + 0.5 * (value_start + value_end)
            ),
        )

    return warmup_cosine_schedule


class SGD(Optimizer):
    # init args
    lr: float = schedulable()
    momentum: float | None = schedulable()
    weight_decay: float = schedulable()
    nesterov: bool = schedulable()
    # optimizer states
    accumulation: ModuleParam | None = state()

    def __init__(
        self,
        parameters: ModuleParam,
        lr: float,
        momentum: float | None = None,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(parameters)
        if momentum is None:
            accumulation = None
        else:
            accumulation = tree_map(lambda a: jnp.zeros_like(a), parameters)
        self.assign_(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            accumulation=accumulation,
        )

    def update_params(
        self: Self, params: ModuleParam, grads: ModuleParam
    ) -> tuple[Self, ModuleParam]:
        lr, m, wd = self.lr, self.momentum, self.weight_decay
        grads = tree_map(lambda g, p: g + wd * p, grads, params)
        accu = self.accumulation
        if accu is None:
            new_accu = None
            updates = grads
        else:

            def _accumulate(g, a):
                return g + m * a

            new_accu = tree_map(_accumulate, grads, accu)
            updates = lax_cond(
                self.nesterov,
                lambda: tree_map(_accumulate, grads, new_accu),
                lambda: new_accu,
            )
        new_params = tree_map(lambda p, g: p - lr * g, params, updates)
        return self.update(accumulation=new_accu), new_params
