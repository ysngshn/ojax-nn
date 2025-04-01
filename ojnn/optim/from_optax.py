from __future__ import annotations
from collections.abc import Callable
from typing import Any, cast, Concatenate, TypeVar
from typing_extensions import Self
from functools import wraps
from jax import Array, value_and_grad
from jax.lax import cond as lax_cond
import jax.numpy as jnp
from ojnn.modules.module import Module
from optax import (  # type: ignore
    GradientTransformation,
    TransformInitFn,
    TransformUpdateFn,
    TransformUpdateExtraArgsFn,
    ScalarOrSchedule,
    OptState,
    MaskOrFn,
    apply_updates,
    scale_by_zoom_linesearch,
    adabelief,
    adadelta,
    adan,
    adafactor,
    adagrad,
    adam,
    adamw,
    adamax,
    adamaxw,
    amsgrad,
    fromage,
    lamb,
    lars,
    lbfgs,
    lion,
    nadam,
    nadamw,
    noisy_sgd,
    novograd,
    optimistic_gradient_descent,
    polyak_sgd,
    optimistic_adam,
    radam,
    rmsprop,
    rprop,
    sgd,
    sign_sgd,
    sm3,
    yogi,
    chain,
    add_decayed_weights,
    inject_hyperparams as optax_inject_hyperparams,
)
from optax.tree_utils import (  # type: ignore
    tree_get,
    tree_l2_norm,
)
from ..ftypes import config, state
from .optimizer import Optimizer, ModuleParam, Aux_T, ObjectiveFn, _P


class FromOptax(Optimizer):
    optax_fn: Callable[..., GradientTransformation] = config()
    configs: dict[str, Any] = config()
    init_fn: TransformInitFn = config()
    update_fn: TransformUpdateFn = config()
    states: OptState = state()

    def __init__(
        self,
        optax_fn: Callable[..., GradientTransformation],
        params: ModuleParam,
        **configs,
    ):
        super().__init__(params)
        gt = self._make_gradient_transform(optax_fn, **configs)
        states = gt.init(params)
        self.assign_(
            optax_fn=optax_fn,
            configs=configs,
            init_fn=gt.init,
            update_fn=gt.update,
            states=states,
        )

    @staticmethod
    def _make_gradient_transform(
        optax_fn: Callable[..., GradientTransformation],
        **configs,
    ) -> GradientTransformation:
        return optax_fn(**configs)

    def update_params(
        self: Self, params: ModuleParam, grads: ModuleParam
    ) -> tuple[Self, ModuleParam]:
        updates, states = self.update_fn(grads, self.states, params)
        params = apply_updates(params, updates)
        return self.update(states=states), cast(ModuleParam, params)


class WithWeightDecay(FromOptax):
    def __init__(
        self,
        optax_fn: Callable[..., GradientTransformation],
        params: ModuleParam,
        **configs,
    ):
        super().__init__(optax_fn, params, **configs)

    @staticmethod
    def _make_gradient_transform(
        optax_fn: Callable[..., GradientTransformation],
        **configs,
    ) -> GradientTransformation:
        wd = configs.pop("weight_decay")
        msk = configs.pop("weight_decay_mask", None)
        if wd is None:
            return optax_fn(**configs)
        else:
            return chain(
                add_decayed_weights(weight_decay=wd, mask=msk),
                optax_fn(**configs),
            )


def inject_hyperparams(cls: type[FromOptax]):
    old_make_gt = cls._make_gradient_transform

    if issubclass(cls, WithWeightDecay):

        @wraps(old_make_gt)
        def _ih_make_gt(
            optax_fn: Callable[..., GradientTransformation],
            **configs,
        ) -> GradientTransformation:
            wd = configs.pop("weight_decay")
            if wd is None:
                return optax_inject_hyperparams(optax_fn)(**configs)
            else:
                msk = configs.pop("weight_decay_mask", None)
                return chain(
                    optax_inject_hyperparams(add_decayed_weights)(
                        weight_decay=wd, mask=msk
                    ),
                    optax_inject_hyperparams(optax_fn)(**configs),
                )

        cls._make_gradient_transform = staticmethod(_ih_make_gt)
    elif issubclass(cls, FromOptax):

        @wraps(old_make_gt)
        def _ih_make_gt(
            optax_fn: Callable[..., GradientTransformation],
            **configs,
        ) -> GradientTransformation:
            return optax_inject_hyperparams(optax_fn)(**configs)

        cls._make_gradient_transform = staticmethod(_ih_make_gt)
    else:
        raise ValueError(f"Expects subclass of FromOptax, received {cls}")

    return cls


InjectHyperparamsType = TypeVar("InjectHyperparamsType", bound=FromOptax)


def update_hyperparams(
    optimizer: InjectHyperparamsType, **updates
) -> InjectHyperparamsType:
    states = optimizer.states
    states.hyperparams.update(updates)
    return optimizer.update(states=states)


class AdaBelief(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += adabelief.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-16,
        eps_root: float = 1e-16,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        *,
        nesterov: bool = False,
    ):
        super().__init__(
            adabelief,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=nesterov,
        )


class Adadelta(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += adadelta.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule | None = None,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            adadelta,
            params,
            learning_rate=learning_rate,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class Adan(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += adan.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.98,
        b2: float = 0.92,
        b3: float = 0.99,
        eps: float = 1e-8,
        eps_root: float = 1e-8,
        weight_decay: float | None = None,
        mask=None,
    ):
        super().__init__(
            adan,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            b3=b3,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            mask=mask,
        )


class Adafactor(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += adafactor.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule | None = None,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: float | None = 1.0,
        momentum: float | None = None,
        dtype_momentum: Any = jnp.float32,
        weight_decay_rate: float | None = None,
        eps: float = 1e-30,
        factored: bool = True,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            adafactor,
            params,
            learning_rate=learning_rate,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            clipping_threshold=clipping_threshold,
            momentum=momentum,
            dtype_momentum=dtype_momentum,
            weight_decay_rate=weight_decay_rate,
            eps=eps,
            factored=factored,
            weight_decay_mask=weight_decay_mask,
        )


class AdaGrad(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += adagrad.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        initial_accumulator_value: float = 0.1,
        eps: float = 1e-7,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            adagrad,
            params,
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class Adam(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += adam.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Any = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        *,
        nesterov: bool = False,
    ):
        super().__init__(
            adam,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=nesterov,
        )


class AdamW(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += adamw.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Any = None,
        weight_decay: float = 1e-4,
        mask: MaskOrFn = None,
        *,
        nesterov: bool = False,
    ):
        super().__init__(
            adamw,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            mask=mask,
            nesterov=nesterov,
        )


class Adamax(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += adamax.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            adamax,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class AdamaxW(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += adamaxw.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        mask: MaskOrFn = None,
    ):
        super().__init__(
            adamaxw,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            mask=mask,
        )


class AMSGrad(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += amsgrad.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Any = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            amsgrad,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class Fromage(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += fromage.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: float,
        min_norm: float = 1e-6,
    ):
        super().__init__(
            fromage,
            params,
            learning_rate=learning_rate,
            min_norm=min_norm,
        )


class LAMB(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += lamb.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-6,
        eps_root: float = 0.0,
        weight_decay: float | None = None,
        mask: MaskOrFn = None,
    ):
        super().__init__(
            lamb,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            mask=mask,
        )


class LARS(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += lars.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = True,
        trust_coefficient: float = 0.001,
        eps: float = 0.0,
        trust_ratio_mask: MaskOrFn = True,
        momentum: float = 0.9,
        nesterov: bool = False,
    ):
        super().__init__(
            lars,
            params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            trust_coefficient=trust_coefficient,
            eps=eps,
            trust_ratio_mask=trust_ratio_mask,
            momentum=momentum,
            nesterov=nesterov,
        )


# variant of value_and_grad_from_state adapted to our interface
def _value_and_grad_from_state(
    objective: ObjectiveFn[_P, Aux_T],
) -> Callable[
    Concatenate[ModuleParam, Module, OptState, _P],
    tuple[tuple[Array, tuple[Module, Aux_T]], ModuleParam],
]:
    def _value_model_aux_and_grad(
        params: ModuleParam,
        model: Module,
        opt_states: OptState,
        *fn_args: _P.args,
        **fn_kwargs: _P.kwargs,
    ) -> tuple[tuple[Array, tuple[Module, Aux_T]], ModuleParam]:
        def _forward(p, m):
            m = m.update_states(p)
            m, l, a = objective(m, *fn_args, **fn_kwargs)
            return l, (m, a)

        value = tree_get(opt_states, "value")
        grad = tree_get(opt_states, "grad")
        if (value is None) or (grad is None):
            raise ValueError(
                "Value or gradient not found in the state. "
                "Make sure that these values are stored in the state by the "
                "optimizer."
            )
        (value, (model, aux)), grad = lax_cond(
            (~jnp.isinf(value)) & (~jnp.isnan(value)),
            lambda *_: (value, grad),
            lambda p, a, kwa: value_and_grad(_forward, has_aux=True)(p, model),
            params,
            fn_args,
            fn_kwargs,
        )
        return (value, (model, aux)), grad

    return _value_model_aux_and_grad


class LBFGS(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += lbfgs.__doc__

    configs: dict[str, Any] = config()
    init_fn: TransformInitFn = config()
    update_fn: TransformUpdateExtraArgsFn | TransformUpdateFn = config()
    states: OptState = state()
    has_linesearch: bool = config()

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        linesearch=scale_by_zoom_linesearch(max_linesearch_steps=15),
    ):
        super().__init__(
            lbfgs,
            params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
            linesearch=linesearch,
        )
        self.assign_(has_linesearch=linesearch is not None)

    def update_params(
        self: Self, params: ModuleParam, grads: ModuleParam
    ) -> tuple[Self, ModuleParam]:
        if self.has_linesearch:
            raise RuntimeError(
                "LBFGS with line search has no .update_params() method, use "
                ".step instead"
            )
        else:
            return super().update_params(params, grads)

    def _step_linesearch(
        self: Self,
        model: Module,
        objective: ObjectiveFn[_P, Aux_T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> tuple[Self, Module, Array, Aux_T]:
        old_params = model.trainable_parameters()
        opt_states = self.states

        def _forward(p, m):
            m = m.update_states(p)
            m, l, a = objective(m, *args, **kwargs)
            return l, (m, a)

        def _value_fn(p):
            return _forward(p, model)[0]

        (loss, (model, aux)), grads = _value_and_grad_from_state(objective)(
            old_params, model, opt_states, *args, **kwargs
        )
        updates, opt_states = self.update_fn(
            grads,
            opt_states,
            old_params,
            value=loss,
            grad=grads,
            value_fn=_value_fn,
        )
        new_params = cast(ModuleParam, apply_updates(old_params, updates))
        return (
            self.update(states=opt_states),
            model.load_states(new_params),
            loss,
            aux,
        )

    def step(
        self: Self,
        model: Module,
        objective: ObjectiveFn[_P, Aux_T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> tuple[Self, Module, Array, Aux_T]:
        if self.has_linesearch:
            return self._step_linesearch(model, objective, *args, **kwargs)
        else:
            return super().step(model, objective, *args, **kwargs)

    # properties for stop criteria
    # cf.: optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html

    @property
    def error(self) -> Array:
        if self.has_linesearch:
            grad = tree_get(self.states, "grad")
            return tree_l2_norm(grad)
        else:
            raise RuntimeError("")

    @property
    def iter_num(self) -> Array:
        return tree_get(state, "count")


class Lion(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += lion.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.99,
        mu_dtype: Any = None,
        weight_decay: float = 1e-3,
        mask: MaskOrFn = None,
    ):
        super().__init__(
            lion,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            mask=mask,
        )


class NAdam(Adam):
    """
    (Documentation from Optax)

    """

    __doc__ += nadam.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Any = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=True,
        )


class NAdamW(AdamW):
    """
    (Documentation from Optax)

    """

    __doc__ += nadamw.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Any = None,
        weight_decay: float = 1e-4,
        mask: MaskOrFn = None,
    ):
        super().__init__(
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay,
            mask=mask,
            nesterov=True,
        )


class NoisySGD(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += noisy_sgd.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        eta: float = 0.01,
        gamma: float = 0.55,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        seed: int = 0,
    ):
        super().__init__(
            noisy_sgd,
            params,
            learning_rate=learning_rate,
            eta=eta,
            gamma=gamma,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            seed=seed,
        )


class NovoGrad(FromOptax):
    """
    (Documentation from Optax)

    """

    __doc__ += novograd.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.25,
        eps: float = 1e-6,
        eps_root: float = 0.0,
        weight_decay: float | None = None,
    ):
        super().__init__(
            novograd,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
        )


class OptimisticGD(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += optimistic_gradient_descent.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        alpha: ScalarOrSchedule = 1.0,
        beta: ScalarOrSchedule = 1.0,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            optimistic_gradient_descent,
            params,
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class OptimisticAdam(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += optimistic_adam.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        optimism: float | None = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        eps_root: float = 0.0,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        mu_dtype: Any = None,
        *,
        nesterov: bool = True,
    ):
        super().__init__(
            optimistic_adam,
            params,
            learning_rate=learning_rate,
            optimism=optimism,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        )


class PolyakSGD(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += polyak_sgd.__doc__

    configs: dict[str, Any] = config()
    init_fn: TransformInitFn = config()
    update_fn: TransformUpdateExtraArgsFn = config()
    states: OptState = state()

    def __init__(
        self,
        params: ModuleParam,
        max_learning_rate: float = 1.0,
        scaling: ScalarOrSchedule = 1.0,
        f_min: float = 0.0,
        eps: float = 0.0,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            polyak_sgd,
            params,
            max_learning_rate=max_learning_rate,
            scaling=scaling,
            f_min=f_min,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )

    def update_params(self: Self, _, __):
        raise RuntimeError(
            "PolyakSGD has no .update_params method, use .step instead"
        )

    def step(
        self: Self,
        model: Module,
        objective: Callable[..., tuple[Module, Array, Aux_T]],
        *args,
        **kwargs,
    ) -> tuple[Self, Module, Array, Aux_T]:
        old_params = model.trainable_parameters()
        opt_states = self.states

        def _forward(params, m):
            m = m.update_states(params)
            m, l, a = objective(m, *args, **kwargs)
            return l, (m, a)

        (loss, (model, aux)), grads = value_and_grad(_forward, has_aux=True)(
            old_params, model
        )
        updates, opt_states = self.update_fn(
            grads,
            opt_states,
            old_params,
            value=loss,
        )
        new_params = apply_updates(old_params, updates)
        return (
            self.update(states=opt_states),
            model.load_states(new_params),
            loss,
            aux,
        )


class RAdam(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += radam.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        threshold: float = 5.0,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        *,
        nesterov: bool = False,
    ):
        super().__init__(
            radam,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            threshold=threshold,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=nesterov,
        )


class RMSProp(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += rmsprop.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        decay: float = 0.9,
        eps: float = 1e-8,
        initial_scale: float = 0.0,
        eps_in_sqrt: bool = True,
        centered: bool = False,
        momentum: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        nesterov: bool = False,
        bias_correction: bool = False,
    ):
        super().__init__(
            rmsprop,
            params,
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            centered=centered,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=nesterov,
            bias_correction=bias_correction,
        )


class RProp(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += rprop.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        eta_minus: float = 0.5,
        eta_plus: float = 1.2,
        min_step_size: float = 1e-06,
        max_step_size: float = 50.0,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            sgd,
            params,
            learning_rate=learning_rate,
            eta_minus=eta_minus,
            eta_plus=eta_plus,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class SGD(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += sgd.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        momentum: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
        nesterov: bool = False,
        accumulator_dtype: Any = None,
    ):
        super().__init__(
            sgd,
            params,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            nesterov=nesterov,
            accumulator_dtype=accumulator_dtype,
        )


class SignSGD(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += sign_sgd.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            sign_sgd,
            params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class SM3(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += sm3.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            sm3,
            params,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


class Yogi(WithWeightDecay):
    """
    (Documentation from Optax)

    """

    __doc__ += yogi.__doc__

    def __init__(
        self,
        params: ModuleParam,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-3,
        weight_decay: float | None = None,
        weight_decay_mask: MaskOrFn = None,
    ):
        super().__init__(
            yogi,
            params,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            ep=eps,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
        )
