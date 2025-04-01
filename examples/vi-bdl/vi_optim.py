# some variational inference optimizers
from typing import TypeAlias, TypeVar, Any
from typing_extensions import Self
from functools import partial
import abc
import numpy as np
from jax import Array, value_and_grad, vmap
import jax.numpy as jnp
import jax.tree as jtree
from jax.typing import ArrayLike
from jax.nn import softplus, sigmoid
from jax.lax import scan, rsqrt, square
from jax.random import split as jrsplit, normal as jrnormal
from ojnn import Module, config, state, schedulable, new
from ojnn.optim import Optimizer


ParamType: TypeAlias = TypeVar("ParamType")


def _welford_mean(avg: ParamType, newval: ParamType, count: int) -> ParamType:
    return jtree.map(lambda m, v: m + (v - m) / count, avg, newval)


def softplus_inv(x: float) -> float:
    return x + np.log(-np.expm1(-x))


def randn_like(rng: Array, t: ParamType) -> ParamType:
    tleaves, tdef = jtree.flatten(t)
    keys = jrsplit(rng, len(tleaves))
    randn = jrnormal
    samples = [randn(k, l.shape, l.dtype) for k, l in zip(keys, tleaves)]
    return jtree.unflatten(tdef, samples)


class VIOptimizer(Optimizer):
    lr: float = schedulable()
    ess: float = schedulable()
    momentum: float = schedulable()
    prior_prec: float = schedulable()
    init_std: float = schedulable()
    mc_samples: int = config()
    rngkey: Array = state()

    def __init__(
        self,
        params: ParamType,
        rngkey: Array,
        lr: float,
        ess: float,
        momentum: float = 0.0,
        prior_prec: float = 1.0,
        init_std: float = 0.01,
        mc_samples: int = 1,
    ):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError(
                "Invalid number of MC samples: {}".format(mc_samples)
            )
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not 0.0 <= prior_prec:
            raise ValueError("Invalid prior prec: {}".format(prior_prec))
        if not 0.0 < init_std:
            raise ValueError(
                "Invalid Hessian initialization: {}".format(init_std)
            )
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        super().__init__(params)
        self.assign_(
            lr=lr,
            ess=ess,
            momentum=momentum,
            prior_prec=prior_prec,
            init_std=init_std,
            rngkey=rngkey,
            mc_samples=mc_samples,
        )

    @property
    @abc.abstractmethod
    def std(self) -> ParamType:
        raise NotImplementedError

    @abc.abstractmethod
    def pre_accumulate(self: Self, old_mean: ParamType) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(
        self: Self, old_mean: ParamType, psample: ParamType, grad: ParamType
    ) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def post_accumulate(
        self: Self, old_mean: ParamType
    ) -> tuple[Self, ParamType]:
        raise NotImplementedError

    def update_params(
        self: Self, params: ParamType, grads: ParamType
    ) -> tuple[Self, ParamType]:
        lr = self.lr
        return self, jtree.map(lambda v, g: v - lr * g, params, grads)

    def sample_parameters(
        self: Self, params: ParamType, rngkey: Array
    ) -> ParamType:
        return jtree.map(
            lambda p, n, s: p + n * s,
            params,
            randn_like(rngkey, params),
            self.std,
        )

    def step(
        self: Self,
        model: Module,
        objective,
        *args,
        **kwargs,
    ) -> tuple[Self, Module, Array, Any]:
        keys = jrsplit(self.rngkey, self.mc_samples + 1)
        rngkey, ffkeys = keys[0], keys[1:]
        optimizer = self.update(rngkey=rngkey)
        old_mean = model.trainable_parameters()

        def _forward(p, m):
            m = m.load_states(p)
            m, l, a = objective(m, *args, **kwargs)
            return l, (m, a)

        def _step(state, key):
            model, optimizer = state
            psample = optimizer.sample_parameters(old_mean, key)
            (loss, (model, aux)), grads = value_and_grad(
                _forward, has_aux=True
            )(psample, model)
            optimizer = optimizer.accumulate(old_mean, psample, grads)
            return (model, optimizer), (loss, aux)

        optimizer = optimizer.pre_accumulate(old_mean)
        (model, optimizer), (losses, auxes) = scan(
            _step, (model, optimizer), ffkeys
        )
        optimizer, updates = optimizer.post_accumulate(old_mean)

        optimizer, new_mean = optimizer.update_params(old_mean, updates)
        return optimizer, model.load_states(new_mean), jnp.mean(losses), auxes


# Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015
class BayesByBackprop(VIOptimizer):
    rho: Any = state()
    acc_grad: Any = state()
    avg_grad: Any = state()
    avg_nxg: Any = state()
    count: ArrayLike = state()
    lr_rho: float = config()

    def __init__(
        self,
        params: ParamType,
        rngkey: Array,
        lr: float,
        ess: float,
        momentum: float = 0.0,
        prior_prec: float = 1.0,
        init_std: float = 0.01,
        mc_samples: int = 1,
    ):
        super().__init__(
            params, rngkey, lr, ess, momentum, prior_prec, init_std, mc_samples
        )
        init_rho = softplus_inv(init_std)
        self.assign_(
            rho=jtree.map(lambda a: jnp.full_like(a, init_rho), params),
            acc_grad=jtree.map(lambda a: jnp.zeros_like(a), params),
            avg_grad=None,
            avg_nxg=None,
            count=0,
            lr_rho=lr,
        )

    @property
    def std(self) -> ParamType:
        return jtree.map(softplus, self.rho)

    def pre_accumulate(self: Self, old_mean: ParamType) -> Self:
        avg_grad = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        avg_nxg = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        return new(self, avg_grad=avg_grad, avg_nxg=avg_nxg, count=0)

    def accumulate(
        self: Self, old_mean: ParamType, psample: ParamType, grad: ParamType
    ) -> Self:
        nxg = jtree.map(
            lambda m, p, g, s: (p - m) * g / s,
            old_mean,
            psample,
            grad,
            self.std,
        )
        count = self.count + 1
        avg_grad = _welford_mean(self.avg_grad, grad, count)
        avg_nxg = _welford_mean(self.avg_nxg, nxg, count)
        return self.update(avg_grad=avg_grad, avg_nxg=avg_nxg, count=count)

    # Eqs. 3 & 4 in Sec. 3.2 of the BBB paper
    # gaussian KL expression: https://stats.stackexchange.com/a/7449

    @staticmethod
    def _grad_avg(
        sample_grad: ParamType, avg: ParamType, prior_prec: float, ess: float
    ) -> ParamType:
        return jtree.map(
            lambda g, a: g + prior_prec / ess * a, sample_grad, avg
        )

    @staticmethod
    def _grad_rho(
        sample_nxg: ParamType, rho: ParamType, prior_prec: float, ess: float
    ) -> ParamType:
        return jtree.map(
            lambda n, r: sigmoid(r)
            * (n + (prior_prec * (s := softplus(r)) - 1 / s) / ess),
            sample_nxg,
            rho,
        )

    @staticmethod
    def _momentum(acc: ParamType, val: ParamType, m: float) -> ParamType:
        return jtree.map(lambda a, v: m * a + v, acc, val)

    @staticmethod
    def _grad_desc(vals: ParamType, grads: ParamType, lr: float) -> ParamType:
        return jtree.map(lambda v, g: v - lr * g, vals, grads)

    def post_accumulate(self: Self, old_mean: ParamType) -> Self:
        prior_prec = self.prior_prec
        ess = self.ess
        m = self.momentum
        rho = self.rho
        grad_avg = self._grad_avg(self.avg_grad, old_mean, prior_prec, ess)
        grad_rho = self._grad_rho(self.avg_nxg, rho, prior_prec, ess)
        acc_grad = self._momentum(self.acc_grad, grad_avg, m)
        rho = self._grad_desc(rho, grad_rho, self.lr_rho)
        return (
            new(
                self,
                rho=rho,
                acc_grad=acc_grad,
                avg_grad=None,
                avg_nxg=None,
                count=0,
            ),
            acc_grad,
        )


# VON with square gradient hessian estimator
# Khan et al., "Fast and Scalable Bayesian Deep Learning by Weight-Perturbation
# in Adam", ICML 2018
class SGVON(VIOptimizer):
    momentum_hess: float = config()
    step_vmap_in_axes: Any = config()
    hess: Any = state()
    acc_grad: Any = state()
    avg_grad: Any = state()
    avg_grad_sq: Any = state()
    count: ArrayLike = state()

    def __init__(
        self,
        params: ParamType,
        rngkey: Array,
        lr: float,
        ess: float,
        momentum: float = 0.0,
        momentum_hess: float = 0.99999,
        prior_prec: float = 1.0,
        init_std: float = 0.01,
        mc_samples: int = 1,
        step_vmap_in_axes: int | tuple | None = 0,
    ):
        super().__init__(
            params, rngkey, lr, ess, momentum, prior_prec, init_std, mc_samples
        )
        init_hess = ((init_std**-2) - prior_prec) / ess
        self.assign_(
            momentum_hess=momentum_hess,
            step_vmap_in_axes=step_vmap_in_axes,
            hess=jtree.map(lambda a: jnp.full_like(a, init_hess), params),
            acc_grad=jtree.map(lambda a: jnp.zeros_like(a), params),
            avg_grad=None,
            avg_grad_sq=None,
            count=0,
        )

    @property
    def std(self) -> ParamType:
        pp = self.prior_prec
        ess = self.ess
        return jtree.map(lambda h: rsqrt(h * ess + pp), self.hess)

    def pre_accumulate(self: Self, old_mean: ParamType) -> Self:
        avg_grad = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        avg_grad_sq = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        return new(self, avg_grad=avg_grad, avg_grad_sq=avg_grad_sq, count=0)

    def accumulate(self: Self, _, __, grad: ParamType) -> Self:
        count = self.count + 1
        avg_grad = _welford_mean(self.avg_grad, grad, count)
        avg_grad_sq = _welford_mean(
            self.avg_grad_sq, jtree.map(lambda g: square(g), grad), count
        )
        return self.update(
            avg_grad=avg_grad, avg_grad_sq=avg_grad_sq, count=count
        )

    @staticmethod
    def _lerp(
        start_val: ParamType, end_val: ParamType, weight: float
    ) -> ParamType:
        return jtree.map(lambda s, e: s + weight * (e - s), start_val, end_val)

    @staticmethod
    def _mmt_grad(
        acc: ParamType, grad: ParamType, m: float, avg: ParamType, wd: float
    ) -> ParamType:
        return jtree.map(lambda a, g, v: m * a + (g + wd * v), acc, grad, avg)

    @staticmethod
    def _nat_grad(
        sample_grad: ParamType, hess: ParamType, wd: float
    ) -> ParamType:
        return jtree.map(lambda g, h: g / (h + wd), sample_grad, hess)

    def post_accumulate(self: Self, old_mean: ParamType) -> Self:
        prior_prec = self.prior_prec
        ess = self.ess
        m1 = self.momentum
        m2 = self.momentum_hess
        wd = prior_prec / ess
        hess = self._lerp(self.hess, self.avg_grad_sq, 1 - m2)
        acc_grad = self._mmt_grad(
            self.acc_grad, self.avg_grad, m1, old_mean, wd
        )
        return (
            new(
                self,
                acc_grad=acc_grad,
                hess=hess,
                avg_grad=None,
                avg_grad_sq=None,
                count=0,
            ),
            self._nat_grad(acc_grad, hess, wd),
        )


# Osawa et al., "Practical Deep Learning with Bayesian Principles", NeurIPS
# 2019
class VOGN(SGVON):
    def accumulate(self: Self, _, __, grad_batch: ParamType) -> Self:
        count = self.count + 1
        avg_grad = _welford_mean(
            self.avg_grad,
            jtree.map(lambda g: jnp.mean(g, axis=0), grad_batch),
            count,
        )
        avg_grad_sq = _welford_mean(
            self.avg_grad_sq,
            jtree.map(lambda g: jnp.mean(square(g), axis=0), grad_batch),
            count,
        )
        return self.update(
            avg_grad=avg_grad, avg_grad_sq=avg_grad_sq, count=count
        )

    def step(
        self: Self,
        model: Module,
        objective,
        *args,
        **kwargs,
    ) -> tuple[Self, Module, Array, Any]:
        keys = jrsplit(self.rngkey, self.mc_samples + 1)
        rngkey, ffkeys = keys[0], keys[1:]
        optimizer = self.update(rngkey=rngkey)
        old_mean = model.trainable_parameters()

        def _ff(p, m, args, kwargs):
            m = m.load_states(p)
            m, l, a = objective(m, *args, **kwargs)
            return l, (m, a)

        def _step(state, key):
            model, optimizer = state
            psample = optimizer.sample_parameters(old_mean, key)
            (losses, (model, aux)), grad_batch = vmap(
                partial(value_and_grad(_ff, has_aux=True), psample, model),
                in_axes=self.step_vmap_in_axes,
                out_axes=((0, (None, 0)), 0),
            )(args, kwargs)
            optimizer = optimizer.accumulate(None, None, grad_batch)
            return (model, optimizer), (jnp.mean(losses), aux)

        optimizer = optimizer.pre_accumulate(old_mean)
        (model, optimizer), (losses, auxes) = scan(
            _step, (model, optimizer), ffkeys
        )
        optimizer, updates = optimizer.post_accumulate(old_mean)

        optimizer, new_mean = optimizer.update_params(old_mean, updates)
        return optimizer, model.load_states(new_mean), jnp.mean(losses), auxes


# Yuesong et al., "Variational Learning is Effective for Large Deep Networks",
# ICML 2024
class IVON(VIOptimizer):
    momentum_hess: float = config()
    hess: Any = state()
    acc_grad: Any = state()
    current_step: ArrayLike = state()
    avg_grad: Any = state()
    avg_nxg: Any = state()
    count: ArrayLike = state()

    def __init__(
        self,
        params: ParamType,
        rngkey: Array,
        lr: float,
        ess: float,
        momentum: float = 0.0,
        momentum_hess: float = 0.99999,
        prior_prec: float = 1.0,
        init_std: float = 0.01,
        mc_samples: int = 1,
    ):
        super().__init__(
            params, rngkey, lr, ess, momentum, prior_prec, init_std, mc_samples
        )
        init_hess = ((init_std**-2) - prior_prec) / ess
        self.assign_(
            momentum_hess=momentum_hess,
            hess=jtree.map(lambda a: jnp.full_like(a, init_hess), params),
            acc_grad=jtree.map(lambda a: jnp.zeros_like(a), params),
            avg_grad=None,
            avg_nxg=None,
            count=0,
            current_step=0,
        )

    @property
    def std(self) -> ParamType:
        pp = self.prior_prec
        ess = self.ess
        return jtree.map(lambda h: rsqrt(h * ess + pp), self.hess)

    def pre_accumulate(self: Self, old_mean: ParamType) -> Self:
        avg_grad = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        avg_nxg = jtree.map(lambda a: jnp.zeros_like(a), old_mean)
        return new(self, avg_grad=avg_grad, avg_nxg=avg_nxg, count=0)

    def accumulate(
        self: Self, old_mean: ParamType, psample: ParamType, grad: ParamType
    ) -> Self:
        nxg = jtree.map(lambda m, p, g: (p - m) * g, old_mean, psample, grad)
        count = self.count + 1
        avg_grad = _welford_mean(self.avg_grad, grad, count)
        avg_nxg = _welford_mean(self.avg_nxg, nxg, count)
        return self.update(avg_grad=avg_grad, avg_nxg=avg_nxg, count=count)

    @staticmethod
    def _update_hess(hess, avg_nxg, ess, m2, pp) -> ParamType:
        wd = pp / ess
        nll_hess = jtree.map(lambda a, h: ess * a * (h + wd), avg_nxg, hess)
        return jtree.map(
            lambda h, f: m2 * h
            + (1.0 - m2) * f
            + 0.5 * square((1.0 - m2) * (h - f)) / (h + wd),
            hess,
            nll_hess,
        )

    @staticmethod
    def _update_acc_grad(acc_grad, avg_grad, b1) -> ParamType:
        return jtree.map(
            lambda g, m: b1 * m + (1.0 - b1) * g, avg_grad, acc_grad
        )

    @staticmethod
    def _nat_grad(
        sample_grad: ParamType,
        avg: ParamType,
        hess: ParamType,
        prior_prec: float,
        ess: float,
        debias: float,
    ) -> ParamType:
        wd = prior_prec / ess
        return jtree.map(
            lambda g, a, h: (g / debias + wd * a) / (h + wd),
            sample_grad,
            avg,
            hess,
        )

    def post_accumulate(self: Self, old_mean: ParamType) -> Self:
        prior_prec = self.prior_prec
        ess = self.ess
        m1 = self.momentum
        m2 = self.momentum_hess
        current_step = self.current_step + 1
        debias = 1.0 - m1**current_step
        hess = self._update_hess(self.hess, self.avg_nxg, ess, m2, prior_prec)
        acc_grad = self._update_acc_grad(self.acc_grad, self.avg_grad, m1)
        return (
            new(
                self,
                acc_grad=acc_grad,
                hess=hess,
                current_step=current_step,
                avg_grad=None,
                avg_nxg=None,
                count=0,
            ),
            self._nat_grad(acc_grad, old_mean, hess, prior_prec, ess, debias),
        )
