from typing import Sequence, Callable
from typing_extensions import Self
import abc
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jax.lax import scan as lax_scan
from jax.random import split as jrsplit
import jax.numpy as jnp
from ..ftypes import parameter, buffer, child, config, make_default_array
from ..utils import KeyArg, maybe_split
from ..struct import new
from .utils import _assert_no_key, _assert_negative_axis
from .module import (
    Module,
    Sequential,
    MapReduce,
    MapConcat,
    NamedSequential,
    NamedMapReduce,
    NamedMapConcat,
)
from .init import zeros, eye, identity_or_orthogonal, uniform
from .linear import dense
from .activation import tanh, sigmoid


class RecStep(Module, metaclass=abc.ABCMeta):
    recurrent_state_initialized: bool = config(default=False, init=False)

    @abc.abstractmethod
    def reset(
        self: Self,
        input_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        return self, tuple(input_shape)

    @property
    @abc.abstractmethod
    def reset_state_rngkey_count(self) -> int:
        return 0

    @abc.abstractmethod
    def reset_recurrent_state(
        self: Self,
        batch_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> Self:
        return new(self, recurrent_state_initialized=True)

    @abc.abstractmethod
    def unset_recurrent_state(self: Self) -> Self:
        return new(self, recurrent_state_initialized=False)

    @property
    @abc.abstractmethod
    def recurrent_state(self):
        return None

    @abc.abstractmethod
    def forward(
        self: Self,
        input_value: ArrayLike,
        rngkey: KeyArg = None,
    ) -> tuple[Self, Array]:
        return self, jnp.asarray(input_value)

    def __call__(self, *args, **kwargs):
        if not self.recurrent_state_initialized:
            raise RuntimeError("recurrent state is not set.")
        return super().__call__(*args, **kwargs)


class _RecStepContainerExtension(RecStep):

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        raise RuntimeError("reset method should not be called")

    def forward(
        self: Self, input_value: ArrayLike, rngkey: KeyArg = None
    ) -> tuple[Self, Array]:
        raise RuntimeError("forward method should not be called")

    @property
    def reset_state_rngkey_count(self) -> int:
        return sum(
            m.reset_state_rngkey_count if isinstance(m, RecStep) else 0
            for m in self.get_children().values()
        )

    def reset_recurrent_state(
        self: Self,
        batch_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> Self:
        idx = 0
        new_children = {}
        if self.reset_state_rngkey_count == 0:
            rngkey = None
        else:
            rngkey = maybe_split(rngkey, self.reset_state_rngkey_count)
        for k, m in self.get_children().items():
            if not isinstance(m, RecStep):
                continue
            c = m.reset_state_rngkey_count
            if c == 0:
                key = None
            else:
                key = rngkey[idx : idx + c]
            m = m.reset_recurrent_state(batch_shape, key)
            new_children[k] = m
            idx += c
        return self.update(**new_children)

    def unset_recurrent_state(self: Self) -> Self:
        new_children = {
            k: (m.unset_recurrent_state() if isinstance(m, RecStep) else m)
            for k, m in self.get_children().items()
        }
        return self.update(**new_children)

    @property
    def recurrent_state(self) -> dict:
        return {
            k: (m.recurrent_state if isinstance(m, RecStep) else None)
            for k, m in self.get_children().items()
        }


class SequentialRecStep(Sequential, _RecStepContainerExtension):
    pass


class MapReduceRecStep(MapReduce, _RecStepContainerExtension):
    pass


class MapConcatRecStep(MapConcat, _RecStepContainerExtension):
    pass


class NamedSequentialRecStep(NamedSequential, _RecStepContainerExtension):
    pass


class NamedMapReduceRecStep(NamedMapReduce, _RecStepContainerExtension):
    pass


class NamedMapConcatRecStep(NamedMapConcat, _RecStepContainerExtension):
    pass


class RNNStep(RecStep):
    out_features: int = config()
    activation: Callable[[ArrayLike], Array] = config()
    dtype: DTypeLike | None = config()
    weight_ih: Array = parameter()
    weight_hh: Array = parameter()
    bias: Array = parameter()
    state_h: Array = buffer()

    def __init__(
        self,
        out_features: int,
        activation: Callable[[ArrayLike], Array] = tanh,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        if not with_bias:
            self.assign_(bias=None)
        self.assign_(
            out_features=out_features,
            activation=activation,
            dtype=dtype,
        )
        super().__init__()

    @property
    def reset_rngkey_count(self) -> int:
        return 1

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        dtype = self.dtype
        in_features, out_features = input_shape[-1], self.out_features
        output_shape = tuple(input_shape[:-1]) + (out_features,)
        rngkey = maybe_split(rngkey, 1)[0]
        weight_ih = identity_or_orthogonal(
            (out_features, in_features), rngkey=rngkey, dtype=dtype
        )
        weight_hh = eye(out_features, out_features, dtype=dtype)
        if self.bias is None:
            bias = None
        else:
            bias = zeros((out_features,), dtype=dtype)
        return (
            new(
                self,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
            ),
            output_shape,
        )

    @property
    def reset_state_rngkey_count(self) -> int:
        return 0

    def reset_recurrent_state(
        self: Self, batch_shape: Sequence[int], _=None
    ) -> Self:
        _assert_no_key(_)
        dtype = self.dtype
        state_h = zeros([*batch_shape, self.out_features], dtype=dtype)
        return new(self, state_h=state_h, recurrent_state_initialized=True)

    def unset_recurrent_state(self: Self) -> Self:
        return new(
            self,
            state_h=make_default_array(),
            recurrent_state_initialized=False,
        )

    @property
    def recurrent_state(self) -> dict[str, Array]:
        return {"state_h": self.state_h}

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        output = self.activation(
            dense(x, self.weight_ih, self.bias)
            + dense(self.state_h, self.weight_hh)
        )
        return self.update(state_h=output), output


class LSTMStep(RecStep):
    out_features: int = config()
    cell_size: int = config()
    dtype: DTypeLike | None = config()
    weight_ic: Array = parameter()
    weight_hc: Array = parameter()
    weight_ch: Array | None = parameter()
    bias_ic: Array | None = parameter()
    bias_hc: Array | None = parameter()
    state_h: Array = buffer()
    state_c: Array = buffer()

    def __init__(
        self,
        out_features: int,
        cell_size: int | None = None,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        if not with_bias:
            self.assign_(bias_ic=None, bias_hc=None)
        if cell_size is None:
            self.assign_(weight_ch=None)
            cell_size = out_features
        elif cell_size < out_features:
            raise ValueError("out_features shouldn't be larger than cell_size")
        self.assign_(
            out_features=out_features, cell_size=cell_size, dtype=dtype
        )
        super().__init__()

    @property
    def reset_rngkey_count(self) -> int:
        c = 2
        if self.bias_ic is not None:
            c += 1
        if self.bias_hc is not None:
            c += 1
        if self.weight_ch is not None:
            c += 1
        return c

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        dtype = self.dtype
        cell_size = self.cell_size
        in_features, out_features = input_shape[-1], self.out_features
        output_shape = tuple(input_shape[:-1]) + (out_features,)
        bound = 1.0 / cell_size
        keys = list(maybe_split(rngkey, self.reset_rngkey_count))
        keys, k = keys[:-1], keys[-1]
        weight_ic = uniform(
            (4 * cell_size, in_features),
            rngkey=k,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )
        keys, k = keys[:-1], keys[-1]
        weight_hc = uniform(
            (4 * cell_size, out_features),
            rngkey=k,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )
        if self.bias_ic is None:
            bias_ic = None
        else:
            keys, k = keys[:-1], keys[-1]
            bias_ic = uniform(
                (4 * cell_size,),
                rngkey=k,
                minval=-bound,
                maxval=bound,
                dtype=dtype,
            )
        if self.bias_hc is None:
            bias_hc = None
        else:
            keys, k = keys[:-1], keys[-1]
            bias_hc = uniform(
                (4 * cell_size,),
                rngkey=k,
                minval=-bound,
                maxval=bound,
                dtype=dtype,
            )
        if self.weight_ch is None:
            weight_ch = None
        else:
            keys, k = keys[:-1], keys[-1]
            weight_ch = uniform(
                (out_features, cell_size),
                rngkey=k,
                minval=-bound,
                maxval=bound,
                dtype=dtype,
            )
        return (
            new(
                self,
                weight_ic=weight_ic,
                weight_hc=weight_hc,
                weight_ch=weight_ch,
                bias_ic=bias_ic,
                bias_hc=bias_hc,
            ),
            output_shape,
        )

    @property
    def reset_state_rngkey_count(self) -> int:
        return 0

    def reset_recurrent_state(
        self: Self, batch_shape: Sequence[int], _=None
    ) -> Self:
        _assert_no_key(_)
        dtype = self.dtype
        state_h = zeros([*batch_shape, self.out_features], dtype=dtype)
        state_c = zeros([*batch_shape, self.cell_size], dtype=dtype)
        return new(
            self,
            state_h=state_h,
            state_c=state_c,
            recurrent_state_initialized=True,
        )

    def unset_recurrent_state(self: Self) -> Self:
        return new(
            self,
            state_h=make_default_array(),
            state_c=make_default_array(),
            recurrent_state_initialized=False,
        )

    @property
    def recurrent_state(self) -> dict[str, Array]:
        return {"state_h": self.state_h, "state_c": self.state_c}

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        pre_gate = dense(x, self.weight_ic, self.bias_ic) + dense(
            self.state_h, self.weight_hc, self.bias_hc
        )
        i, val_in, f, o = jnp.split(pre_gate, 4, axis=-1)
        state_c = sigmoid(f) * self.state_c + sigmoid(i) * val_in
        state_h = sigmoid(o) * tanh(state_c)
        if self.weight_ch is not None:
            state_h = self.weight_ch * state_h
        return self.update(state_h=state_h, state_c=state_c), state_h


class GRUStep(RecStep):
    out_features: int = config()
    dtype: DTypeLike | None = config()
    weight_ih: Array = parameter()
    weight_hh: Array = parameter()
    bias: Array = parameter()
    state_h: Array = buffer()

    def __init__(
        self,
        out_features: int,
        with_bias: bool = True,
        dtype: DTypeLike | None = None,
    ):
        if not with_bias:
            self.assign_(bias=None)
        self.assign_(
            out_features=out_features,
            dtype=dtype,
        )
        super().__init__()

    @property
    def reset_rngkey_count(self) -> int:
        c = 2
        if self.bias is not None:
            c += 1
        return c

    def reset(
        self: Self, input_shape: Sequence[int], rngkey: KeyArg = None
    ) -> tuple[Self, tuple[int, ...]]:
        dtype = self.dtype
        in_features, out_features = input_shape[-1], self.out_features
        output_shape = tuple(input_shape[:-1]) + (out_features,)
        bound = 1.0 / out_features
        keys = list(maybe_split(rngkey, self.reset_rngkey_count))
        keys, k = keys[:-1], keys[-1]
        weight_ih = uniform(
            (3 * out_features, in_features),
            rngkey=k,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )
        keys, k = keys[:-1], keys[-1]
        weight_hh = uniform(
            (3 * out_features, out_features),
            rngkey=k,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )
        if self.bias is None:
            bias = None
        else:
            keys, k = keys[:-1], keys[-1]
            bias = uniform(
                (3 * out_features,),
                rngkey=k,
                minval=-bound,
                maxval=bound,
                dtype=dtype,
            )
        return (
            new(
                self,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
            ),
            output_shape,
        )

    @property
    def reset_state_rngkey_count(self) -> int:
        return 0

    def reset_recurrent_state(
        self: Self, batch_shape: Sequence[int], _=None
    ) -> Self:
        _assert_no_key(_)
        dtype = self.dtype
        state_h = zeros([*batch_shape, self.out_features], dtype=dtype)
        return new(self, state_h=state_h, recurrent_state_initialized=True)

    def unset_recurrent_state(self: Self) -> Self:
        return new(
            self,
            state_h=make_default_array(),
            recurrent_state_initialized=False,
        )

    @property
    def recurrent_state(self) -> dict[str, Array]:
        return {"state_h": self.state_h}

    def forward(self: Self, x: ArrayLike, _=None) -> tuple[Self, Array]:
        _assert_no_key(_)
        out_features = self.out_features
        h = self.state_h
        x_pregate, x_val = jnp.split(
            dense(x, self.weight_ih, self.bias), [2 * out_features], axis=-1
        )
        w_hg, w_hv = jnp.split(self.weight_hh, [2 * out_features], axis=0)
        gr, gz = jnp.split(sigmoid(x_pregate + dense(h, w_hg)), 2, axis=-1)
        new_val = tanh(x_val + dense(gr * h, w_hv))
        h = gz * h + (1 - gz) * new_val
        return self.update(state_h=h), h


class Recurrent(Module):
    step_module: RecStep = child()
    axis: int = config()
    feature_dims: int = config()

    def __init__(
        self, step_module: RecStep, axis: int = -2, feature_dims: int = 1
    ):
        assert feature_dims > 0, "feature_dims should be None or larger than 1"
        self.assign_(
            step_module=step_module, axis=axis, feature_dims=feature_dims
        )

    @property
    def reset_rngkey_count(self) -> int:
        m = self.step_module
        return m.reset_rngkey_count

    def reset(
        self: Self,
        input_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        axis = self.axis
        if axis < 0:  # need positive index so pop/insert works as intended
            axis = axis + len(input_shape)
        inshape = list(input_shape)
        seqlen = inshape.pop(axis)
        step_module, outshape = self.step_module.reset(inshape, rngkey)
        outshapelist = list(outshape)
        outshapelist.insert(axis, seqlen)
        return new(self, step_module=step_module), tuple(outshapelist)

    @property
    def forward_rngkey_count(self) -> int:
        c = 0 if self.step_module.forward_rngkey_count == 0 else 1
        return self.step_module.reset_state_rngkey_count + c

    def forward(
        self: Self,
        input_values: ArrayLike,
        rngkey: KeyArg = None,
    ) -> tuple[Self, Array]:
        def _scan_step(carry, val):
            ckey, recstep = carry
            if ckey is None:
                ckey, newkeys = None, ()
            else:
                newkeys = jrsplit(ckey, recstep.forward_rngkey_count + 1)
                ckey, newkeys = newkeys[0], newkeys[1:]
            recstep, val = recstep.forward(val, newkeys)
            return (ckey, recstep), val

        step_module = self.step_module
        vals = jnp.moveaxis(input_values, self.axis, 0)
        batch_shape = vals[0].shape[: -self.feature_dims]
        if self.forward_rngkey_count == 0:
            initkeys, fwdkeys = None, None
        else:
            keys = maybe_split(rngkey, self.forward_rngkey_count)[0]
            initkeycount = self.step_module.reset_state_rngkey_count
            initkeys, fwdkeys = keys[:initkeycount], keys[initkeycount:]
        step_module = step_module.reset_recurrent_state(batch_shape, initkeys)
        if fwdkeys:
            key = fwdkeys[0]
        else:
            key = None
        (_, step_module), vals = lax_scan(_scan_step, (key, step_module), vals)
        vals = jnp.moveaxis(vals, 0, self.axis)
        step_module = step_module.unset_recurrent_state()
        return new(self, step_module=step_module), vals


class BiRecurrent(Module):
    left_step_module: RecStep = child()
    right_step_module: RecStep = child()
    axis: int = config()
    channel_axis: int = config()
    feature_dims: int = config()

    def __init__(
        self,
        left_step_module: RecStep,
        right_step_module: RecStep,
        axis: int = -2,
        channel_axis: int = -1,
        feature_dims: int = 1,
    ):
        assert feature_dims > 0, "feature_dims should be None or larger than 1"
        _assert_negative_axis(channel_axis)
        self.assign_(
            left_step_module=left_step_module,
            right_step_module=right_step_module,
            axis=axis,
            channel_axis=channel_axis,
            feature_dims=feature_dims,
        )

    @property
    def reset_rngkey_count(self) -> int:
        lm, rm = self.left_step_module, self.right_step_module
        return lm.reset_rngkey_count + rm.reset_rngkey_count

    @staticmethod
    def _merge_shape(
        shp1: Sequence[int], shp2: Sequence[int], axis: int
    ) -> list[int]:
        if len(shp1) != len(shp2):
            raise ValueError(f"dimension mismatch between {shp1} and {shp2}")
        for i in range(len(shp1)):
            if shp1[i] != shp2[i] and i != axis:
                raise ValueError(f"size mismatch between {shp1} and {shp2}")
        outshp = list(shp1)
        outshp[axis] += shp2[axis]
        return outshp

    def reset(
        self: Self,
        input_shape: Sequence[int],
        rngkey: KeyArg = None,
    ) -> tuple[Self, tuple[int, ...]]:
        axis = self.axis
        if axis < 0:  # need positive index so pop/insert works as intended
            axis = axis + len(input_shape)
        left_step = self.left_step_module
        right_step = self.right_step_module

        nkey_total = self.reset_rngkey_count
        nkey_left = left_step.reset_rngkey_count
        if nkey_total == 0:
            lkey, rkey = None, None
        elif nkey_left == 0:
            lkey, rkey = None, rngkey
        elif nkey_total == nkey_left:
            lkey, rkey = rngkey, None
        else:
            keys = maybe_split(rngkey, nkey_total)
            lkey, rkey = keys[:nkey_left], keys[nkey_left:]
        inshape = list(input_shape)
        seqlen = inshape.pop(axis)
        left_step, left_outshape = left_step.reset(inshape, lkey)
        right_step, right_outshape = right_step.reset(inshape, rkey)
        outshapelist = self._merge_shape(
            left_outshape, right_outshape, self.channel_axis
        )
        outshapelist.insert(axis, seqlen)
        return new(
            self, left_step_module=left_step, right_step_module=right_step
        ), tuple(outshapelist)

    @property
    def forward_rngkey_count(self) -> int:
        lstep = self.left_step_module
        rstep = self.right_step_module
        lrc = lstep.reset_state_rngkey_count
        rrc = rstep.reset_state_rngkey_count
        lrfc = lstep.forward_rngkey_count + rstep.forward_rngkey_count
        c = 0 if lrfc == 0 else 1
        return c + lrc + rrc

    def forward(
        self: Self,
        input_values: ArrayLike,
        rngkey: KeyArg = None,
    ) -> tuple[Self, Array]:

        def _scan_step(carry, val):
            ckey, recstep = carry
            if ckey is None:
                ckey, newkeys = None, ()
            else:
                newkeys = jrsplit(ckey, recstep.forward_rngkey_count + 1)
                ckey, newkeys = newkeys[0], newkeys[1:]
            recstep, val = recstep.forward(val, newkeys)
            return (ckey, recstep), val

        saxis = self.axis
        lstep = self.left_step_module
        rstep = self.right_step_module
        vals = jnp.moveaxis(input_values, saxis, 0)
        batch_shape = vals[0].shape[: -self.feature_dims]
        if self.forward_rngkey_count == 0:
            linitkeys, rinitkeys, fwdkeys = None, None, None
        else:
            keys = maybe_split(rngkey, self.forward_rngkey_count)[0]
            lc = lstep.reset_state_rngkey_count
            rc = rstep.reset_state_rngkey_count
            linitkeys, rinitkeys = keys[:lc], keys[lc : lc + rc]
            fwdkeys = keys[lc + rc :]
        lstep = lstep.reset_recurrent_state(batch_shape, linitkeys)
        rstep = rstep.reset_recurrent_state(batch_shape, rinitkeys)
        if fwdkeys:
            key = fwdkeys[0]
        else:
            key = None

        (key, lstep), lval = lax_scan(_scan_step, (key, lstep), vals)
        (key, rstep), rval = lax_scan(
            _scan_step, (key, rstep), jnp.flip(vals, 0)
        )
        lval = jnp.moveaxis(lval, 0, saxis)
        rval = jnp.moveaxis(jnp.flip(rval, 0), 0, saxis)
        vals = jnp.concatenate([lval, rval], axis=self.channel_axis)
        lstep = lstep.unset_recurrent_state()
        rstep = rstep.unset_recurrent_state()
        return new(self, left_step_module=lstep, right_step_module=rstep), vals
