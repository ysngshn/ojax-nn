from collections.abc import Callable
from typing_extensions import Self
from jax import Array
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from jax.nn import relu
import ojnn
from ojnn.modules.utils import _assert_negative_axis
from indrnn import indrnn


class IndRNN(ojnn.Module):
    seq_dim: int
    channel_dim: int
    activation: Callable[[Array], Array]
    unroll: bool | int
    weight: Array = ojnn.parameter()

    def __init__(
        self,
        seq_dim: int = -2,
        channel_dim: int = -1,
        activation: Callable[[Array], Array] = relu,
        unroll: bool | int = False,
    ):
        super().__init__()
        _assert_negative_axis(seq_dim)
        _assert_negative_axis(channel_dim)
        self.assign_(
            seq_dim=seq_dim,
            channel_dim=channel_dim,
            activation=activation,
            unroll=unroll,
        )

    @property
    def reset_rngkey_count(self) -> int:
        return 0

    def reset(self, input_shape, rngkey=None):
        weight = ojnn.zeros([input_shape[self.channel_dim]])
        return ojnn.new(self, weight=weight), tuple(input_shape)

    def forward(self, x, _=None):
        out = indrnn(
            x,
            self.weight,
            activation=self.activation,
            seq_dim=self.seq_dim,
            channel_dim=self.channel_dim,
            unroll=self.unroll,
        )
        return self, out


class GloVeEmbed(ojnn.Embed):
    gensim_glove_model_name: str

    def __init__(self, gensim_glove_model_name: str, vocab_size: int = 400001):
        import gensim.downloader as api

        if gensim_glove_model_name not in api.info()["models"]:
            raise ValueError(
                f"unknown model {gensim_glove_model_name}, must be one of:\n"
                + "\n".join([f"- {n}" for n in api.info()["models"]])
            )
        if vocab_size > 400001:
            raise ValueError("pretrained gloVe have max vocab size 400001")
        features = int(gensim_glove_model_name.split("-")[-1])
        super().__init__(vocab_size, features, "float32")
        self.assign_(gensim_glove_model_name=gensim_glove_model_name)

    def reset(self, input_shape, rngkey=None):
        import gensim.downloader as api

        model = api.load(self.gensim_glove_model_name)
        newself, outshape = super().reset(input_shape, rngkey)
        w = newself.weight
        w = w.at[0].set(0)
        w = w.at[1:].set(model.vectors[: len(w) - 1])
        return ojnn.new(newself, weight=w), outshape


class PosEmbed(ojnn.Embed):

    def reset(self, input_shape, rngkey=None):
        return super().reset(input_shape, rngkey)

    def forward(
        self: Self, x: ArrayLike,  _=None
    ) -> tuple[Self, Array]:
        pos = jnp.arange(x.shape[-1])
        newself, out = super().forward(pos)
        return newself, jnp.broadcast_to(
            out, list(x.shape[:-1])+list(out.shape)
        )


# conv 2d with Kaiming He initialization
class HeInitConv1d(ojnn.Conv1d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        with_bias: bool = True,
    ):
        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            with_bias=with_bias,
        )

    def reset(self, input_shape, rngkey=None):
        newself, out_shapes = super().reset(input_shape, rngkey)
        w = newself.weight
        return (
            newself.update(
                weight=ojnn.he_normal(
                    w.shape, ojnn.maybe_split(rngkey, 1)[0], w.dtype
                )
            ),
            out_shapes,
        )


class HeInitDense(ojnn.Dense):

    def __init__(
        self,
        out_features: int,
        with_bias: bool = True,
    ):
        super().__init__(out_features, with_bias=with_bias)

    def reset(self, input_shape, rngkey=None):
        newself, out_shapes = super().reset(input_shape, rngkey)
        w = newself.weight
        return (
            newself.update(
                weight=ojnn.he_normal(
                    w.shape, ojnn.maybe_split(rngkey, 1)[0], w.dtype
                )
            ),
            out_shapes,
        )


class TC2CT(ojnn.Module):
    def reset(self, input_shape, rngkey=None):
        assert len(input_shape) >= 2
        output_shape = (*input_shape[:-2], input_shape[-1], input_shape[-2])
        return self, output_shape

    def forward(self, x: Array, _=None):
        return self, x.swapaxes(-2, -1)


class TakeLast(ojnn.Module):
    axis: int

    def __init__(self, axis: int):
        self.assign_(axis=axis)

    def reset(self, input_shape, rngkey=None):
        axis = self.axis
        output_shape = (*input_shape[:axis], *input_shape[axis + 1 :])
        return self, output_shape

    def forward(self, x: Array, _=None):
        return self, jnp.take(x, -1, axis=self.axis)


# cf. "1D CNN over time dimension"
class TextConvNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7
        conv_kernel = 3
        conv_stride = 1

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)
            # embed_layer = embed_layer.config_trainable(weight=False)

        conv_layers = []

        for _ in range(depth - 2):
            conv_layers.extend(
                [
                    HeInitConv1d(hidden_dim, conv_kernel, stride=conv_stride),
                    ojnn.LayerNorm(channel_dim=-2),
                    ojnn.ReLU(),
                ]
            )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            TC2CT(),
            # BCT
            *conv_layers,
            ojnn.GlobalPool("max", (-1,), keepdims=False),
            # BC
            HeInitDense(2),
            # B2
        )


class RNNNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        rec_layers = ojnn.SequentialRecStep(
            *[ojnn.RNNStep(hidden_dim) for _ in range(depth - 2)]
        )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            ojnn.Recurrent(rec_layers),
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class LSTMNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        rec_layers = ojnn.SequentialRecStep(
            *[ojnn.LSTMStep(hidden_dim) for _ in range(depth - 2)]
        )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            ojnn.Recurrent(rec_layers),
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class GRUNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        rec_layers = ojnn.SequentialRecStep(
            *[ojnn.GRUStep(hidden_dim) for _ in range(depth - 2)]
        )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            ojnn.Recurrent(rec_layers),
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class IndRNNNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        rec_layers = []
        for _ in range(depth - 2):
            rec_layers.extend([
                ojnn.Dense(hidden_dim),
                ojnn.LayerNorm(),
                IndRNN(activation=ojnn.relu),
            ])

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            *rec_layers,
            ojnn.GlobalPool("max", (-2,), keepdims=False),  # 0.9076
            # ojnn.GlobalPool("avg", (-2,), keepdims=False),  # blown up
            # BC
            HeInitDense(2),
            # B2
        )


class BiRNNNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        l_layers = ojnn.SequentialRecStep(
            *[ojnn.RNNStep(hidden_dim) for _ in range(depth - 2)]
        )
        r_layers = ojnn.SequentialRecStep(
            *[ojnn.RNNStep(hidden_dim) for _ in range(depth - 2)]
        )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            ojnn.BiRecurrent(l_layers, r_layers),
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class BiLSTMNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 2

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        l_layers = ojnn.SequentialRecStep(
            *[ojnn.LSTMStep(hidden_dim) for _ in range(depth - 2)]
        )
        r_layers = ojnn.SequentialRecStep(
            *[ojnn.LSTMStep(hidden_dim) for _ in range(depth - 2)]
        )

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            ojnn.BiRecurrent(l_layers, r_layers),
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class BiGRUNet(ojnn.Sequential):
    def __init__(
        self,
        depth: int = 4,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):
        assert depth >= 3

        embed_dim = 300
        hidden_dim = 128
        drop_rate = 0.7

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, embed_dim)

        # acc:
        l_layers = ojnn.SequentialRecStep(
            *[ojnn.GRUStep(hidden_dim) for _ in range(depth - 3)]
        )
        r_layers = ojnn.SequentialRecStep(
            *[ojnn.GRUStep(hidden_dim) for _ in range(depth - 3)]
        )
        rec_layers = [
            ojnn.BiRecurrent(l_layers, r_layers),
            ojnn.Recurrent(ojnn.GRUStep(hidden_dim)),
        ]
        # # acc:
        # rec_layers = [
        #     ojnn.BiRecurrent(
        #         ojnn.GRUStep(hidden_dim), ojnn.GRUStep(hidden_dim)
        #     ) for _ in range(depth - 3)] + [ojnn.GRUStep(hidden_dim)]

        super().__init__(
            # BT
            embed_layer,
            # BTC
            ojnn.Dropout1d(drop_rate),
            *rec_layers,
            TakeLast(-2),
            # BC
            HeInitDense(2),
            # B2
        )


class TransformerNet(ojnn.Sequential):
    def __init__(
        self,
        n_block: int,
        max_len: int = 500,
        vocab_size: int = 400001,
        pretrained_glove: bool = False,
    ):

        token_embed_dim = 300
        # why 30: # https://james-simon.github.io/blog/gpt2-positional-encs/
        pos_embed_dim = 30
        embed_dim = token_embed_dim + pos_embed_dim
        nhead: int = 10
        embed_drop_rate = 0.7
        block_drop_rate = 0.1

        if pretrained_glove:
            embed_layer = GloVeEmbed("glove-wiki-gigaword-300", vocab_size)
            embed_layer = embed_layer.config_trainable(weight=False)
        else:
            embed_layer = ojnn.Embed(vocab_size, token_embed_dim)

        blocks = sum(
            (
                self._make_block(
                    embed_dim, nhead, block_drop_rate
                ) for _ in range(n_block)
            ),
            start=[]
        )

        super().__init__(
            ojnn.MapConcat(
                embed_layer,
                PosEmbed(max_len, pos_embed_dim, embed_layer.dtype),
                axis=-1,
            ),
            # BTC
            ojnn.Dropout1d(embed_drop_rate),
            *blocks,
            ojnn.LayerNorm(),
            ojnn.GlobalPool("avg", -2),
            # BC
            HeInitDense(2),
            # B2
        )

    @staticmethod
    def _make_block(
            out_features: int,
            nhead: int,
            p: float | None,
    ) -> list[ojnn.Module]:
        attn_block = ojnn.MapReduce(
            ojnn.Sequential(
                ojnn.LayerNorm(),
                ojnn.MultiHeadSelfAttention(out_features, nhead),
                ojnn.Identity() if p is None else ojnn.Dropout1d(p),
            ),
            ojnn.Identity(),
        )
        mlp_block = ojnn.MapReduce(
            ojnn.Sequential(
                ojnn.LayerNorm(),
                ojnn.Dense(4 * out_features),
                ojnn.GELU(),
                ojnn.Dense(out_features),
                ojnn.Identity() if p is None else ojnn.Dropout1d(p),
            ),
            ojnn.Identity(),
        )
        return [attn_block, mlp_block]
