import math
import jax
import ojnn
import unittest


class TestModule(unittest.TestCase):
    def test_abstract(self):
        with self.assertRaises(TypeError):
            _ = ojnn.Module()

        with self.assertRaises(NotImplementedError):

            class NoForward(ojnn.Module):
                def forward(self, x, _=()):
                    return super().forward(x, _)

            model = NoForward()
            _ = model.forward(jax.numpy.ones(3))

    def test_default_reset(self):
        class MyIdentity(ojnn.Module):
            def forward(self, x, _=None):
                return self, x

        model = MyIdentity()
        self.assertEqual(model.reset_rngkey_count, 0)
        model, out_shape = model.reset([3])
        self.assertEqual(out_shape, (3,))

    def test_mode(self):
        layer = ojnn.Dense(3)
        self.assertEqual(layer.mode, "train")
        layer = layer.update_mode("eval")
        self.assertEqual(layer.mode, "eval")
        layer = layer.update_mode("train", recursive=False)
        self.assertEqual(layer.mode, "train")
        with self.assertRaises(ValueError):
            _ = layer.update_mode("test")

    def test_param_update(self):
        layer = ojnn.Dense(3)
        with self.assertRaises(ValueError):
            _ = layer.update(weight=[1.0, 2.0])

    def test_child_update(self):
        class MyModel(ojnn.Module):
            child: ojnn.Module = ojnn.child()

            def __init__(self):
                self.assign_(child=ojnn.ReLU())

            def forward(self, x):
                return self, self.child.forward(x)

        model = MyModel()
        with self.assertRaises(ValueError):
            _ = model.update(child=42)

    def test_no_const_update(self):
        class ConstScale(ojnn.Module):
            scale: float = ojnn.const(default=1.0)

            def forward(self, x):
                return self, x * self.scale

        model = ConstScale()
        with self.assertRaises(ValueError):
            _ = model.update(scale=2.0)

    def test_update_mode_recursive(self):
        class DoubleDense(ojnn.Module):
            layer1: ojnn.Dense
            layer2: ojnn.Dense

            def __init__(self, out_dim):
                self.assign_(
                    layer1=ojnn.Dense(out_dim),
                    layer2=ojnn.Dense(out_dim, with_bias=False),
                )

            @property
            def reset_rngkey_count(self):
                return self.layer1.reset_rngkey_count

            def reset(self, inshape, rngkey=None):
                newl1, outshape = self.layer1.reset(inshape, rngkey)
                newl2, _ = self.layer2.reset(inshape, rngkey)
                return self.update(layer1=newl1, layer2=newl2), outshape

            def forward(self, x, _=None):
                return self, self.layer1(x) + self.layer2(x)

        model = DoubleDense(3)
        model = model.update_mode("eval", recursive=False)
        self.assertEqual(model.mode, "eval")
        self.assertEqual(model.layer1.mode, "train")
        self.assertEqual(model.layer2.mode, "train")
        model = model.update_mode("eval", recursive=True)
        self.assertEqual(model.mode, "eval")
        self.assertEqual(model.layer1.mode, "eval")
        self.assertEqual(model.layer2.mode, "eval")
        model = model.update_mode("train")
        self.assertEqual(model.mode, "train")
        self.assertEqual(model.layer1.mode, "train")
        self.assertEqual(model.layer2.mode, "train")

    def test_call(self):
        class ScaleAndNoise(ojnn.Module):
            scale: jax.Array = ojnn.parameter()

            def reset(self, inshape, _=None):
                outshape = tuple(inshape)
                return self.update(scale=jax.numpy.ones(())), outshape

            @property
            def forward_rngkey_count(self):
                return 1

            def forward(self, x, rngkey=None):
                rngkey = ojnn.maybe_split(rngkey, 1)[0]
                return self, x * self.scale + jax.random.normal(
                    rngkey, x.shape
                )

        model = ScaleAndNoise()
        key = jax.random.key(42)
        val = jax.numpy.arange(3)
        # cannot call before .init is called
        with self.assertRaises(RuntimeError):
            _ = model(val, key)

        model = model.init(val.shape, ())
        try:
            _ = model(val, key)
            _ = model(
                val,
                key,
                parameters=({"scale": jax.numpy.full((), fill_value=2)}, {}),
            )
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))


class TestModuleComposition(unittest.TestCase):
    def test_compose(self):
        my_network = ojnn.Sequential(
            ojnn.MapReduce(
                ojnn.Dense(3),
                ojnn.Dense(3),
                reduce_fn=math.prod,
            ),
            ojnn.MapConcat(
                ojnn.Dense(3),
                ojnn.Dense(2),
                axis=-1,
            ),
        )
        self.assertEqual(my_network.reset_rngkey_count, 4)
        x = jax.numpy.arange(3, dtype=None)
        my_network = my_network.init(x.shape, jax.random.key(42))
        my_network, out = my_network(x)
        self.assertTrue(
            jax.numpy.allclose(out, jax.numpy.asarray([0, 1, 4, 0, 1]))
        )

    def test_compose_named(self):
        my_network = ojnn.NamedSequential(
            block1=ojnn.NamedMapReduce(
                layer1=ojnn.Dense(3),
                layer2=ojnn.Dense(3),
                reduce_fn=math.prod,
            ),
            block2=ojnn.NamedMapConcat(
                layer1=ojnn.Dense(3),
                layer2=ojnn.Dense(2),
                axis=-1,
            ),
        )
        self.assertEqual(my_network.reset_rngkey_count, 4)
        x = jax.numpy.arange(3, dtype=None)
        my_network = my_network.init(x.shape, jax.random.key(42))
        my_network, out = my_network(x)
        self.assertTrue(
            jax.numpy.allclose(out, jax.numpy.asarray([0, 1, 4, 0, 1]))
        )

    def test_compose_seq(self):
        class AddNoise(ojnn.Module):
            @property
            def forward_rngkey_count(self):
                return 1

            def forward(self, x, key=None):
                key = ojnn.maybe_split(key, 1)[0]
                return self, x + jax.random.normal(key, x.shape)

        model = ojnn.NamedSequential(
            l1=ojnn.Sequential(AddNoise(), AddNoise()),
            l2=ojnn.Sequential(AddNoise(), ojnn.Dense(3)),
            l3=ojnn.Sequential(ojnn.Dense(3), ojnn.Dense(3)),
        )
        self.assertEqual(model.reset_rngkey_count, 3)
        self.assertEqual(model.forward_rngkey_count, 3)
        try:
            model = model.init([2, 3], jax.random.split(jax.random.key(42), 3))
            _ = model(
                jax.numpy.ones([2, 3]), jax.random.split(jax.random.key(0), 3)
            )
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))

        model = ojnn.Sequential(
            ojnn.NamedSequential(l1=AddNoise(), l2=AddNoise()),
            ojnn.NamedSequential(l1=AddNoise(), l2=ojnn.Dense(3)),
            ojnn.NamedSequential(l1=ojnn.Dense(3), l2=ojnn.Dense(3)),
        )
        self.assertEqual(model.reset_rngkey_count, 3)
        self.assertEqual(model.forward_rngkey_count, 3)
        try:
            model = model.init([2, 3], jax.random.split(jax.random.key(42), 3))
            _ = model(
                jax.numpy.ones([2, 3]), jax.random.split(jax.random.key(0), 3)
            )
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))

    def test_compose_map(self):
        class AddNoise(ojnn.Module):
            @property
            def forward_rngkey_count(self):
                return 1

            def forward(self, x, key=None):
                key = ojnn.maybe_split(key, 1)[0]
                return self, x + jax.random.normal(key, x.shape)

        model = ojnn.MapReduce(
            ojnn.NamedMapReduce(l1=AddNoise(), l2=AddNoise()),
            ojnn.NamedMapReduce(l1=AddNoise(), l2=ojnn.Dense(3)),
            ojnn.NamedMapReduce(l1=ojnn.Dense(3), l2=ojnn.Dense(3)),
        )
        self.assertEqual(model.reset_rngkey_count, 3)
        self.assertEqual(model.forward_rngkey_count, 3)
        try:
            model = model.init([2, 3], jax.random.split(jax.random.key(42), 3))
            _ = model(
                jax.numpy.ones([2, 3]), jax.random.split(jax.random.key(0), 3)
            )
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))

        model = ojnn.NamedMapReduce(
            l1=ojnn.MapReduce(AddNoise(), AddNoise()),
            l2=ojnn.MapReduce(AddNoise(), ojnn.Dense(3)),
            l3=ojnn.MapReduce(ojnn.Dense(3), ojnn.Dense(3)),
        )
        self.assertEqual(model.reset_rngkey_count, 3)
        self.assertEqual(model.forward_rngkey_count, 3)
        try:
            model = model.init([2, 3], jax.random.split(jax.random.key(42), 3))
            _ = model(
                jax.numpy.ones([2, 3]), jax.random.split(jax.random.key(0), 3)
            )
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))

    def test_bad_reduce(self):
        with self.assertRaises(ValueError):
            model = ojnn.MapReduce(
                ojnn.Dense(3),
                ojnn.Dense(4),
            )
            _ = model.init((2,), jax.random.split(jax.random.key(42), 2))

    def test_mapconcat(self):
        with self.assertRaises(ValueError):
            model = ojnn.MapConcat(
                ojnn.Dense(3),
                ojnn.Sequential(ojnn.Conv1d(4, 3), ojnn.Flatten()),
                axis=-1,
            )
            _ = model.init((3, 32), jax.random.split(jax.random.key(42), 2))

        with self.assertRaises(ValueError):
            model = ojnn.MapConcat(
                ojnn.Dense(3),
                ojnn.Dense(4),
                axis=1,
            )
            _ = model.init((2,), jax.random.split(jax.random.key(42), 2))

        with self.assertRaises(ValueError):
            model = ojnn.MapConcat(
                ojnn.Conv1d(4, 3, padding=2),
                ojnn.Conv1d(4, 3),
                axis=0,
            )
            _ = model.init((3, 32), jax.random.split(jax.random.key(42), 2))

        try:
            model = ojnn.MapConcat(
                ojnn.Conv1d(2, 3),
                ojnn.Conv1d(4, 3),
                axis=-2,
            )
            _ = model.init((3, 32), jax.random.split(jax.random.key(42), 2))
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))


if __name__ == "__main__":
    unittest.main()
