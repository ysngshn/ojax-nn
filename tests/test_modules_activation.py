import jax
import ojnn
import unittest


class TestActivations(unittest.TestCase):
    def test_activation_consistency(self):
        x = jax.random.normal(jax.random.key(42), (3, 2))
        activ_fns = {
            ojnn.relu: ojnn.ReLU,
            ojnn.relu6: ojnn.ReLU6,
            ojnn.sigmoid: ojnn.Sigmoid,
            ojnn.softplus: ojnn.Softplus,
            ojnn.sparse_plus: ojnn.SparsePlus,
            ojnn.sparse_sigmoid: ojnn.SparseSigmoid,
            ojnn.soft_sign: ojnn.SoftSign,
            ojnn.silu: ojnn.SiLU,
            ojnn.log_sigmoid: ojnn.LogSigmoid,
            ojnn.leaky_relu: ojnn.LeakyReLU,
            ojnn.hard_sigmoid: ojnn.HardSigmoid,
            ojnn.hard_silu: ojnn.HardSiLU,
            ojnn.hard_tanh: ojnn.HardTanh,
            ojnn.elu: ojnn.ELU,
            ojnn.celu: ojnn.CELU,
            ojnn.selu: ojnn.SELU,
            ojnn.gelu: ojnn.GELU,
            ojnn.glu: ojnn.GLU,
            ojnn.squareplus: ojnn.SquarePlus,
            ojnn.mish: ojnn.Mish,
            ojnn.tanh: ojnn.Tanh,
            ojnn.softmax: ojnn.Softmax,
            ojnn.log_softmax: ojnn.LogSoftmax,
        }

        for f, c in activ_fns.items():
            with self.subTest(f"{f.__name__}"):
                self.assertTrue(jax.numpy.allclose(f(x), c()(x)[1]))


if __name__ == "__main__":
    unittest.main()
