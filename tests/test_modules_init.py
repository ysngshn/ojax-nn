import jax
import ojnn
import unittest


class TestDeltaIdOrthogonalInit(unittest.TestCase):
    def test_matrix_init(self):
        # square
        t = ojnn.init.identity_or_orthogonal((3, 3), jax.random.key(42))
        self.assertTrue(jax.numpy.allclose(t, jax.numpy.eye(3)))
        # c_out < c_in
        t = ojnn.init.identity_or_orthogonal((2, 3), jax.random.key(42))
        self.assertTrue(jax.numpy.allclose(t, jax.numpy.eye(2, 3)))
        # c_out > c_in
        t = ojnn.init.identity_or_orthogonal((3, 2), jax.random.key(42))
        self.assertFalse(jax.numpy.allclose(t, jax.numpy.eye(3, 2), atol=1e-6))
        self.assertTrue(
            jax.numpy.allclose(t.T @ t, jax.numpy.eye(2), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
