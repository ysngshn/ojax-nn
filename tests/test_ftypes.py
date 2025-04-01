import unittest
import ojnn


class TestFtypes(unittest.TestCase):
    def test_bad_defaults(self):
        with self.assertRaises(ValueError):
            _ = ojnn.config(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojnn.state(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojnn.const(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojnn.ftypes._internal_state(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojnn.child(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojnn.parameter(default=0.0, default_factory=float)
        with self.assertRaises(ValueError):
            _ = ojnn.buffer(default=0.0, default_factory=float)
        with self.assertRaises(ValueError):
            _ = ojnn.schedulable(default=0.0, default_factory=float)
        with self.assertRaises(ValueError):
            _ = ojnn.external(default=(), default_factory=tuple)


if __name__ == "__main__":
    unittest.main()
