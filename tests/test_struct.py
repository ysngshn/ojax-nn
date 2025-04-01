from typing import ClassVar
import unittest
import dataclasses
import jax
import ojnn


class TestStruct(unittest.TestCase):
    def test_bad_admissible_field_type(self):
        with self.subTest("no StructField"), self.assertRaises(ValueError):

            class BadStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (
                    ojnn.Config,
                    ojnn.StructField,
                )
                data: jax.Array = ojnn.buffer()

            _ = BadStruct()

        with self.subTest("no Field"), self.assertRaises(ValueError):

            class BadStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (
                    ojnn.Config,
                    dataclasses.Field,
                )
                data: jax.Array = ojnn.buffer()

            _ = BadStruct()

        with self.subTest("no helper fn"), self.assertRaises(ValueError):

            class BadStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (ojnn.config,)
                data: jax.Array = ojnn.buffer()

            _ = BadStruct()

        with self.subTest("no type"), self.assertRaises(ValueError):

            class BadStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (
                    ojnn.Config,
                    int,
                )
                data: jax.Array = ojnn.buffer()

            _ = BadStruct()

    def test_inadmissible_field_type(self):
        with self.assertRaises(ValueError):

            class BadStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (ojnn.Config,)
                data: jax.Array = ojnn.buffer()

            _ = BadStruct()

    def test_admissible_field_type(self):
        try:

            class GoodStruct(ojnn.Struct):
                admissible_field_types: ClassVar = (ojnn.Config,)
                size: int = ojnn.config()
                dim: int

                def __init__(self, size: int, dim: int):
                    self.assign_(size=size, dim=dim)

            gs = GoodStruct(2, 1)

        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))
        self.assertTrue(len(gs.fields()) == 2)
        self.assertTrue(len(gs.fields(infer=False)) == 1)

    def test_field_update(self):
        class MyStruct(ojnn.Struct):
            admissible_field_types: ClassVar = (ojnn.Config, ojnn.Buffer)
            config: int = ojnn.config()
            buffer: float = ojnn.buffer()

            def __init__(self, config: int):
                self.assign_(config=config)

            def infer_field_type(self, f):
                return ojnn.Config

        my_struct = MyStruct(1)
        with self.assertRaises(ValueError):
            _ = my_struct.update(config=42)
        with self.assertRaises(ValueError):
            _ = my_struct.update(mistyped_name=42)
        try:
            new_struct = my_struct.update(buffer=42)
        except Exception as e:
            self.fail(getattr(e, "message", repr(e)))
        self.assertEqual(new_struct.buffer, 42)

    def test_new(self):
        class MyStruct(ojnn.Struct):
            admissible_field_types: ClassVar = (ojnn.Config,)
            value: int = ojnn.config()

            def __init__(self, value: int):
                self.assign_(value=value)

            def infer_field_type(self, f):
                return ojnn.Config

        my_struct = MyStruct(42)
        my_other_struct = ojnn.new(my_struct)
        self.assertTrue(my_struct is not my_other_struct)
        self.assertTrue(my_struct.value == my_other_struct.value)


if __name__ == "__main__":
    unittest.main()
