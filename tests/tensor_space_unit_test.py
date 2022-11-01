import unittest
import sys

sys.path.append('../src/stol/')

from tensor_space import TensorSpace, DuplicateSpaceError


class TestTensorSpace(unittest.TestCase):

    spin = TensorSpace("spin", 1)
    rel = TensorSpace("rel", 0)
    cm = TensorSpace("cm", 3)

    def test_creation_duplicate_error(self):
        with self.assertRaises(DuplicateSpaceError):
            TensorSpace("rel", 1)

    def test_couple_spaces(self):
        self.assertEqual("{spin x rel}", (self.spin + self.rel).name)
        self.assertEqual("{rel x spin}", (self.rel + self.spin).name)
        self.assertEqual(0, (self.spin + self.rel).order)
        self.assertEqual("{{spin x rel} x cm}", (self.spin + self.rel + self.cm).name)

    def test_logical_eq(self):
        self.assertTrue(self.rel == TensorSpace("rel", 0))

    def test_logical_neq(self):
        self.assertFalse(self.rel == TensorSpace("rel2", 0))

    def test_logical_gt(self):
        space_a = self.rel + self.spin
        self.assertTrue(self.rel > space_a)
        space_b = space_a + self.rel
        self.assertTrue(space_a > space_b)
        space_c = space_a + self.spin
        self.assertTrue(space_c > space_b)
        space_d = space_b + self.rel
        space_e = space_c + self.rel
        self.assertTrue(space_e > space_d)
        self.assertFalse(space_d > space_e)

    def test_iterator(self):
        space_a = self.rel + self.spin
        self.assertTrue(self.rel > space_a)
        space_b = self.rel + self.cm
        space_c = space_a + space_b
        space_d = space_a + self.spin
        self.assertEqual(["{rel x spin}", "rel", "spin", "{rel x cm}", "rel", "cm"], [x.name for x in space_c])
        self.assertEqual(["rel", "spin"], [x.name for x in space_a])
        self.assertEqual(["{rel x spin}", "rel", "spin", "spin"], [x.name for x in space_d])


if __name__ == "__main__":
    unittest.main()
