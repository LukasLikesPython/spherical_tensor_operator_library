import unittest
from tensor_space import TensorSpace

class TestTensorSpace(unittest.TestCase):

    spin = TensorSpace('spin', 1)
    rel = TensorSpace('rel', 0)
    cm = TensorSpace('cm', 3)

    def test_creation_duplicate_error(self):
        with self.assertRaises(AttributeError):
            TensorSpace('rel', 1)

    def test_couple_spaces(self):
        self.assertEqual('{spin x rel}', (self.spin + self.rel).name)
        self.assertEqual('{rel x spin}', (self.rel + self.spin).name)
        self.assertEqual(0, (self.spin + self.rel).order)
        self.assertEqual('{{spin x rel} x cm}', (self.spin + self.rel + self.cm).name)

    def test_logical_eq(self):
        self.assertTrue(self.rel == TensorSpace('rel', 0))

    def test_logical_neq(self):
        self.assertFalse(self.rel == TensorSpace('rel2', 0))



if __name__ == '__main__':
    unittest.main()
