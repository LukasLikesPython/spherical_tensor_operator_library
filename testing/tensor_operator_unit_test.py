import unittest
from tensor_operator import TensorOperator, TensorOperatorComposite

class TestTensorOperator(unittest.TestCase):
    def create_building_block(self):
        basic_block = TensorOperator(rank=1, representation='q', space='relative')
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
