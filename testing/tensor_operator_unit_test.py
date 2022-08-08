import unittest
from tensor_operator import TensorOperator, TensorOperatorComposite
import win_unicode_console

class TestTensorOperator(unittest.TestCase):
    def test_building_block(self):
        basic_block = TensorOperator(rank=1, representation='q', space='relative')
        self.assertEqual(basic_block.to_expression(), "q1")  # add assertion here


if __name__ == '__main__':
    unittest.main()
