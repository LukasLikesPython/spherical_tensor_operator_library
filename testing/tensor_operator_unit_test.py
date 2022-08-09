import unittest
from tensor_operator import TensorOperator, TensorOperatorComposite
import win_unicode_console

class TestTensorOperator(unittest.TestCase):

    basic_block = TensorOperator(rank=1, representation='q', space='relative')
    other_block = TensorOperator(rank=1, representation='k', space='relative')

    def test_building_block(self):
        self.assertEqual("1 * q_1", str(self.basic_block))  # add assertion here

    def test_couple(self):
        self.assertEqual("2 * {q_1 x q_1}_2", str(self.basic_block.couple(self.basic_block, 2, 2)))
        self.assertEqual("1 * {q_1 x k_1}_2", str(self.basic_block.couple(self.other_block, 2, 1)))
        self.assertEqual("1 * {q_1 x k_1}_1", str(self.basic_block.couple(self.other_block, 1, 1)))
        self.assertEqual("1 * {q_1 x k_1}_0", str(self.basic_block.couple(self.other_block, 0, 1)))
        self.assertEqual("0", str(self.basic_block.couple(self.other_block, 3, 1)))  # 3 > 1 + 1, so not allowed
        self.assertEqual("0", str(self.basic_block.couple(self.basic_block, 100, 1)))
        self.assertEqual("0", str(self.basic_block.couple(self.basic_block, -1, 1)))

    def test_couple_tensor_operator_list(self):
        pass

    def test_mul(self):
        self.assertEqual("5.0 * q_1", str(self.basic_block * 5.))
        self.assertEqual("2 * q_1", str(2 * self.basic_block))

    def test_add_same(self):
        self.assertEqual("2 * q_1", str(self.basic_block + self.basic_block))

    def test_add_other(self):
        self.assertEqual("1 * q_1 + 1 * k_1", str(self.basic_block + self.other_block))

    def test_add_tensor_operator_list(self):
        tensorList = self.basic_block + self.other_block
        self.assertEqual("2 * q_1 + 1 * k_1", str(tensorList + self.basic_block))
        self.assertEqual("2 * q_1 + 1 * k_1", str(self.basic_block + tensorList))


if __name__ == '__main__':
    unittest.main()
