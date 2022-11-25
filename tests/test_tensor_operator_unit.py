import unittest
import sys

sys.path.append('../src/')

from stolpy.tensor_operator import TensorOperator
from stolpy.tensor_space import TensorSpace

prior_spin_space = TensorSpace("prior_spin", -1)
rel_space = TensorSpace("relative", 0)
spin_space = TensorSpace("spin", 1)

so_space = rel_space + spin_space


class TestTensorOperator(unittest.TestCase):

    basic_block = TensorOperator(rank=1, symbol="q", space=rel_space)
    other_block = TensorOperator(rank=1, symbol="k", space=rel_space)
    other_space = TensorOperator(rank=1, symbol="sig", space=spin_space)
    other_space_2 = TensorOperator(rank=1, symbol="sig", space=prior_spin_space)
    qk0_block = basic_block.couple(other_block, 0, 1, False)
    qk1_block = basic_block.couple(other_block, 1, 1, False)
    qk2_block = basic_block.couple(other_block, 2, 1, False)
    kq0_block = other_block.couple(basic_block, 0, 1, False)
    kq1_block = other_block.couple(basic_block, 1, 1, False)
    kq2_block = other_block.couple(basic_block, 2, 1, False)

    def test_building_block(self):
        self.assertEqual("1 * q_1", str(self.basic_block))
        self.assertEqual("1 * k_1", str(self.other_block))
        self.assertEqual("1 * sig_1", str(self.other_space))

    def test_couple(self):
        self.assertEqual("2 * {q_1 x q_1}_2", str(self.basic_block.couple(self.basic_block, 2, 2)))
        self.assertEqual("1 * {k_1 x q_1}_2", str(self.basic_block.couple(self.other_block, 2, 1)))
        # Sorts alphabetically, so k comes before q
        self.assertEqual("-1 * {k_1 x q_1}_1", str(self.basic_block.couple(self.other_block, 1, 1)))
        self.assertEqual("1 * {k_1 x q_1}_0", str(self.basic_block.couple(self.other_block, 0, 1)))
        self.assertEqual(None, self.basic_block.couple(self.other_block, 3, 1))  # 3 > 1 + 1, so not allowed
        self.assertEqual(None, self.basic_block.couple(self.basic_block, 100, 1))
        self.assertEqual(None, self.basic_block.couple(self.basic_block, -1, 1))
        # Ordering for more complex operators
        self.assertEqual(
            "-1 * {{k_1 x k_1}_2 x {q_1 x q_1}_2}_1",
            str(
                self.basic_block.couple(self.basic_block, 2, 1).couple(
                    (self.other_block.couple(self.other_block, 2, 1)), 1, 1
                )
            ),
        )
        self.assertEqual(
            "-1 * {{k_1 x q_1}_1 x {q_1 x q_1}_2}_3",
            str(
                self.basic_block.couple(self.basic_block, 2, 1).couple(
                    (self.basic_block.couple(self.other_block, 1, 1)), 3, 1
                )
            ),
        )
        self.assertEqual(
            "1 * {{k_1 x q_1}_1 x {q_1 x q_1}_2}_2",
            str(
                self.basic_block.couple(self.basic_block, 2, 1).couple(
                    (self.basic_block.couple(self.other_block, 1, 1)), 2, 1
                )
            ),
        )
        self.assertEqual(
            rel_space,
            self.basic_block.couple(self.basic_block, 2, 1)
            .couple((self.basic_block.couple(self.other_block, 1, 1)), 2, 1)
            .space,
        )
        self.assertEqual(
            "1 * {{q_1 x q_1}_0 x k_1}_1",
            str(self.basic_block.couple(self.basic_block, 0, 1).couple(self.other_block, 1, 1)),
        )
        # Different space
        self.assertEqual("-1 * {q_1 x sig_1}_1", str(self.other_space.couple(self.basic_block, 1, 1)))
        self.assertEqual(so_space, self.other_space.couple(self.basic_block, 1, 1).space)
        self.assertEqual("1 * {sig_1 x q_1}_1", str(self.other_space_2.couple(self.basic_block, 1, 1)))

    def test_couple_tensor_operator_list(self):
        tlist = self.basic_block + self.other_block
        other_tlist = self.basic_block + self.other_space
        # Basic Block to Tensor List
        self.assertEqual("1 * {q_1 x sig_1}_1 + 1 * {k_1 x sig_1}_1", str(tlist.couple(self.other_space, 1, 1)))
        self.assertEqual("-1 * {q_1 x sig_1}_1 + -1 * {k_1 x sig_1}_1", str(self.other_space.couple(tlist, 1, 1)))
        # Handle zero elements
        self.assertEqual("-1 * {k_1 x q_1}_1", str(self.basic_block.couple(tlist, 1, 1)))
        # Coupling two tensor lists
        self.assertEqual(
            "1 * {q_1 x q_1}_0 + 1 * {q_1 x sig_1}_0 + 1 * {k_1 x q_1}_0 + 1 * {k_1 x sig_1}_0",
            str(tlist.couple(other_tlist, 0, 1)),
        )
        # Coupling two tensor lists with zero elements
        self.assertEqual(
            "1 * {q_1 x sig_1}_1 + 1 * {k_1 x q_1}_1 + 1 * {k_1 x sig_1}_1", str(tlist.couple(other_tlist, 1, 1))
        )

    def test_mul(self):
        self.assertEqual("5.0 * q_1", str(self.basic_block * 5.0))
        self.assertEqual("2 * q_1", str(2 * self.basic_block))
        # multiply coupled tensors
        coupled_block = (self.basic_block * 2).couple(self.basic_block, 2, 1) * 3
        self.assertEqual("6 * {q_1 x q_1}_2", str(coupled_block))
        self.assertEqual(6, coupled_block.factor)
        # Test the substructure, the factor should be one as it is carried to the outer layer
        self.assertEqual(1, coupled_block.substructure[0].factor)
        self.assertEqual(1, coupled_block.substructure[1].factor)

    def test_add_same(self):
        self.assertEqual("2 * q_1", str(self.basic_block + self.basic_block))
        self.assertEqual(None, self.basic_block - self.basic_block)

    def test_add_other(self):
        self.assertEqual("1 * q_1 + 1 * k_1", str(self.basic_block + self.other_block))
        self.assertEqual("1 * k_1", str(self.basic_block + self.other_block - self.basic_block))

    def test_add_tensor_operator_list(self):
        tensor_list = self.basic_block + self.other_block
        self.assertEqual("2 * q_1 + 1 * k_1", str(tensor_list + self.basic_block))
        self.assertEqual("2 * q_1 + 1 * k_1", str(self.basic_block + tensor_list))
        self.assertEqual("1 * k_1", str(-1 * self.basic_block + tensor_list))

    def test_add_tensor_lists(self):
        tensor_list = self.basic_block + self.other_block
        self.assertEqual("2 * q_1 + 2 * k_1", str(tensor_list + tensor_list))
        self.assertEqual("-1 * k_1", str(tensor_list - (self.basic_block + 2 * self.other_block)))
        self.assertEqual("", str(tensor_list - tensor_list))

    def test_commute(self):
        self.assertEqual(self.qk0_block, self.kq0_block.commute())
        self.assertEqual(-1 * self.qk1_block, self.kq1_block.commute())
        self.assertEqual(self.qk2_block, self.kq2_block.commute())


if __name__ == "__main__":
    unittest.main()
