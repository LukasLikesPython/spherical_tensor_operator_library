import unittest
from sympy import I, sqrt
import sys

sys.path.append('..')

from tensor_space import TensorSpace
from tensor_transformation import TensorFromVectors
from tensor_operator import TensorOperator


rel_space = TensorSpace("relative", 0)


class TestVectorToTensor(unittest.TestCase):

    q_1 = TensorOperator(rank=1, factor=1, symbol="q", space=rel_space)
    k_1 = TensorOperator(rank=1, factor=1, symbol="k", space=rel_space)

    def test_scalar_product(self):
        tensor_object_scalar = self.q_1.couple(self.k_1, 0, -sqrt(3))

        self.assertEqual(str(tensor_object_scalar), str(TensorFromVectors.scalar_product(self.q_1, self.k_1)))

    def test_vector_product(self):
        tensor_object_vector = self.q_1.couple(self.k_1, 1, -I * sqrt(2))

        self.assertEqual(str(tensor_object_vector), str(TensorFromVectors.vector_product(self.q_1, self.k_1)))


if __name__ == "__main__":
    unittest.main()
