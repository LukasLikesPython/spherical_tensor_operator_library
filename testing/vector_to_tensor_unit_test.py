import unittest
from tensor_transformation import VectorOperator, TensorFromVectors
from tensor_operator import TensorOperator
from sympy import I, sqrt

class TestVectorToTensor(unittest.TestCase):
    q_1 = TensorOperator(rank=1, factor=1, symbol='q', space='relative')
    k_1 = TensorOperator(rank=1, factor=1, symbol='k', space='relative')
    vec_q = VectorOperator(space='relative', symbol='q')
    vec_k = VectorOperator(space='relative', symbol='k')

    def test_scalar_product(self):
        tensor_object_scalar = self.q_1.couple(self.k_1, 0, -sqrt(3))
        self.assertEqual(str(tensor_object_scalar),
                         str(TensorFromVectors.tensor_from_scalar_product(self.vec_q, self.vec_k)))

        self.assertEqual(str(tensor_object_scalar),
                         str(TensorFromVectors.tensor_from_scalar_product(self.q_1, self.k_1)))

    def test_vector_product(self):
        tensor_object_vector = self.q_1.couple(self.k_1, 1, -I * sqrt(2))
        self.assertEqual(str(tensor_object_vector),
                         str(TensorFromVectors.tensor_from_vector_product(self.vec_q, self.vec_k)))

        self.assertEqual(str(tensor_object_vector),
                         str(TensorFromVectors.tensor_from_vector_product(self.q_1, self.k_1)))

if __name__ == '__main__':
    unittest.main()
