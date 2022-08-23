import unittest
from tensor_operator import TensorOperator
from tensor_algebra import TensorAlgebra
from tensor_transformation import TensorFromVectors
from sympy import I
from tensor_space import TensorSpace


rel_space = TensorSpace('relative', 0)
spin_space = TensorSpace('spin', 1)
cm_space = TensorSpace('cm', 2)

class TestTensorAlgebra(unittest.TestCase):
    q = TensorOperator(rank=1, symbol='q', space=rel_space)
    k = TensorOperator(rank=1, symbol='k', space=rel_space)
    P = TensorOperator(rank=1, symbol='P', space=cm_space)
    sig1 = TensorOperator(rank=1, symbol='sig1', space=spin_space)
    sig2 = TensorOperator(rank=1, symbol='sig2', space=spin_space)
    qk0_block = q.couple(k, 0, 1, False)
    qk1_block = q.couple(k, 1, 1, False)
    qk2_block = q.couple(k, 2, 1, False)
    kq0_block = k.couple(q, 0, 1, False)
    kq1_block = k.couple(q, 1, 1, False)
    kq2_block = k.couple(q, 2, 1, False)

    def test_commute(self):
        self.assertEqual(self.qk0_block, TensorAlgebra.commute(self.kq0_block))
        self.assertEqual(-1 * self.qk1_block, TensorAlgebra.commute(self.kq1_block))
        self.assertEqual(self.qk2_block, TensorAlgebra.commute(self.kq2_block))

    def test_recouple_ABxCD_ACxBD(self):
        operator = TensorFromVectors.scalar_product(self.q, self.sig1). \
            couple(TensorFromVectors.scalar_product(self.q, self.sig2), 0, 1)
        self.assertEqual(
            '1 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 + sqrt(5) * {{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0',
            str(TensorAlgebra.recouple(operator)))

    def test_recouple_ABxCD_ABCxD(self):
        operator = I * TensorFromVectors.scalar_product(TensorFromVectors.vector_product(self.q, self.P), self.sig1)
        self.assertEqual('sqrt(6) * {{q_1 x sig1_1}_1 x P_1}_0', str(TensorAlgebra.recouple(operator)))

if __name__ == '__main__':
    unittest.main()
