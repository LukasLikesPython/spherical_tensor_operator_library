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


    def test_recouple_ABxCD_ACxBD(self):
        operator = TensorFromVectors.scalar_product(self.q, self.sig1). \
            couple(TensorFromVectors.scalar_product(self.q, self.sig2), 0, 1)
        self.assertEqual(
            '1 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 + sqrt(5) * {{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0',
            str(TensorAlgebra.recouple(operator)))

        qxk = TensorFromVectors.vector_product(self.q, self.k)
        operator = TensorFromVectors.scalar_product(qxk, self.sig1). \
            couple(TensorFromVectors.scalar_product(qxk, self.sig2), 0, 1)
        self.assertEqual(
            '-2 * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_0 x {sig1_1 x sig2_1}_0}_0 + -2*sqrt(5) * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_2 x {sig1_1 x sig2_1}_2}_0',
            str(TensorAlgebra.recouple(operator)))

    def test_recouple_ABxC_ACxB(self):
        operator = I * TensorFromVectors.scalar_product(TensorFromVectors.vector_product(self.q, self.P), self.sig1)
        self.assertEqual('sqrt(6) * {{q_1 x sig1_1}_1 x P_1}_0', str(TensorAlgebra.recouple(operator)))

    def test_complicated_recouple(self):
        operator = TensorFromVectors.scalar_product(self.k, self.k).\
            couple(TensorFromVectors.scalar_product(self.P, self.P), 0, 1).\
            couple(TensorFromVectors.scalar_product(self.sig1, self.sig2), 0, 1)
        self.assertEqual(
            '-3*sqrt(3) * {{{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0 x {P_1 x P_1}_0}_0',
            str(TensorAlgebra.recouple(operator)))

        operator = TensorFromVectors.scalar_product(TensorFromVectors.vector_product(self.k, self.P),
                                                    TensorFromVectors.vector_product(self.k, self.P)).\
            couple(TensorFromVectors.scalar_product(self.sig1, self.sig2), 0, 1)
        self.assertEqual(
            '-2*sqrt(3) * {{{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0 x {P_1 x P_1}_0}_0 + sqrt(15) * {{{k_1 x k_1}_2 x {sig1_1 x sig2_1}_0}_2 x {P_1 x P_1}_2}_0',
            str(TensorAlgebra.recouple(operator)))

        operator = TensorFromVectors.scalar_product(TensorFromVectors.vector_product(self.sig1, self.sig2),
                                                    TensorFromVectors.vector_product(self.q, self.k)).\
            couple(TensorFromVectors.scalar_product(self.q, self.P), 0, 1)
        self.assertEqual(
            '-2 * {{{{k_1 x q_1}_1 x q_1}_0 x {sig1_1 x sig2_1}_1}_1 x P_1}_0 + -2*sqrt(3) * {{{{k_1 x q_1}_1 x q_1}_1 x {sig1_1 x sig2_1}_1}_1 x P_1}_0 + -2*sqrt(5) * {{{{k_1 x q_1}_1 x q_1}_2 x {sig1_1 x sig2_1}_1}_1 x P_1}_0',
            str(TensorAlgebra.recouple(operator)))



if __name__ == '__main__':
    unittest.main()
