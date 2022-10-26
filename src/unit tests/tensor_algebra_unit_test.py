import unittest
from src.tensor_operator import TensorOperator
from src.tensor_algebra import TensorAlgebra
from src.tensor_transformation import TensorFromVectors
from sympy import I
from src.tensor_space import TensorSpace

rel_space = TensorSpace("relative", 0)
spin_space = TensorSpace("spin", 1)
cm_space = TensorSpace("cm", 2)


class TestTensorAlgebra(unittest.TestCase):

    q = TensorOperator(rank=1, symbol="q", space=rel_space)
    k = TensorOperator(rank=1, symbol="k", space=rel_space)
    P = TensorOperator(rank=1, symbol="P", space=cm_space)
    sig1 = TensorOperator(rank=1, symbol="sig1", space=spin_space)
    sig2 = TensorOperator(rank=1, symbol="sig2", space=spin_space)
    qsq = TensorFromVectors.scalar_product(q, q)
    ksq = TensorFromVectors.scalar_product(k, k)
    qxk = TensorFromVectors.vector_product(q, k)
    sxs = TensorFromVectors.vector_product(sig1, sig2)
    ssq = TensorFromVectors.scalar_product(sig1, sig2)
    qxP = TensorFromVectors.vector_product(q, P)
    kxP = TensorFromVectors.vector_product(k, P)
    kP = TensorFromVectors.scalar_product(k, P)
    qP = TensorFromVectors.scalar_product(q, P)
    Psq = TensorFromVectors.scalar_product(P, P)
    qs1 = TensorFromVectors.scalar_product(q, sig1)
    qs2 = TensorFromVectors.scalar_product(q, sig2)
    ks1 = TensorFromVectors.scalar_product(k, sig1)
    ks2 = TensorFromVectors.scalar_product(k, sig2)

    def test_recouple_ABxCD_ACxBD(self):
        print("test_recouple_ABxCD_ACxBD")
        operator = self.qs1.couple(self.qs2, 0, 1)
        self.assertEqual(
            "1 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0" + " + sqrt(5) * {{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra._recouple_ABxCD_ACxBD(operator)),
        )

        operator = TensorFromVectors.scalar_product(self.qxk, self.sig1).couple(
            TensorFromVectors.scalar_product(self.qxk, self.sig2), 0, 1
        )
        self.assertEqual(
            "-2 * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -2*sqrt(5) * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

    def test_recouple_ABxC_ACxB(self):
        print("test_recouple_ABxC_ACxB")
        operator = I * TensorFromVectors.scalar_product(self.qxP, self.sig1)
        self.assertEqual("sqrt(6) * {{q_1 x sig1_1}_1 x P_1}_0", str(TensorAlgebra._recouple_ABxC_ACxB(operator)))

    def test_recouple_ABxCD_ABCxD(self):
        print("test_recouple_ABxCD_ABCxD")
        operator = TensorFromVectors.scalar_product(self.qxk, self.sxs).couple(self.qP, 0, 1)
        self.assertEqual(
            "6 * {{{{k_1 x q_1}_1 x {sig1_1 x sig2_1}_1}_0 x q_1}_1 x P_1}_0",
            str(TensorAlgebra._recouple_ABxCD_ABCxD(operator)),
        )

    def test_recouple_ABxC_AxBC(self):
        print("test_recouple_ABxC_AxBC")
        operator = self.qs1.couple(self.sig2, 1, 1)  # Artificial test case, not a real operator
        self.assertEqual(
            "sqrt(3)/3 * {q_1 x {sig1_1 x sig2_1}_0}_1 + -1 * {q_1 x {sig1_1 x sig2_1}_1}_1"
            + " + sqrt(15)/3 * {q_1 x {sig1_1 x sig2_1}_2}_1",
            str(TensorAlgebra._recouple_ABxC_AxBC(operator)),
        )

    def test_recouple_NLO_central(self):
        print("test_recouple_NLO_central")
        operator = self.qsq
        self.assertEqual("-sqrt(3) * {q_1 x q_1}_0", str(TensorAlgebra.recouple(operator)))
        operator = self.ksq
        self.assertEqual("-sqrt(3) * {k_1 x k_1}_0", str(TensorAlgebra.recouple(operator)))
        operator = self.qsq.couple(self.ssq, 0, 1)
        self.assertEqual("3 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0", str(TensorAlgebra.recouple(operator)))
        operator = self.ksq.couple(self.ssq, 0, 1)
        self.assertEqual("3 * {{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0", str(TensorAlgebra.recouple(operator)))

    def test_recouple_NLO_spin_orbit(self):
        print("test_recouple_NLO_spin_orbit")
        operator = TensorFromVectors.scalar_product(self.qxk, self.sig1) * (I / 2)
        self.assertEqual("sqrt(6)/2 * {{k_1 x q_1}_1 x sig1_1}_0", str(TensorAlgebra.recouple(operator)))

    def test_recouple_NLO_tensor(self):
        print("test_recouple_NLO_tensor")
        operator = self.qs1.couple(self.qs2, 0, 1)
        self.assertEqual(
            "1 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 + sqrt(5) * {{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )
        operator = self.ks1.couple(self.ks2, 0, 1)
        self.assertEqual(
            "1 * {{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0 + sqrt(5) * {{k_1 x k_1}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

    def test_recouple_N3LO_central(self):
        print("test_recouple_N3LO_central")
        operator = self.qsq.couple(self.qsq, 0, 1)
        self.assertEqual("3 * {{q_1 x q_1}_0 x {q_1 x q_1}_0}_0", str(TensorAlgebra.recouple(operator)))
        operator = self.ksq.couple(self.ksq, 0, 1)
        self.assertEqual("3 * {{k_1 x k_1}_0 x {k_1 x k_1}_0}_0", str(TensorAlgebra.recouple(operator)))
        operator = self.qsq.couple(self.ksq, 0, 1)
        self.assertEqual("3 * {{k_1 x k_1}_0 x {q_1 x q_1}_0}_0", str(TensorAlgebra.recouple(operator)))
        operator = TensorFromVectors.scalar_product(self.qxk, self.qxk)
        self.assertEqual("2*sqrt(3) * {{k_1 x q_1}_1 x {k_1 x q_1}_1}_0", str(TensorAlgebra.recouple(operator)))
        # we skip q^4 sig sig and k^4 sig sig since they are all similar
        operator = TensorFromVectors.scalar_product(self.qxk, self.qxk).couple(self.ssq, 0, 1)
        self.assertEqual(
            "-6 * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_0 x {sig1_1 x sig2_1}_0}_0", str(TensorAlgebra.recouple(operator))
        )

    def test_recouple_N3LO_spin_orbit(self):
        print("test_recouple_N3LO_spin_orbit")
        operator = I / 2 * TensorFromVectors.scalar_product(self.sig1, self.qxk).couple(self.qsq, 0, 1)
        self.assertEqual(
            "3*sqrt(2)/2 * {{{k_1 x q_1}_1 x {q_1 x q_1}_0}_1 x sig1_1}_0", str(TensorAlgebra.recouple(operator))
        )

        operator = I / 2 * TensorFromVectors.scalar_product(self.sig1, self.qxk).couple(self.ksq, 0, 1)
        self.assertEqual(
            "3*sqrt(2)/2 * {{{k_1 x k_1}_0 x {k_1 x q_1}_1}_1 x sig1_1}_0", str(TensorAlgebra.recouple(operator))
        )

    def test_recouple_N3LO_tensor(self):
        print("test_recouple_N3LO_tensor")
        operator = self.qs1.couple(self.qs2, 0, 1).couple(self.qsq, 0, 1)
        self.assertEqual(
            "-sqrt(3) * {{{q_1 x q_1}_0 x {q_1 x q_1}_0}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -sqrt(15) * {{{q_1 x q_1}_2 x {q_1 x q_1}_0}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = self.qs1.couple(self.qs2, 0, 1).couple(self.ksq, 0, 1)
        self.assertEqual(
            "-sqrt(3) * {{{k_1 x k_1}_0 x {q_1 x q_1}_0}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -sqrt(15) * {{{k_1 x k_1}_0 x {q_1 x q_1}_2}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = self.ks1.couple(self.ks2, 0, 1).couple(self.qsq, 0, 1)
        self.assertEqual(
            "-sqrt(3) * {{{k_1 x k_1}_0 x {q_1 x q_1}_0}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -sqrt(15) * {{{k_1 x k_1}_2 x {q_1 x q_1}_0}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = self.ks1.couple(self.ks2, 0, 1).couple(self.ksq, 0, 1)
        self.assertEqual(
            "-sqrt(3) * {{{k_1 x k_1}_0 x {k_1 x k_1}_0}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -sqrt(15) * {{{k_1 x k_1}_2 x {k_1 x k_1}_0}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = TensorFromVectors.scalar_product(self.qxk, self.sig1).couple(
            TensorFromVectors.scalar_product(self.qxk, self.sig2), 0, 1
        )
        self.assertEqual(
            "-2 * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_0 x {sig1_1 x sig2_1}_0}_0"
            + " + -2*sqrt(5) * {{{k_1 x q_1}_1 x {k_1 x q_1}_1}_2 x {sig1_1 x sig2_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

    def test_three_spaces(self):
        print("test_three_spaces")
        operator = self.ksq.couple(self.Psq, 0, 1).couple(self.ssq, 0, 1)
        self.assertEqual(
            "-3*sqrt(3) * {{{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0 x {P_1 x P_1}_0}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = TensorFromVectors.scalar_product(self.kxP, self.kxP).couple(self.ssq, 0, 1)
        self.assertEqual(
            "-2*sqrt(3) * {{{k_1 x k_1}_0 x {sig1_1 x sig2_1}_0}_0 x {P_1 x P_1}_0}_0"
            + " + sqrt(15) * {{{k_1 x k_1}_2 x {sig1_1 x sig2_1}_0}_2 x {P_1 x P_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = TensorFromVectors.scalar_product(self.sxs, self.qxk).couple(self.qP, 0, 1)
        self.assertEqual(
            "-2 * {{{{k_1 x q_1}_1 x q_1}_0 x {sig1_1 x sig2_1}_1}_1 x P_1}_0"
            + " + -2*sqrt(3) * {{{{k_1 x q_1}_1 x q_1}_1 x {sig1_1 x sig2_1}_1}_1 x P_1}_0"
            + " + -2*sqrt(5) * {{{{k_1 x q_1}_1 x q_1}_2 x {sig1_1 x sig2_1}_1}_1 x P_1}_0",
            str(TensorAlgebra.recouple(operator)),
        )

        operator = TensorFromVectors.scalar_product(self.qxP, self.sig1).couple(
            TensorFromVectors.scalar_product(self.qxP, self.sig2), 0, 1
        )
        self.assertEqual(
            "-2*sqrt(3)/3 * {{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 x {P_1 x P_1}_0}_0"
            + " + sqrt(15)/3 * {{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0 x {P_1 x P_1}_0}_0"
            + " + sqrt(15)/3 * {{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_2}_2 x {P_1 x P_1}_2}_0"
            + " + sqrt(15)/3 * {{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_0}_2 x {P_1 x P_1}_2}_0"
            + " + -sqrt(105)/3 * {{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_2 x {P_1 x P_1}_2}_0",
            str(TensorAlgebra.recouple(operator)),
        )


if __name__ == "__main__":
    unittest.main()
