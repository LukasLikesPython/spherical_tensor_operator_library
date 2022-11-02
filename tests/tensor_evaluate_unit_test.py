import unittest
from sympy import Symbol
import sys

sys.path.append('../src/')

from stolpy.tensor_space import TensorSpace
from stolpy.tensor_transformation import TensorFromVectors
from stolpy.quantum_states import BasicState
from stolpy.tensor_operator import TensorOperator
from stolpy.tensor_evaluate import MatrixElement

# spaces
rel_space = TensorSpace("rel", 0)
spin_space = TensorSpace("spin", 1)
cm_space = TensorSpace("cm", 2)

# states
s = BasicState(Symbol("s"), spin_space)
l = BasicState(Symbol("l"), rel_space, Symbol("p"))
ket_2 = l.couple(s, Symbol("j"))
sp = BasicState(Symbol("s'"), spin_space)
lp = BasicState(Symbol("l'"), rel_space, Symbol("p'"))
bra_2 = lp.couple(sp, Symbol("j'"))

# basic operators
q = TensorOperator(rank=1, symbol=Symbol("q"), space=rel_space)
sig1 = TensorOperator(rank=1, symbol=Symbol("sig1"), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol("sig2"), space=spin_space)

# coupled operator 2 body
tensor_op_2 = TensorFromVectors.scalar_product(q, sig1).couple(TensorFromVectors.scalar_product(q, sig2), 0, 1)

# double decoupling test
L = BasicState(Symbol("L"), cm_space, Symbol("P"))
Lp = BasicState(Symbol("L'"), cm_space, Symbol("P'"))
ket_3 = l.couple(s, Symbol("j")).couple(L, Symbol("J"))
bra_3 = lp.couple(sp, Symbol("j'")).couple(Lp, Symbol("J'"))
P = TensorOperator(rank=1, symbol=Symbol("P"), space=cm_space)
op1 = TensorFromVectors.scalar_product(TensorFromVectors.vector_product(q, P), sig1)
op2 = TensorFromVectors.scalar_product(TensorFromVectors.vector_product(q, P), sig2)
tensor_op_3 = op1.couple(op2, 0, 1)

# substitution dictionary
subsdict = {
    Symbol("J"): 1,
    Symbol("J'"): 1,
    Symbol("L"): 2,
    Symbol("L'"): 2,
    Symbol("l"): 1,
    Symbol("l'"): 1,
    Symbol("s"): 1,
    Symbol("s'"): 1,
    Symbol("j"): 2,
    Symbol("j'"): 2,
}


class TensorEvaluateTest(unittest.TestCase):
    """
    Note, the comparison terms were calculated by hand, but the order and factor splitting was adapted to match the
    output.
    """

    def test_init_matrix_element_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        self.assertEqual(
            "<p'j'(l's')m_j'|{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0|pj(ls)m_j> + "
            "sqrt(5) * <p'j'(l's')m_j'|{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0|pj(ls)m_j>",
            str(me),
        )

    def test_matrix_element_decouple_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        composite = me.full_decouple()
        self.assertEqual(
            "(-1)**(j' + l + s')*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 0) * "
            "<p'l'||{q_1 x q_1}_0||pl><s'||{sig1_1 x sig2_1}_0||s> + "
            "(-1)**(j' + l + s' + 2)*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 2) * "
            "<p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_2||s>",
            str(composite),
        )

    def test_add_unit_space(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_2)
        self.assertEqual(
            "<p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 x I_cm_0}_0|pPJ(j(ls)L)m_J>"
            " + sqrt(5) * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_2 x "
            "{sig1_1 x sig2_1}_2}_0 x I_cm_0}_0|pPJ(j(ls)L)m_J>",
            str(me),
        )

    def test_evaluate_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        result = me.evaluate(subsdict)
        self.assertEqual(
            "<1||{sig1_1 x sig2_1}_0||1>*<p'1||{q_1 x q_1}_0||p1>/3 +"
            " <1||{sig1_1 x sig2_1}_2||1>*<p'1||{q_1 x q_1}_2||p1>/30",
            str(result),
        )

    def test_init_matrix_element_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        self.assertEqual(
            "-2*sqrt(3)/3 * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 x "
            "{P_1 x P_1}_0}_0|pPJ(j(ls)L)m_J> + sqrt(15)/3 * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_2 x "
            "{sig1_1 x sig2_1}_2}_0 x {P_1 x P_1}_0}_0|pPJ(j(ls)L)m_J> + sqrt(15)/3 * "
            "<p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_2}_2 x "
            "{P_1 x P_1}_2}_0|pPJ(j(ls)L)m_J> + sqrt(15)/3 * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_2 x "
            "{sig1_1 x sig2_1}_0}_2 x {P_1 x P_1}_2}_0|pPJ(j(ls)L)m_J> + "
            "-sqrt(105)/3 * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_2 x "
            "{sig1_1 x sig2_1}_2}_2 x {P_1 x P_1}_2}_0|pPJ(j(ls)L)m_J>",
            str(me),
        )

    def test_matrix_element_full_decouple_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        composite = me.full_decouple()
        self.assertEqual(
            "-2*(-1)**(J' + L' + j)*sqrt(3)*KroneckerDelta(J, J')/3 * SixJ(j' L' J'; L j 0) * "
            "sqrt(2*j + 1)*sqrt(2*j' + 1) * NineJ(0 0 0; l' s' j'; l s j) * <p'l'||{q_1 x q_1}_0||pl>"
            "<s'||{sig1_1 x sig2_1}_0||s><P'L'||{P_1 x P_1}_0||PL>"
            " + (-1)**(J' + L' + j)*sqrt(15)*KroneckerDelta(J, J')/3 * SixJ(j' L' J'; L j 0) * "
            "sqrt(2*j + 1)*sqrt(2*j' + 1) * NineJ(2 2 0; l' s' j'; l s j) * "
            "<p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_2||s><P'L'||{P_1 x P_1}_0||PL>"
            " + (-1)**(J' + L' + j + 2)*sqrt(3)*KroneckerDelta(J, J')/3 * SixJ(j' L' J'; L j 2) * "
            "sqrt(5)*sqrt(2*j + 1)*sqrt(2*j' + 1) * NineJ(0 2 2; l' s' j'; l s j) * "
            "<p'l'||{q_1 x q_1}_0||pl><s'||{sig1_1 x sig2_1}_2||s><P'L'||{P_1 x P_1}_2||PL>"
            " + (-1)**(J' + L' + j + 2)*sqrt(3)*KroneckerDelta(J, J')/3 * SixJ(j' L' J'; L j 2) * "
            "sqrt(5)*sqrt(2*j + 1)*sqrt(2*j' + 1) * NineJ(2 0 2; l' s' j'; l s j) * "
            "<p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_0||s><P'L'||{P_1 x P_1}_2||PL>"
            " + -(-1)**(J' + L' + j + 2)*sqrt(21)*KroneckerDelta(J, J')/3 * SixJ(j' L' J'; L j 2) * "
            "sqrt(5)*sqrt(2*j + 1)*sqrt(2*j' + 1) * NineJ(2 2 2; l' s' j'; l s j) * "
            "<p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_2||s><P'L'||{P_1 x P_1}_2||PL>",
            str(composite),
        )

    def test_evaluate_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        result = me.evaluate(subsdict)
        self.assertEqual(
            "-2*sqrt(15)*<1||{sig1_1 x sig2_1}_0||1>*<P'2||{P_1 x P_1}_0||P2>*<p'1||{q_1 x q_1}_0||p1>/45"
            " + sqrt(21)*<1||{sig1_1 x sig2_1}_0||1>*<P'2||{P_1 x P_1}_2||P2>*<p'1||{q_1 x q_1}_2||p1>/180"
            " + sqrt(15)*<1||{sig1_1 x sig2_1}_2||1>*<P'2||{P_1 x P_1}_0||P2>*<p'1||{q_1 x q_1}_2||p1>/450"
            " + sqrt(21)*<1||{sig1_1 x sig2_1}_2||1>*<P'2||{P_1 x P_1}_2||P2>*<p'1||{q_1 x q_1}_0||p1>/180"
            " + sqrt(105)*<1||{sig1_1 x sig2_1}_2||1>*<P'2||{P_1 x P_1}_2||P2>"
            "*<p'1||{q_1 x q_1}_2||p1>/900",
            str(result),
        )


if __name__ == "__main__":
    unittest.main()
