import unittest
from sympy import Symbol

from tensor_space import TensorSpace
from tensor_transformation import TensorFromVectors
from quantum_states import BasicState
from tensor_operator import TensorOperator
from tensor_evaluate import MatrixElement

# spaces
rel_space = TensorSpace('rel', 0)
spin_space = TensorSpace('spin', 1)
cm_space = TensorSpace('cm', 2)

# states
s = BasicState(Symbol('s'), spin_space)
l = BasicState(Symbol('l'), rel_space, Symbol('p'))
ket_2 = l.couple(s, Symbol('j'))
sp = BasicState(Symbol("s'"), spin_space)
lp = BasicState(Symbol("l'"), rel_space, Symbol("p'"))
bra_2 = lp.couple(sp, Symbol("j'"))

# basic operators
q = TensorOperator(rank=1, symbol=Symbol('q'), space=rel_space)
sig1 = TensorOperator(rank=1, symbol=Symbol('sig1'), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol('sig2'), space=spin_space)

# coupled operator 2 body
tensor_op_2 = TensorFromVectors.scalar_product(q, sig1).couple(TensorFromVectors.scalar_product(q, sig2), 0, 1)

# double decoupling test
L = BasicState(Symbol('L'), cm_space, Symbol('P'))
Lp = BasicState(Symbol("L'"), cm_space, Symbol("P'"))
ket_3 = l.couple(s, Symbol('j')).couple(L, Symbol('J'))
bra_3 = lp.couple(sp, Symbol("j'")).couple(Lp, Symbol("J'"))
P = TensorOperator(rank=1, symbol=Symbol('P'), space=cm_space)
Psq = TensorFromVectors.scalar_product(P, P)
tensor_op_3 = tensor_op_2.couple(Psq, 0, 1)
# substitution dictionary
subsdict = {Symbol('J'): 0, Symbol("J'"): 0, Symbol('L'): 0, Symbol("L'"): 0, Symbol('l'): 1,
            Symbol("l'"): 1, Symbol('s'): 1, Symbol("s'"): 1, Symbol('j'): 0, Symbol("j'"): 0}


class TensorEvaluateTest(unittest.TestCase):

    def test_init_matrix_element_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        self.assertEqual("<p'j'(l's')m_j'|{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0|pj(ls)m_j> + "
                         "sqrt(5) * <p'j'(l's')m_j'|{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0|pj(ls)m_j>", str(me))

    def test_matrix_element_decouple_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        composite = me.decouple()
        self.assertEqual("(-1)**(j' + l + s')*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 0) * "
                         "<p'l'||{q_1 x q_1}_0||pl><s'||{sig1_1 x sig2_1}_0||s> + "
                         "(-1)**(j' + l + s')*sqrt(5)*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 0) * "
                         "<p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_2||s>", str(composite))

    def test_add_unit_space(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_2)
        self.assertEqual("<p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 x I_cm_0}_0|pPJ(j(ls)L)m_J>"
                         " + sqrt(5) * <p'P'J'(j'(l's')L')m_J'|{{{q_1 x q_1}_2 x "
                         "{sig1_1 x sig2_1}_2}_0 x I_cm_0}_0|pPJ(j(ls)L)m_J>", str(me))

    def test_evaluate_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        result = me.decouple().evaluate(subsdict)
        self.assertEqual("<1||{sig1_1 x sig2_1}_0||1>*<p'1||{q_1 x q_1}_0||p1>/3 +"
                         " sqrt(5)*<1||{sig1_1 x sig2_1}_2||1>*<p'1||{q_1 x q_1}_2||p1>/3", str(result))

    def test_init_matrix_element_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        self.assertEqual(True, False)

    def test_matrix_element_decouple_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        composite = me.decouple()
        self.assertEqual(True, False)

    def test_evaluate_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        result = me.decouple().evaluate(subsdict)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
