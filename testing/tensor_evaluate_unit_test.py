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
sig1 = TensorOperator(rank=1, symbol=Symbol('\u03C3\u2081'), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol('\u03C3\u2082'), space=spin_space)

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
        self.assertEqual(True, False)

    def test_matrix_element_decouple_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        composite = me.decouple()
        self.assertEqual(True, False)

    def test_add_unit_space(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_2)
        self.assertEqual(True, False)

    def test_evaluate_two_spaces(self):
        me = MatrixElement(bra_2, ket_2, tensor_op_2)
        me.evaluate(subsdict)  # TODO does this work? do I need the composite in between?
        self.assertEqual(True, False)

    def test_init_matrix_element_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        self.assertEqual(True, False)

    def test_matrix_element_decouple_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        composite = me.decouple()
        self.assertEqual(True, False)

    def test_evaluate_three_spaces(self):
        me = MatrixElement(bra_3, ket_3, tensor_op_3)
        me.evaluate(subsdict)  # TODO does this work? do I need the composite in between?
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
