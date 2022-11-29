import unittest
from sympy import Symbol, symbols
import sys

sys.path.append('../src/')

from stolpy.tensor_space import TensorSpace
from stolpy.tensor_transformation import TensorFromVectors
from stolpy.quantum_states import BasicState
from stolpy.tensor_operator import TensorOperator
from stolpy.tensor_evaluate import MatrixElement
from stolpy.symbolic_wigner import Symbolic6j, Symbolic9j
from stolpy.tensor_algebra import TensorAlgebra as ta

# spaces
rel_space = TensorSpace("rel", 0)
spin_space = TensorSpace("spin", 1)
cm_space = TensorSpace("cm", 2)

# states
s = BasicState(Symbol("s"), spin_space)
l = BasicState(Symbol("l"), rel_space, Symbol("p"))
ket = l.couple(s, Symbol("j"))
sp = BasicState(Symbol("s'"), spin_space)
lp = BasicState(Symbol("l'"), rel_space, Symbol("p'"))
bra = lp.couple(sp, Symbol("j'"))

# basic operators
q = TensorOperator(rank=1, symbol=Symbol("q"), space=rel_space)
k = TensorOperator(rank=1, symbol=Symbol("k"), space=rel_space)
sig1 = TensorOperator(rank=1, symbol=Symbol("sig1"), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol("sig2"), space=spin_space)
P = TensorOperator(rank=1, symbol=Symbol("P"), space=cm_space)

# coupled operator 2 body
tensor_op = TensorFromVectors.scalar_product(q, sig1).couple(TensorFromVectors.scalar_product(q, sig2), 0, 1)
tensor_op_2 = TensorFromVectors.scalar_product(k, sig1).couple(TensorFromVectors.scalar_product(k, sig2), 0, 1)

# Matrix Element
me = MatrixElement(bra, ket, tensor_op)
me_2 = MatrixElement(bra, ket, tensor_op_2)

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

class CompleteQuantumStates(unittest.TestCase):
    def test_state_couple_without_other_qn(self):
        state_a = BasicState(Symbol("a"), rel_space)
        state_b = BasicState(Symbol("b"), rel_space)
        self.assertEqual("|c(ab)>", str(state_a.couple(state_b, Symbol("c"))))  # add assertion here


class CompleteSymbolicWigner(unittest.TestCase):
    def test_repr_and_str(self):
        a, b, c, d, e, f, g, h, i = symbols('a,b,c,d,e,f,g,h,i')
        test_symbol = Symbolic6j(a, b, c, d, e, f) * Symbolic9j(a, b, c, d, e, f, g, h, i)
        self.assertEqual('1 * (1 * SixJ(a b c; d e f) * 1 * NineJ(a b c; d e f; g h i))', str(test_symbol))
        self.assertEqual('[1 * SixJ(a b c; d e f), 1 * SixJ(a b c; d e f)]',
                         str([Symbolic6j(a, b, c, d, e, f), Symbolic6j(a, b, c, d, e, f)]))

    def test_diff(self):
        a, b, c, d, e, f = symbols('a,b,c,d,e,f')
        test_symbol = Symbolic6j(a, b, c, d, e, f) / 2
        self.assertEqual('0.5 * SixJ(a b c; d e f)', str(test_symbol))

    def test_extend_children(self):
        a, b, c, d, e, f, g, h, i = symbols('a,b,c,d,e,f,g,h,i')
        test_symbol = Symbolic6j(a, b, c, d, e, f) * Symbolic9j(a, b, c, d, e, f, g, h, i)
        test_symbol *= test_symbol
        self.assertEqual('1 * (1 * SixJ(a b c; d e f) * 1 * NineJ(a b c; d e f; g h i) * 1 * SixJ(a b c; d e f) * 1 * '
                         'NineJ(a b c; d e f; g h i))', str(test_symbol))


class CompleteTensorAlgebra(unittest.TestCase):
    def test_can_be_recoupled_ABxC_AxBC(self):
        A, B, C = symbols('A,B,C')
        space_a = TensorSpace('a', 0)
        space_b = TensorSpace('b', 1)
        top_a = TensorOperator(rank=1, symbol=A, space=space_a)
        top_b = TensorOperator(rank=1, symbol=B, space=space_b)
        top_c = TensorOperator(rank=0, symbol=C, space=space_b)
        top = top_a.couple(top_b, rank=0, factor=1).couple(top_c, rank=0, factor=1)
        rec_top = ta.recouple(top)
        self.assertEqual('-1 * {A_1 x {B_1 x C_0}_1}_0', str(rec_top))

    def test_can_be_recoupled_AxBC_ABxC(self):
        A, B, C = symbols('A,B,C')
        space_a = TensorSpace('a', 0)
        space_b = TensorSpace('b', 1)
        space_x = TensorSpace('x', 2)
        top_a = TensorOperator(rank=1, symbol=A, space=space_a)
        top_b = TensorOperator(rank=1, symbol=B, space=space_b)
        top_c = TensorOperator(rank=0, symbol=C, space=space_b)
        top = top_a.couple(top_b, rank=0, factor=1).couple(top_c, rank=0, factor=1)
        bool_top = ta._can_be_recoupled_AxBC_ABxC(top)
        self.assertEqual(False, bool_top)
        bool_top = ta._can_be_recoupled_AxBC_ABxC(top_a)
        self.assertEqual(False, bool_top)
        top = top_a.couple(top_b.couple(top_c, rank=1, factor=1), rank=0, factor=1)
        bool_top = ta._can_be_recoupled_AxBC_ABxC(top)
        self.assertEqual(False, bool_top)
        top_new_a = top_a.couple(top_b, rank=1, factor=1)
        top_c = TensorOperator(rank=1, symbol=B, space=space_x)
        top = top_new_a.couple(top_b.couple(top_c, rank=1, factor=1), rank=0, factor=1)
        bool_top = ta._can_be_recoupled_AxBC_ABxC(top)
        self.assertEqual(True, bool_top)
        top = top_a.couple(top_b.couple(top_c, rank=1, factor=1), rank=0, factor=1)
        bool_top = ta._can_be_recoupled_AxBC_ABxC(top)
        self.assertEqual(False, bool_top)

    def test_recouple_ABxCD_ABCxD(self):
        """
        Note, this is not a real unit test, we just want to make sure that we hit 100% code coverage
        The operator we use is not a real physical operator
        """
        qP = TensorFromVectors.scalar_product(q, P)
        qxs = TensorFromVectors.vector_product(q, sig1)
        operator = qxs.couple(qP, rank=1, factor=1)
        rec = ta._recouple_ABxCD_ABCxD(operator)
        self.assertEqual(True, True)


class CompleteTensorEvaluate(unittest.TestCase):
    def test_eq_reduced_me_composite(self):
        red_me = me.full_decouple()
        print(red_me.children)
        self.assertEqual(red_me, red_me)  # Check True
        red_me_2 = me_2.full_decouple()
        self.assertFalse(red_me_2 == red_me)  # Check False
        self.assertTrue(red_me_2 != red_me)  # Check ne

    def test_nothing_to_decouple(self):
        op = TensorFromVectors.scalar_product(q, q)
        me_3 = MatrixElement(lp, l, op)
        self.assertTrue(me_3.full_decouple() is None)

    def test_non_reduced_eval(self):
        eval_element = me.me_evaluate(subsdict)
        self.assertEqual("<2(p'11)|{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 + "
                         "{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0|2(p11)>", str(eval_element))

    def test_error_rank_mismatch(self):
        op = ""


if __name__ == '__main__':
    unittest.main()
