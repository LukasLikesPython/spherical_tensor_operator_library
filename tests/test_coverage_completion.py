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


class CompleteQuantumStates(unittest.TestCase):
    def test_something(self):
        state_a = BasicState(Symbol("a"), rel_space)
        state_b = BasicState(Symbol("b"), rel_space)
        self.assertEqual("|c(ab)>", str(state_a.couple(state_b, Symbol("c"))))  # add assertion here


if __name__ == '__main__':
    unittest.main()
