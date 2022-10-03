import unittest
from sympy import Symbol

from tensor_space import TensorSpace
from tensor_transformation import TensorFromVectors
from quantum_states import BasicState

# spaces
rel_space = TensorSpace('rel', 0)
spin_space = TensorSpace('spin', 1)
cm_space = TensorSpace('cm', 2)

# states
s = BasicState(Symbol('s'), spin_space)
l = BasicState(Symbol('l'), rel_space, Symbol('p'))
ket = l.couple(s, Symbol('j'))
print(ket, ket.substructure)
sp = BasicState(Symbol("s'"), spin_space)
lp = BasicState(Symbol("l'"), rel_space, Symbol("p'"))
bra = lp.couple(sp, Symbol("j'"))
print(bra, bra.substructure)

# basic operators
q = TensorOperator(rank=1, symbol=Symbol('q'), space=rel_space)
sig1 = TensorOperator(rank=1, symbol=Symbol('\u03C3\u2081'), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol('\u03C3\u2082'), space=spin_space)
print(q)
print(sig1)
print(sig2)

# coupled operator
tensor_op = TensorFromVectors.scalar_product(q, sig1).couple(TensorFromVectors.scalar_product(q, sig2), 0, 1)
print(tensor_op)

# matrix elements
me = MatrixElement(bra, ket, tensor_op)
print(me)

# decouple
composite = me.decouple()
print(composite)

# double decoupling test
L = BasicState(Symbol('L'), cm_space, Symbol('P'))
Lp = BasicState(Symbol("L'"), cm_space, Symbol("P'"))
ket = l.couple(s, Symbol('j')).couple(L, Symbol('J'))
print(ket, ket.substructure)
bra = lp.couple(sp, Symbol("j'")).couple(Lp, Symbol("J'"))
print(bra, ket.substructure)
P = TensorOperator(rank=1, symbol=Symbol('P'), space=cm_space)
Psq = TensorFromVectors.scalar_product(P, P)
tensor_op = tensor_op.couple(Psq, 0, 1)

me = MatrixElement(bra, ket, tensor_op)
print(me)

composite = me.decouple()
print(composite.decouple())

# evaluate
subsdict = {Symbol('J'): 0, Symbol("J'"): 0, Symbol('L'): 0, Symbol("L'"): 0, Symbol('l'): 1,
            Symbol("l'"): 1, Symbol('s'): 1, Symbol("s'"): 1, Symbol('j'): 0, Symbol("j'"): 0}

print(composite.evaluate(subsdict))


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
