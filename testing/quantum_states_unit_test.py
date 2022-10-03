import unittest
from sympy import Symbol

from tensor_space import TensorSpace
from quantum_states import BasicState


space_a = TensorSpace('space_a', 0)
space_b = TensorSpace('space_b', 1)
space_c = TensorSpace('space_c', 2)

state_a = BasicState(angular_quantum_number=Symbol('a'), space=space_a, other_quantum_number=Symbol('x'))
state_b = BasicState(angular_quantum_number=Symbol('b'), space=space_b, other_quantum_number=Symbol('y'))
state_a_2 = BasicState(angular_quantum_number=Symbol('a2'), space=space_a)
state_c = BasicState(angular_quantum_number=Symbol('c'), space=space_c)


class TestQuantumStates(unittest.TestCase):

    def test_basic_state_bulk(self):
        self.assertEqual('|xa>', str(state_a))
        self.assertEqual(space_a, state_a.space)
        self.assertEqual(Symbol('a'), state_a.angular_quantum_number)
        self.assertEqual(Symbol('x'), state_a.other_quantum_number)
        self.assertEqual(None, state_c.other_quantum_number)
        self.assertEqual(None, state_a.substructure)

    def test_coupled_state_bulk(self):
        coupled_ab = state_a.couple(state_b, Symbol("w"))
        self.assertEqual('|xyw(ab)>', str(coupled_ab))
        self.assertEqual(space_a + space_b, coupled_ab.space)
        self.assertEqual(Symbol('w'), coupled_ab.angular_quantum_number)
        self.assertEqual(Symbol('xy'), coupled_ab.other_quantum_number)
        self.assertEqual([state_a, state_b], coupled_ab.substructure)

    """
    sig1 = BasicState(Symbol('\u03C3\u2081'), spin_space)
    sig2 = BasicState(Symbol('\u03C3\u2082'), spin_space)

    print(sig1, sig2)
    s = sig1.couple(sig2, Symbol('s'))
    print(s, s.space)
    l = BasicState(Symbol('l'), rel_space)
    j = l.couple(s, Symbol('j'))
    print(j, j.space)
    j = s.couple(l, Symbol('j'))
    print(j, j.space)
    print(j.substructure[0], j.substructure[1])
    """

if __name__ == '__main__':
    unittest.main()
