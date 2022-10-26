import unittest
from sympy import Symbol, symbols

from src.tensor_space import TensorSpace
from src.quantum_states import BasicState

space_a = TensorSpace("space_a", 0)
space_b = TensorSpace("space_b", 1)
space_c = TensorSpace("space_c", 2)

state_a = BasicState(angular_quantum_number=Symbol("a"), space=space_a, other_quantum_number=Symbol("x"))
state_b = BasicState(angular_quantum_number=Symbol("b"), space=space_b, other_quantum_number=Symbol("y"))
state_c = BasicState(angular_quantum_number=Symbol("c"), space=space_c)


class TestQuantumStates(unittest.TestCase):
    def test_basic_state_bulk(self):
        self.assertEqual("|xa>", str(state_a))
        self.assertEqual(space_a, state_a.space)
        self.assertEqual(Symbol("a"), state_a.angular_quantum_number)
        self.assertEqual(Symbol("x"), state_a.other_quantum_number)
        self.assertEqual(None, state_c.other_quantum_number)
        self.assertEqual(None, state_a.substructure)

    def test_coupled_state_bulk(self):
        coupled_ab = state_a.couple(state_b, Symbol("w"))
        self.assertEqual("|xyw(ab)>", str(coupled_ab))
        self.assertEqual(space_a + space_b, coupled_ab.space)
        self.assertEqual(Symbol("w"), coupled_ab.angular_quantum_number)
        self.assertEqual(Symbol("xy"), coupled_ab.other_quantum_number)
        self.assertEqual([state_a, state_b], coupled_ab.substructure)

    def test_coupled_ordering(self):
        right_order = state_a.couple(state_b, Symbol("w"))
        other_order = state_b.couple(state_a, Symbol("w"))
        self.assertEqual(right_order, other_order)

        right_order = state_a.couple(state_b, Symbol("w")).couple(state_c, Symbol("v"))
        other_order = state_c.couple((state_a.couple(state_b, Symbol("w"))), Symbol("v"))
        self.assertEqual(right_order, other_order)

    def test_substructure(self):
        other_order = state_c.couple((state_a.couple(state_b, Symbol("w"))), Symbol("v"))
        self.assertEqual("[|xyw(ab)>, |c>]", str(other_order.substructure))

    def test_evaluate(self):
        other_order = state_c.couple((state_a.couple(state_b, Symbol("w"))), Symbol("v"))
        replace_dict = dict(zip(symbols(["w", "a", "b"]), [1, 2, 1]))
        result = other_order.evaluate(replace_dict)
        self.assertEqual("v(1(x2y1)c)", result)  # Note the other quantum numbers are always next to their actual state
        # This function is usually only called once a state has been completely decoupled


if __name__ == "__main__":
    unittest.main()
