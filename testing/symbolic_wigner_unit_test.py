import unittest
from sympy.physics.wigner import wigner_6j, wigner_9j
from sympy import symbols
from symbolic_wigner import Symbolic6j, Symbolic9j
from itertools import combinations

J_MAX_VALUE = 2  # WARNING, the duration scales as n**6 for the Six-J test and n**9 for the Nine-J test.


class TestSymbolicWigner(unittest.TestCase):

    def test_symbolic_6j(self):
        symbol_int = 6
        symbol_list = symbols(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        symb_6j = Symbolic6j(*symbol_list)
        j_range = list(range(J_MAX_VALUE + 1))
        combs = sorted(list(set(combinations(j_range*symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6 in combs:
            six_j_value = wigner_6j(j1, j2, j3, j4, j5, j6)
            symb_six_j_value = symb_6j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6])))
            self.assertEqual(symb_six_j_value, six_j_value)

    def test_symbolic_9j(self):
        symbol_int = 9
        symbol_list = symbols(['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9'])
        symb_9j = Symbolic9j(*symbol_list)
        j_range = list(range(J_MAX_VALUE + 1))
        combs = sorted(list(set(combinations(j_range*symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6, j7, j8, j9 in combs:
            nine_j_value = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
            symb_nine_j_value = symb_9j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9])))
            self.assertEqual(symb_nine_j_value, nine_j_value)

    def test_symbolic_multiplication(self):
        symbol_int = 15
        symbol_list_9 = symbols(['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9'])
        symb_9j = Symbolic9j(*symbol_list_9)
        symbol_list_6 = symbols(['j10', 'j11', 'j12', 'j13', 'j14', 'j15'])
        symb_6j = Symbolic6j(*symbol_list_6)
        symb = symb_6j * symb_9j
        symbol_list = symbol_list_9.extend(symbol_list_6)
        j_range = list(range(J_MAX_VALUE + 1))
        combs = sorted(list(set(combinations(j_range * symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15 in combs:
            symbolic_comb = symb.evaluate(dict(zip(symbol_list,
                                                   [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15])))
            if not symbolic_comb:
                continue  # The symbolic_comb is cached. This helps to reduce calculation time for cases that are zero
            wigner_comb = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9) * wigner_6j(j10, j11, j12, j13, j14, j15)
            self.assertEqual(symbolic_comb, wigner_comb)


if __name__ == '__main__':
    unittest.main()
