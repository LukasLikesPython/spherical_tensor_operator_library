import unittest
from sympy.physics.wigner import wigner_6j, wigner_9j
from sympy import symbols
from itertools import combinations
import sys

sys.path.append('../stol/')

from symbolic_wigner import Symbolic6j, Symbolic9j

J_MAX_VALUE = 2  # WARNING, the duration scales as (n + 1)**6 for the Six-J test and (n + 1)**9 for the Nine-J test.
J_MAX_VALUE_COMBINATION = 1  # WARNING, the duration scales as (n + 1)**15


class TestSymbolicWigner(unittest.TestCase):
    def test_symbolic_6j(self):
        symbol_int = 6
        symbol_list = symbols(["j1", "j2", "j3", "j4", "j5", "j6"])
        symb_6j = Symbolic6j(*symbol_list)
        j_range = list(range(J_MAX_VALUE + 1))
        combs = sorted(list(set(combinations(j_range * symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6 in combs:
            six_j_value = wigner_6j(j1, j2, j3, j4, j5, j6)
            symb_six_j_value = symb_6j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6])))
            self.assertEqual(six_j_value, symb_six_j_value)

    def test_symbolic_9j(self):
        symbol_int = 9
        symbol_list = symbols(["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9"])
        symb_9j = Symbolic9j(*symbol_list)
        j_range = list(range(J_MAX_VALUE + 1))
        combs = sorted(list(set(combinations(j_range * symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6, j7, j8, j9 in combs:
            nine_j_value = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
            symb_nine_j_value = symb_9j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9])))
            self.assertEqual(nine_j_value, symb_nine_j_value)

    def test_symbolic_multiplication(self):
        symbol_int = 15
        symbol_list_9 = symbols(["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9"])
        symb_9j = Symbolic9j(*symbol_list_9)
        symbol_list_6 = symbols(["j10", "j11", "j12", "j13", "j14", "j15"])
        symb_6j = Symbolic6j(*symbol_list_6)
        symb = symb_6j * symb_9j
        symbol_list = symbol_list_9 + symbol_list_6
        j_range = list(range(J_MAX_VALUE_COMBINATION + 1))
        combs = sorted(list(set(combinations(j_range * symbol_int, symbol_int))))
        for j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15 in combs:
            if (
                    not (abs(j1 - j2) <= j3 <= j1 + j2)
                    or not (abs(j4 - j5) <= j6 <= j4 + j5)
                    or not (abs(j7 - j8) <= j9 <= j7 + j8)
                    or not (abs(j1 - j4) <= j7 <= j1 + j4)
                    or not (abs(j2 - j5) <= j8 <= j2 + j5)
                    or not (abs(j3 - j6) <= j9 <= j3 + j6)
                    or not (abs(j11 - j12) <= j10 <= j11 + j12)
                    or not (abs(j14 - j15) <= j10 <= j14 + j15)
                    or not (abs(j11 - j15) <= j13 <= j11 + j15)
                    or not (abs(j12 - j14) <= j13 <= j12 + j14)
            ):
                continue  # Symmetries state those elements are zero. We tested the whole set already above.
            symbolic_comb = symb.evaluate(
                dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]))
            )
            wigner_comb = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9) * wigner_6j(j10, j11, j12, j13, j14, j15)
            self.assertEqual(wigner_comb, symbolic_comb)

    def test_mult_6j(self):
        symbol_list = symbols(["j1", "j2", "j3", "j4", "j5", "j6"])
        symb_6j = Symbolic6j(*symbol_list)
        mult_symb_6j = symb_6j * 2
        j_range = [[1, 1, 2, 1, 1, 2]]
        for j1, j2, j3, j4, j5, j6 in j_range:
            six_j_value = wigner_6j(j1, j2, j3, j4, j5, j6)
            symb_six_j_value = symb_6j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6])))
            mult_symb_six_j_value = mult_symb_6j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6])))
            self.assertEqual(six_j_value, symb_six_j_value)
            self.assertEqual(six_j_value * 2, mult_symb_six_j_value)

    def test_mult_9j(self):
        symbol_list = symbols(["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9"])
        symb_9j = Symbolic9j(*symbol_list)
        mult_symb_9j = symb_9j * 2
        j_range = [[1, 1, 2, 1, 1, 2, 1, 1, 0]]
        for j1, j2, j3, j4, j5, j6, j7, j8, j9 in j_range:
            nine_j_value = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
            symb_nine_j_value = symb_9j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9])))
            mult_symb_nine_j_value = mult_symb_9j.evaluate(dict(zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9])))
            self.assertEqual(nine_j_value, symb_nine_j_value)
            self.assertEqual(nine_j_value * 2, mult_symb_nine_j_value)

    def test_mult_composite(self):
        symbol_list_6_1 = symbols(["j1", "j2", "j3", "j4", "j5", "j6"])
        symbol_list_6_2 = symbols(["j7", "j8", "j9", "j10", "j11", "j12"])
        symbol_list_6_3 = symbols(["j13", "j14", "j15", "j16", "j17", "j18"])
        symb = Symbolic6j(*symbol_list_6_1) * Symbolic6j(*symbol_list_6_2) * Symbolic6j(*symbol_list_6_3) * 2
        symbol_list = symbol_list_6_1 + symbol_list_6_2 + symbol_list_6_3
        j_range = [[1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 0, 3, 2, 1, 3, 2, 1]]
        for j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15, j16, j17, j18 in j_range:
            symbolic_comb = symb.evaluate(
                dict(
                    zip(symbol_list, [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15, j16, j17, j18])
                )
            )
            wigner_comb = (
                    wigner_6j(j1, j2, j3, j4, j5, j6)
                    * wigner_6j(j7, j8, j9, j10, j11, j12)
                    * wigner_6j(j13, j14, j15, j16, j17, j18)
                    * 2
            )
            self.assertEqual(wigner_comb, symbolic_comb)


if __name__ == "__main__":
    unittest.main()
