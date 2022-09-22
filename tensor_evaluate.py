from __future__ import annotations
from sympy.physics.wigner import wigner_6j, wigner_9j
from sympy import KroneckerDelta
from typing import Union
from abc import ABC, abstractmethod

from tensor_algebra import jsc
from quantum_states import State
from tensor_algebra import TensorAlgebra
from tensor_operator import TensorOperator, TensorOperatorComposite
from symbolic_wigner import Symbolic6j, Symbolic9j, SymbolicWigner


class MatrixElementInterface(ABC):

    def __init__(self, bra_state: State, ket_state: State,
                 operator: Union[TensorOperator, TensorOperatorComposite, None], factor=1, recouple=True):
        self._bra = bra_state
        self._ket = ket_state
        if recouple:
            self._operator = TensorAlgebra.recouple(operator)  # Simplifies the operator structure
        else:
            self._operator = operator
        if operator:
            self._factor = factor * operator.factor
        else:
            self._factor = factor

    @property
    def bra(self):
        return self._bra

    @property
    def ket(self):
        return self._ket

    @property
    def operator(self):
        return self._operator

    @property
    def factor(self):
        return self._factor

    def __str__(self):
        if isinstance(self.operator, TensorOperatorComposite):
            return " + ".join([f"{op.factor} * {self._state_representation(op.to_expression_no_factor())}" for op in
                               self.operator.children])
        else:
            return f"{self.operator.factor} * {self._state_representation(self.operator.to_expression_no_factor())}"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _state_representation(self, op):
        pass

    @abstractmethod
    def _basic_decouple(self, operator):
        pass

    def decouple(self):
        if isinstance(self.operator, TensorOperatorComposite):
            factors = []
            matrix_elements_a = []
            matrix_elements_b = []
            for op in self.operator.children:
                factor, me_a, me_b = self._basic_decouple(op)
                factors.append(factor)
                matrix_elements_a.append(me_a)
                matrix_elements_b.append(me_b)
            return factors, matrix_elements_a, matrix_elements_b
        else:
            return self._basic_decouple(self.operator)

    def evaluate(self, symbol_replace_dict):
        # decouple
        factor, reduced_matrix_element_1, reduced_matrix_element_2 = self.decouple()
        while reduced_matrix_element_1:
            break


class MatrixElement(MatrixElementInterface):

    def _state_representation(self, op):
        return f"<{str(self.bra)[1:-1]}{self.bra.anuglar_quantum_projection}|{op}{str(self.ket)[:-1]}" \
               + f"{self.ket.anuglar_quantum_projection}>"

    def _basic_decouple(self, operator):
        if not self.bra.substructure or not self.ket.substructure:
            print('[INFO] There is nothing to decouple')
            return None, None, None

        tensor_a, tensor_b = operator.substructure
        rank = operator.rank

        bra_a, bra_b = self.bra.substructure
        ket_a, ket_b = self.ket.substructure
        bra_j = self.bra.angular_quantum_number
        ket_j = self.ket.angular_quantum_number
        bra_j_a = bra_a.angular_quantum_number
        bra_j_b = bra_b.angular_quantum_number
        ket_j_a = ket_a.angular_quantum_number
        ket_j_b = ket_b.angular_quantum_number

        factor = pow(-1, bra_j + bra_j_b + ket_j_a + rank) / jsc(rank) * KroneckerDelta(bra_j, ket_j) \
                 * Symbolic6j(bra_j_a, bra_j_b, bra_j, ket_j_b, ket_j_a, rank)
        reduced_matrix_element_a = ReducedMatrixElement(bra_a, ket_a, tensor_a)
        reduced_matrix_element_b = ReducedMatrixElement(bra_b, ket_b, tensor_b)
        return factor, reduced_matrix_element_a, reduced_matrix_element_b


class ReducedMatrixElement(MatrixElementInterface):

    def __init__(self, bra_state: State, ket_state: State, operator: Union[TensorOperator, TensorOperatorComposite]):
        super().__init__(bra_state, ket_state, operator, factor=1, recouple=False)
        self._value = None

    @property
    def value(self):
        if not self._value:
            print(f"[WARNING] The value of the reduced matrix element {self} has not been set.")
            return self._state_representation(self.operator)
        return self._value

    @value.setter
    def value(self, other):
        self._value = other

    def _state_representation(self, op):
        return f"<{str(self.bra)[1:-1]}||{op}|{self.ket}"

    def _basic_decouple(self, operator):
        if operator.substructure:
            tensor_a, tensor_b = operator.substructure
            if tensor_a.space != tensor_b.space:
                rank = operator.rank
                a = tensor_a.rank
                b = tensor_b.rank
                bra_a, bra_b = self.bra.substructure
                ket_a, ket_b = self.ket.substructure
                bra_j = self.bra.angular_quantum_number
                ket_j = self.ket.angular_quantum_number
                bra_j_a = bra_a.angular_quantum_number
                bra_j_b = bra_b.angular_quantum_number
                ket_j_a = ket_a.angular_quantum_number
                ket_j_b = ket_b.angular_quantum_number
                new_factor = jsc(bra_j, ket_j, rank) \
                             * Symbolic9j(a, b, rank, bra_j_a, bra_j_b, bra_j, ket_j_a, ket_j_b, ket_j)
                reduced_matrix_element_a = ReducedMatrixElement(bra_a, ket_a, tensor_a)
                reduced_matrix_element_b = ReducedMatrixElement(bra_b, ket_b, tensor_b)
                return new_factor * self.factor, reduced_matrix_element_a, reduced_matrix_element_b
        print(f'[INFO] Further decoupling not possible for reduced matrix element {self}')
        return None, None, None


class ReducedMatrixElementComposite(MatrixElementInterface):

    def __init__(self, bra_state: State, ket_state: State, operator_a, operator_b, factor):
        super().__init__(bra_state, ket_state, None, factor=factor, recouple=False)
        self._operator_a = operator_a
        self._operator_b = operator_b

    @property
    def operator_a(self):
        return self._operator_a

    @property
    def operator_b(self):
        return self._operator_b




if __name__ == "__main__":
    from sympy import Symbol
    from tensor_space import TensorSpace
    from tensor_transformation import TensorFromVectors
    from quantum_states import BasicState

    # spaces
    rel_space = TensorSpace('rel', 0)
    spin_space = TensorSpace('spin', 1)
    cm_space = TensorSpace('cm', 2)

    # states
    s = BasicState(Symbol('s'))
    l = BasicState(Symbol('l'), Symbol('p'))
    ket = l.couple(s, Symbol('j'))
    print(ket, ket.substructure)
    sp = BasicState(Symbol("s'"))
    lp = BasicState(Symbol("l'"), Symbol("p'"))
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
    fac, red_me_a, red_me_b = me.decouple()
    print(fac)
    print(red_me_a)
    print(red_me_b)

