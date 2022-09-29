from __future__ import annotations
from sympy import KroneckerDelta
from typing import Union, Optional, List
from abc import ABC, abstractmethod
import copy

from tensor_algebra import jsc
from quantum_states import StateInterface
from tensor_algebra import TensorAlgebra
from tensor_operator import TensorOperator, TensorOperatorComposite
from symbolic_wigner import Symbolic6j, Symbolic9j, SymbolicWigner


class MatrixElementInterface(ABC):

    @abstractmethod
    def decouple(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def evaluate(self, symbolic_replace_dict):
        pass


class ReducedMatrixElementComposite(MatrixElementInterface):

    def __init__(self, reduced_me_a: Union[List[ReducedMatrixElement], ReducedMatrixElement],
                 reduced_me_b: Union[List[ReducedMatrixElement], ReducedMatrixElement],
                 factor):
        if isinstance(reduced_me_a, list):
            self._reduced_me_a = reduced_me_a
            self._reduced_me_b = reduced_me_b
            self._factor = factor
        else:
            self._reduced_me_a = [reduced_me_a]
            self._reduced_me_b = [reduced_me_b]
            self._factor = [factor]

    @property
    def reduced_matrix_element_a(self):
        return self._reduced_me_a

    @property
    def reduced_matrix_element_b(self):
        return self._reduced_me_b

    @property
    def factor(self):
        return self._factor

    @property
    def children(self):
        return zip(self.factor, self.reduced_matrix_element_a, self.reduced_matrix_element_b)

    def __str__(self):
        content = [f'{x[0]} * {x[1]}{x[2]}' for x in self.children]
        return ' + '.join(content)

    def __eq__(self, other: ReducedMatrixElementComposite) -> bool:
        if self.children == other.children:
            return True
        else:
            return False

    def __neg__(self, other: ReducedMatrixElementComposite) -> bool:
        return not self.__eq__(other)

    def _full_composite_decouple(self) -> ReducedMatrixElementComposite:
        """
        Fully decouples a ReducedMatrixElementComposite if possible. The decoupling is done inplace. The object is
        returned afterwards. The decoupling happens recursively, thus, it is capable to handle arbitrary structures.
        :return: ReducedMatrixElementComposite
        """
        changed = True
        while changed:
            changed = False
            for _, me_a, me_b in self.children:
                me_a_copy = copy.deepcopy(me_a)
                me_b_copy = copy.deepcopy(me_b)
                me_a.decouple()
                me_b.decouple()
                if me_a != me_a_copy or me_b != me_b_copy:
                    changed = True
        return self

    def decouple(self) -> ReducedMatrixElementComposite:
        """
        This class already contains matrix elements that have been decoupled at least once. In some situations,
        it is possible to decouple the operators further. E.g., the structure
        <j'(j_ab'(j_a'j_b')j_c')| {{A x B} x C} |j(j_ab(j_aj_b)j_c)>
        decoupled once reads
        <j_ab'(j_a'j_b')|| {A x B} ||j_ab(j_aj_b)> <j_c'|| C ||j_c>.
        This can be decoupled again to
        <j_a'|| A ||j_a> <j_b')|| B ||j_b> <j_c'|| C ||j_c>.
        :return: ReducedMatrixElementComposite
        """
        new_me_a = []
        new_me_b = []
        for _, me_a, me_b in self.children:
            new_composite_a = me_a.decouple()
            new_composite_b = me_b.decouple()

            if new_composite_a:
                new_me_a.append(new_composite_a._full_composite_decouple())
            else:
                new_me_a.append(me_a)

            if new_composite_b:
                new_me_b.append(new_composite_b._full_composite_decouple())
            else:
                new_me_b.append(me_b)

        self._reduced_me_a = new_me_a
        self._reduced_me_b = new_me_b
        return self

    def append(self, other: ReducedMatrixElementComposite) -> None:
        self._reduced_me_a.extend(other.reduced_matrix_element_a)
        self._reduced_me_b.extend(other.reduced_matrix_element_b)
        self._factor.extend(other.factor)

    def evaluate(self, symbolic_replace_dict):
        self.decouple()
        ret_val = None
        for factor, me_a, me_b in self.children:
            if isinstance(factor, Symbol):
                term = factor.subs(symbolic_replace_dict)
            elif isinstance(factor, SymbolicWigner):
                term = factor.evaluate(symbolic_replace_dict)
            else:
                term = factor
            term *= me_a.evaluate(symbolic_replace_dict)
            term *= me_b.evaluate(symbolic_replace_dict)
            if ret_val:
                ret_val += term
            else:
                ret_val = term
        return ret_val


class BasicMatrixElementLeafInterface(MatrixElementInterface):

    def __init__(self, bra_state: StateInterface, ket_state: StateInterface,
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
            return " + ".join([f"{op.factor} * {self._state_representation(op.to_expression_no_factor())}"
                               if op.factor != 1 else f"{self._state_representation(op.to_expression_no_factor())}"
                               for op in self.operator.children])
        else:
            factor = self.operator.factor
            if factor != 1:
                output = f"{factor} * {self._state_representation(self.operator.to_expression_no_factor())}"
            else:
                output = f"{self._state_representation(self.operator.to_expression_no_factor())}"
            return output

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _state_representation(self, operator):
        pass

    @abstractmethod
    def _basic_decouple(self, operator):
        pass

    def decouple(self) -> Optional[ReducedMatrixElementComposite]:
        if isinstance(self.operator, TensorOperatorComposite):
            red_me_composite = ReducedMatrixElementComposite([], [], [])
            for op in self.operator.children:
                additional_red_me_composite = self._basic_decouple(op)
                if additional_red_me_composite:
                    red_me_composite.append(additional_red_me_composite)
            return red_me_composite
        else:
            return self._basic_decouple(self.operator)

    def evaluate(self, symbol_replace_dict):
        if isinstance(self.factor, Symbol):
            ret_val = self.factor.subs(symbol_replace_dict)
        else:
            ret_val = self.factor
        if isinstance(self, ReducedMatrixElement):
            separator = "||"
        else:
            separator = "|"
        if self.operator.factor != 1:
            ret_val *= self.operator.factor
        me = Symbol(f"<{self.bra.evaluate(symbol_replace_dict)}{separator}{self.operator.to_expression_no_factor()}"
                    f"{separator}{self.ket.evaluate(symbol_replace_dict)}>")
        return ret_val * me
        #red_me_composite = self.decouple()
        #return red_me_composite.evaluate(symbol_replace_dict)


class MatrixElement(BasicMatrixElementLeafInterface):

    def _state_representation(self, operator):
        return f"<{str(self.bra)[1:-1]}{self.bra.anuglar_quantum_projection}|{operator}{str(self.ket)[:-1]}" \
               + f"{self.ket.anuglar_quantum_projection}>"

    def _basic_decouple(self, operator) -> Optional[ReducedMatrixElementComposite]:
        if not self.bra.substructure or not self.ket.substructure:
            print('[INFO] There is nothing to decouple')
            return None

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
        return ReducedMatrixElementComposite(reduced_matrix_element_a, reduced_matrix_element_b, factor)


class ReducedMatrixElement(BasicMatrixElementLeafInterface):

    def __init__(self, bra_state: StateInterface, ket_state: StateInterface, operator: Union[TensorOperator, TensorOperatorComposite]):
        super().__init__(bra_state, ket_state, operator, factor=1, recouple=False)
        self._value = None

    def __eq__(self, other: ReducedMatrixElement) -> bool:
        if self.bra == other.bra and self.ket == other.ket and self.operator == other.operator:
            return True
        else:
            return False

    def __neg__(self, other: ReducedMatrixElement) -> bool:
        return not self.__eq__(other)

    @property
    def value(self):
        if not self._value:
            print(f"[WARNING] The value of the reduced matrix element {self} has not been set.")
            return self._state_representation(self.operator)
        return self._value

    @value.setter
    def value(self, other):
        self._value = other

    def _state_representation(self, operator):
        return f"<{str(self.bra)[1:-1]}||{operator}|{self.ket}"

    def _basic_decouple(self, operator) -> Optional[ReducedMatrixElementComposite]:
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
                return ReducedMatrixElementComposite(reduced_matrix_element_a, reduced_matrix_element_b,
                                                     new_factor * self.factor)
        print(f'[INFO] Further decoupling not possible for reduced matrix element {self}')
        return None


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
    composite = me.decouple()
    print(composite)

    # double decoupling test
    L = BasicState(Symbol('L'), Symbol('P'))
    Lp = BasicState(Symbol("L'"), Symbol("P'"))
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



