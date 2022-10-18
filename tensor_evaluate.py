from __future__ import annotations
from sympy import KroneckerDelta, Symbol
from typing import Union, Optional, List
from abc import ABC, abstractmethod
import copy
import logging

from tensor_algebra import jsc
from quantum_states import StateInterface
from tensor_algebra import TensorAlgebra
from tensor_operator import TensorOperator, TensorOperatorComposite, TensorOperatorInterface
from symbolic_wigner import Symbolic6j, Symbolic9j, SymbolicWigner, factor_eval

logging.basicConfig(level=logging.WARNING)


class StateMismatchError(Exception):
    """Raised when the space of the final state does not match that of the initial one or if the operator acts on
    different spaces than the states provide."""
    pass


class RankMismatchError(Exception):
    """Raised when the rank of operators does not match."""
    pass


class MatrixElementInterface(ABC):
    """
    Interface for Leaf and Composite MatrixElement classes.
    """

    @abstractmethod
    def decouple(self):
        """
        Decouple tensor operators and the tensor space. The result is a reduced matrix element of operators, i.e.,
        a product of operators that act in space x encompassed by the state for this space. E.g.
        <j'(j1'j2')| {O1 x O2} |j(j1j2)> -> <j1'||O1||j1> * <j2'||O2||j2> (Note the example is simplified, it does not
        consider ranks of the tensor operators). It can also return a sum of reduced matrix elements. Some Reduced
        Matrix Elements can be decoupled further.
        :return: ReducedMatrixElement
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of Matrix Elements.
        :return: String
        """
        pass

    @abstractmethod
    def evaluate(self, symbolic_replace_dict: dict):
        """
        Evaluate the matrix elements by replacing all or some of the symbols in the objects.
        :param symbolic_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: Evaluation of the expression.
        """
        pass


class ReducedMatrixElementComposite(MatrixElementInterface):
    """
    A collection of ReducedMatrixElement objects. The class contains a list for the reduced_matrix_element_a and
    reduced_matrix_element_b (short rme_a/b). The lists are ordered and contain a MatrixElementInterface list.
    Elements between list a and b with the same index are multiplied, while this produced is added between different
    indexes, e.g., rme_a = [me_a1, me_a2, me_a3, ...], rme_b = [me_b1, me_b2, me_b3, ...] means the actual operator
    structure is me_a1 * me_b1 + me_a2 * me_b2 + me_a3 * me_b3 + ...
    Note that the lists have the same length by construction. The factor list works similarly.

    While it is possible to initialize this element directly, we recommend going via MatrixElement objects and use
    their decouple function.
    """

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
        """
        :return: iterable object that contains the factor as the first element, the reduced_matrix_element_a as the
        second and the reduced_matrix_element_b as the last.
        """
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
        """
        Append another ReduceMatrixElementComposite to the existing one.
        :param other: ReducedMatrixElementComposite object.
        :return: This action happens in place and does not return anything
        """
        self._reduced_me_a.extend(other.reduced_matrix_element_a)
        self._reduced_me_b.extend(other.reduced_matrix_element_b)
        self._factor.extend(other.factor)

    def me_evaluate(self, symbolic_replace_dict: dict):
        """
        Auxilliary function made so that the same structure as MatrixElements and ReducedMatrixElements is met.
        This function just uses the evaluate function.
        :param symbolic_replace_dict:
        :return: The evaluated expression where symbolic components are fully or partially replaced by values.
        """
        return self.evaluate(symbolic_replace_dict)

    def evaluate(self, symbolic_replace_dict: dict):
        """
        Recursively go through all children and call their evaluate function to replace some or all of the symbolic
        contributions in the expression.
        :param symbolic_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: The evaluated expression where symbolic components are fully or partially replaced by values.
        """
        self.decouple()
        ret_val = None
        for factor, me_a, me_b in self.children:
            if isinstance(factor, SymbolicWigner):
                term = factor.evaluate(symbolic_replace_dict)
            else:
                term = factor_eval(factor, symbolic_replace_dict)
            term *= me_a.me_evaluate(symbolic_replace_dict)
            term *= me_b.me_evaluate(symbolic_replace_dict)
            if ret_val:
                ret_val += term
            else:
                ret_val = term
        return ret_val


class BasicMatrixElementLeafInterface(MatrixElementInterface):
    """
    Inferface for ReducedMatrixElement and MatrixElement objects.
    """

    def __init__(self, bra_state: StateInterface, ket_state: StateInterface,
                 operator: Union[TensorOperator, TensorOperatorComposite, None], factor: Union[int, float, Symbol] = 1,
                 recouple=True):
        self._bra = bra_state
        self._ket = ket_state

        if bra_state.space != ket_state.space:
            raise StateMismatchError('[ERROR] Final and initial states do not have the same space configuration.')
        state_basic_spaces = bra_state.space.get_flat_basic_states()
        operator_basic_spaces = operator.space.get_flat_basic_states()
        if state_basic_spaces != operator_basic_spaces:
            logging.debug("Operator spaces do not match State spaces")
            # Case A: The operator does not contain contributions for all spaces in the state -> Add unit operators
            for sbs in state_basic_spaces:
                if sbs not in operator_basic_spaces:
                    unit_operator = TensorOperator(rank=0, factor=1, space=sbs, symbol=Symbol(f'I_{sbs.name}'))
                    operator = operator.couple(unit_operator, operator.rank, 1)
                    logging.debug(f"Adding the unit operator {unit_operator}")
            # Case B: The operator acts on spaces which are not present in the state -> Throw an error
            for obs in operator_basic_spaces:
                if obs not in state_basic_spaces:
                    raise StateMismatchError('[ERROR] The operator acts on spaces that are not present in the states.')

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

    def __str__(self) -> str:
        """
        Visual representation of the object. See _state_representation for more information.

        :return: String
        """
        if isinstance(self.operator, TensorOperatorComposite):
            return " + ".join([f"{op.factor} * {self._state_representation(op)}"
                               if op.factor != 1 else f"{self._state_representation(op)}"
                               for op in self.operator.children])
        else:
            factor = self.operator.factor
            if factor != 1:
                output = f"{factor} * {self._state_representation(self.operator)}"
            else:
                output = f"{self._state_representation(self.operator)}"
            return output

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _state_representation(self, operator: TensorOperatorInterface) -> str:
        """
        Auxiliary function that is used in the __str__ function. The function is implemented differently between
        MatrixElement objects and ReducedMatrixElement objects. It also takes care of a bra and ket notation, as the
        default implementation in State objects is the ket notation. The state representation takes an operator as
        input to keep it more flexible.

        :param operator: TensorOperatorInterface object
        :return: A representative String.
        """
        pass

    @abstractmethod
    def _basic_decouple(self, operator: TensorOperatorInterface):
        """
        Auxiliary decouple formalism that helps to distinguish between MatrixElements and ReducedMatrixElements. It
        also gives more flexibility, as it will decouple an arbitrary (part of the) operator.

        :param operator: TensorOperatorInterface object
        :return: ReducedMatrixElementComposite object.
        """
        pass

    def decouple(self) -> Optional[ReducedMatrixElementComposite]:
        """
        Decouples the given matrix element one level. A different treatment happens for MatrixElement objects and
        ReducedMatrixElement objects via the _basic_decouple function. It is possible that the result can be recoupled
        further. Use the function full_decouple to get the final expression
        """
        if isinstance(self.operator, TensorOperatorComposite):
            red_me_composite = ReducedMatrixElementComposite([], [], [])
            for op in self.operator.children:
                additional_red_me_composite = self._basic_decouple(op)
                if additional_red_me_composite:
                    red_me_composite.append(additional_red_me_composite)
            return red_me_composite
        else:
            return self._basic_decouple(self.operator)

    def full_decouple(self) -> Optional[ReducedMatrixElementComposite]:
        """
        Decouples the given matrix element. A different treatment happens for MatrixElement objects and
        ReducedMatrixElement objects via the _basic_decouple function.
        """
        composite = self.decouple()
        return composite.decouple()  # Make use of the full decoupling routine in composites

    def me_evaluate(self, symbol_replace_dict: dict):
        """
        Replaces all or some of the symbolic expressions with values and returns the result. This function is called
        within the Composite evaluation routine.

        :param symbol_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: The evaluated expression where symbolic components are fully or partially replaced by values.
        """
        if isinstance(self.factor, Symbol):
            ret_val = self.factor.subs(symbol_replace_dict)
        else:
            ret_val = self.factor
        if isinstance(self, ReducedMatrixElement):
            separator = "||"
        else:
            separator = "|"
        me = Symbol(f"<{self.bra.evaluate(symbol_replace_dict)}{separator}{self.operator.to_expression_no_factor()}"
                    f"{separator}{self.ket.evaluate(symbol_replace_dict)}>")
        return me

    def evaluate(self, symbol_replace_dict: dict):
        composite = self.full_decouple()
        return composite.evaluate(symbol_replace_dict)


class MatrixElement(BasicMatrixElementLeafInterface):
    """
    Allows the user to create a matrix element object, containing a bra state, ket state and an operator.
    We recommend to use this class to define objects and create other class instances by using the recouple function.
    """

    def _state_representation(self, operator: TensorOperatorInterface) -> str:
        """
        Auxiliary function that is used in the __str__ function. For completeness, we also provide the projection of the
        angular quantum number. It is, however, not used in the evaluation process in this work. One needs it during the
        application of the Wigner-Eckart theorem.

        :param operator: TensorOperatorInterface object
        :return: A representative String.
        """
        ket_angular_quantum_projection = f"m_{self.ket.angular_quantum_number}"
        bra_angular_quantum_projection = f"m_{self.bra.angular_quantum_number}"
        return f"<{str(self.bra)[1:-1]}{bra_angular_quantum_projection}|{operator.to_expression_no_factor()}" \
               f"{str(self.ket)[:-1]}" + f"{ket_angular_quantum_projection}>"

    def _basic_decouple(self, operator: TensorOperator) -> Optional[ReducedMatrixElementComposite]:
        """
        The function is called for TensorOperators. It decouples potential parts of the complete self.operator.

        :param operator: A part of the self.Operator. It is broken down to TensorOperator objects.
        :return: ReducedMatrixElementComposite or None
        """
        if not self.bra.substructure or not self.ket.substructure:
            logging.info('There is nothing to decouple')
            return None

        tensor_a, tensor_b = operator.substructure
        if tensor_a.rank != tensor_b.rank:
            raise RankMismatchError('[ERROR] Ranks of sub-operators must match')

        rank = tensor_a.rank

        bra_a, bra_b = self.bra.substructure
        ket_a, ket_b = self.ket.substructure
        bra_j = self.bra.angular_quantum_number
        ket_j = self.ket.angular_quantum_number
        bra_j_a = bra_a.angular_quantum_number
        bra_j_b = bra_b.angular_quantum_number
        ket_j_a = ket_a.angular_quantum_number
        ket_j_b = ket_b.angular_quantum_number

        factor = operator.factor * pow(-1, bra_j + bra_j_b + ket_j_a + rank) / jsc(rank) * KroneckerDelta(bra_j, ket_j)\
                 * Symbolic6j(bra_j_a, bra_j_b, bra_j, ket_j_b, ket_j_a, rank)
        reduced_matrix_element_a = ReducedMatrixElement(bra_a, ket_a, tensor_a)
        reduced_matrix_element_b = ReducedMatrixElement(bra_b, ket_b, tensor_b)
        return ReducedMatrixElementComposite(reduced_matrix_element_a, reduced_matrix_element_b, factor)


class ReducedMatrixElement(BasicMatrixElementLeafInterface):
    """
    A class for reduced matrix elements. While it is possible, we do not recommend to initialize objects of this class
    directly. We recommend users to initialize MatrixElement objects and use the decouple function to obtain reduced
    matrix elements. Reduced matrix elements can be calculated using the Wigner-Eckart theorem.
    """

    def __init__(self, bra_state: StateInterface, ket_state: StateInterface,
                 operator: Union[TensorOperator, TensorOperatorComposite]):
        super().__init__(bra_state, ket_state, operator, factor=1, recouple=False)
        self._value = None

    def __eq__(self, other: ReducedMatrixElement) -> bool:
        if self.bra == other.bra and self.ket == other.ket and self.operator == other.operator:
            return True
        else:
            return False

    def __neg__(self, other: ReducedMatrixElement) -> bool:
        return not self.__eq__(other)

    def _state_representation(self, operator):
        return f"<{str(self.bra)[1:-1]}||{operator.to_expression_no_factor()}|{self.ket}"

    def _basic_decouple(self, operator: TensorOperator) -> Optional[ReducedMatrixElementComposite]:
        """
        In some cases, it is possible to decouple reduced matrix elements even further (e.g., when there are three
        operators from distinct spaces). This method decouples those cases or returns None if no further recoupling is
        possible.

        :param operator: TensorOperator object
        :return: ReducedMatrixElementComposite or None
        """
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
        logging.info(f'Further decoupling not possible for reduced matrix element {self}')
        return None
