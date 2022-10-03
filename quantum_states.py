from __future__ import annotations
from abc import ABC, abstractmethod
from sympy import Symbol
from tensor_space import TensorSpace
from typing import Optional, List


class StateInterface(ABC):
    """
    Interface for BasicStates and CoupledStates
    """

    def __init__(self, angular_quantum_number: Symbol, space: TensorSpace,
                 other_quantum_number: Optional[Symbol] = None, factor=1,
                 substructure: Optional[List[StateInterface]] = None):
        self._angular_quantum_number = angular_quantum_number
        self._other_quantum_number = other_quantum_number
        self._factor = factor
        self._substructure = substructure
        self._space = space

    def __str__(self):
        if self.other_quantum_number:
            return f"|{self.other_quantum_number}{self.representation()}>"
        else:
            return f"|{self.representation()}>"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: StateInterface) -> bool:
        if self.angular_quantum_number == other.angular_quantum_number \
                and self.other_quantum_number == other.other_quantum_number \
                and self.factor == other.factor \
                and self.substructure == other.substructure:
            return True

    @property
    def angular_quantum_number(self):
        return self._angular_quantum_number

    @property
    def other_quantum_number(self):
        return self._other_quantum_number

    @property
    def substructure(self):
        return self._substructure

    @property
    def factor(self):
        return self._factor

    @property
    def space(self):
        return self._space

    @abstractmethod
    def representation(self) -> str:
        """
        Provide a string representation of each subclass. The representation is used in the __str__ function and can be
        appended for coupled states.
        :return: String
        """
        pass

    @abstractmethod
    def evaluate(self, symbolic_replace_dict: dict) -> str:
        """
        Evaluate all or some of the symbolic constituents of the state and return a string representation of it.
        :param symbolic_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: A String representation of the state, but with evaluated symbols
        """
        pass

    def couple(self, other: StateInterface, new_quantum_number: Symbol) -> CoupledState:
        """
        Couple another state to this state to form a new CoupledState object. During the coupling process, the states'
        constituents are ordered according to the order of their space object.
        :param other: Another state, either BasicState or CoupledState
        :param new_quantum_number: The symbol of the newly formed state
        :return: A CoupledState object
        """
        if self.space > other.space:
            first = other
            second = self
        else:
            first = self
            second = other
        new_factor = first.factor * second.factor
        new_space = first.space + second.space
        new_other_quantum_number = ''
        if first.other_quantum_number:
            new_other_quantum_number = first.other_quantum_number
        if second.other_quantum_number:
            new_other_quantum_number = Symbol(f"{new_other_quantum_number}{second.other_quantum_number}")
        if not new_other_quantum_number:
            new_other_quantum_number = None
        return CoupledState(angular_quantum_number=new_quantum_number,
                            space=new_space,
                            substructure=[first, second],
                            other_quantum_number=new_other_quantum_number,
                            factor=new_factor)


class CoupledState(StateInterface):
    """
    CoupledState objects are a combination of two substates. They can in principle be initialized, but they can also be
    constructed using the couple function on StateInterface objects. Coupled states know their constituents, such that
    we can recouple them when we need it.
    """

    def __init__(self, angular_quantum_number: Symbol, space: TensorSpace, substructure: List[StateInterface],
                 other_quantum_number: Optional[Symbol] = None, factor=1):
        super().__init__(angular_quantum_number, space, other_quantum_number, factor, substructure)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}" \
               f"({self.substructure[0].representation()}{self.substructure[1].representation()})"

    def evaluate(self, symbolic_replace_dict: dict) -> str:
        """
        A coupled state j has consists of two substates j1 and j2. The representation reads |j(j1j2)> if the substates
        are basic states or |j(j1(j1aj1b(...))j2(...)> if the substates are combined states as well. The substates are
        evaluated recursively.
        :param symbolic_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: A String representation of the state, but with evaluated symbols
        """
        return f"{self.angular_quantum_number.subs(symbolic_replace_dict)}" \
               f"({self.substructure[0].evaluate(symbolic_replace_dict)}" \
               f"{self.substructure[1].evaluate(symbolic_replace_dict)})"


class BasicState(StateInterface):
    """
    BasicState objects are states without a substructure. For us, those are the fundamental building blocks in our
    calculation. They mark the target to which we ultimately want to decouple our states and operators.
    """

    def __init__(self, angular_quantum_number: Symbol, space: TensorSpace,
                 other_quantum_number: Optional[Symbol] = None, factor=1):
        super().__init__(angular_quantum_number, space, other_quantum_number, factor, substructure=None)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}"

    def evaluate(self, symbolic_replace_dict: dict) -> str:
        """
        Replace all or some of the symbolic constituents by values according to the symbolic_replace_dict dictionary.
        :param symbolic_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: A String representation of the state, but with evaluated symbols
        """
        if self.other_quantum_number:
            other = self.other_quantum_number.subs(symbolic_replace_dict)
        else:
            other = ""
        return f"{other}{self.angular_quantum_number.subs(symbolic_replace_dict)}"
