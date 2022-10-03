from __future__ import annotations
from abc import ABC, abstractmethod
from sympy import Symbol
from tensor_space import TensorSpace
from typing import Optional, List


class StateInterface(ABC):

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
        pass

    @abstractmethod
    def evaluate(self, symbolic_replace_dict):
        pass

    def couple(self, other: StateInterface, new_quantum_number: Symbol) -> CoupledState:
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
        if other.other_quantum_number:
            new_other_quantum_number = Symbol(f"{new_other_quantum_number}{second.other_quantum_number}")
        if not new_other_quantum_number:
            new_other_quantum_number = None
        return CoupledState(new_quantum_number, new_space, new_other_quantum_number, new_factor, [first, second])


class CoupledState(StateInterface):

    def __init__(self, angular_quantum_number: Symbol, space: TensorSpace,
                 other_quantum_number: Optional[Symbol] = None, factor=1, substructure: List[StateInterface] = None):
        if not substructure:
            raise AttributeError('[ERROR] A coupled state must have a substructure.')
        super().__init__(angular_quantum_number, space, other_quantum_number, factor, substructure)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}" \
               f"({self.substructure[0].representation()}{self.substructure[1].representation()})"

    def evaluate(self, symbolic_replace_dict):
        return f"{self.angular_quantum_number.subs(symbolic_replace_dict)}" \
               f"({self.substructure[0].evaluate(symbolic_replace_dict)}" \
               f"{self.substructure[1].evaluate(symbolic_replace_dict)})"


class BasicState(StateInterface):

    def __init__(self, angular_quantum_number: Symbol, space: TensorSpace,
                 other_quantum_number: Optional[Symbol] = None, factor=1):
        super().__init__(angular_quantum_number, space, other_quantum_number, factor, substructure=None)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}"

    def evaluate(self, symbolic_replace_dict):
        if self.other_quantum_number:
            other = self.other_quantum_number.subs(symbolic_replace_dict)
        else:
            other = ""
        return f"{other}{self.angular_quantum_number.subs(symbolic_replace_dict)}"
