from __future__ import annotations
from abc import ABC, abstractmethod
from sympy import Symbol
from sympy.physics.wigner import clebsch_gordan
from typing import Optional, List


# TODO add space to basic states and rearrange coupled states accordingly


class StateInterface(ABC):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Optional[Symbol] = None, factor=1,
                 substructure: Optional[List[StateInterface]] = None, projection: Optional[Symbol] = None):
        self._angular_quantum_number = angular_quantum_number
        self._other_quantum_number = other_quantum_number
        self._factor = factor
        self._substructure = substructure
        self._projection = projection

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
    def anuglar_quantum_projection(self):
        if self._projection is None:
            return Symbol(f"m_{self.angular_quantum_number}")
        else:
            return self._projection

    @property
    def other_quantum_number(self):
        return self._other_quantum_number

    @property
    def substructure(self):
        return self._substructure

    @property
    def factor(self):
        return self._factor

    @abstractmethod
    def representation(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, symbolic_replace_dict):
        pass

    def couple(self, other: StateInterface, new_quantum_number: Symbol) -> CoupledState:
        new_factor = self.factor * other.factor
        new_other_quantum_number = ''
        if self.other_quantum_number:
            new_other_quantum_number = self.other_quantum_number
        if other.other_quantum_number:
            new_other_quantum_number = Symbol(f"{new_other_quantum_number}{other.other_quantum_number}")
        if not new_other_quantum_number:
            new_other_quantum_number = None
        return CoupledState(new_quantum_number, new_other_quantum_number, new_factor, [self, other])


class CoupledState(StateInterface):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Optional[Symbol] = None, factor=1,
                 substructure: List[StateInterface] = None):
        if not substructure:
            raise AttributeError('[ERROR] A coupled state must have a substructure.')
        super().__init__(angular_quantum_number, other_quantum_number, factor, substructure)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}" \
               f"({self.substructure[0].representation()}{self.substructure[1].representation()})"

    def evaluate(self, symbolic_replace_dict):
        return f"{self.angular_quantum_number.subs(symbolic_replace_dict)}" \
               f"({self.substructure[0].evaluate(symbolic_replace_dict)}" \
               f"{self.substructure[1].evaluate(symbolic_replace_dict)})"


class BasicState(StateInterface):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Optional[Symbol] = None, factor=1,
                 substructure=None):
        if substructure:
            raise AttributeError('[ERROR] A basic state cannot have a substructure.')
        super().__init__(angular_quantum_number, other_quantum_number, factor)

    def representation(self) -> str:
        return f"{self.angular_quantum_number}"

    def evaluate(self, symbolic_replace_dict):
        if self.other_quantum_number:
            other = self.other_quantum_number.subs(symbolic_replace_dict)
        else:
            other = ""
        return f"{other}{self.angular_quantum_number.subs(symbolic_replace_dict)}"


if __name__ == "__main__":
    sig1 = BasicState(Symbol('\u03C3\u2081'))
    sig2 = BasicState(Symbol('\u03C3\u2082'))
    print(sig1, sig2)
    s = sig1.couple(sig2, Symbol('s'))
    print(s)
    l = BasicState(Symbol('l'))
    j = l.couple(s, Symbol('j'))
    print(j)
    j = s.couple(l, Symbol('j'))
    print(j)
    print(j.substructure[0], j.substructure[1])
