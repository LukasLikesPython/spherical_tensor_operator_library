from __future__ import annotations
from abc import ABC, abstractmethod
from sympy import Symbol
from sympy.physics.wigner import clebsch_gordan
from typing import Optional


class State(ABC):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Symbol = None, factor=1,
                 substructure: Optional[list] = None):
        self._angular_quantum_number = angular_quantum_number
        self._other_quantum_number = other_quantum_number
        self._factor = factor
        self._substructure = substructure

    def __str__(self):
        if self.other_quantum_number:
            return f"|{self.other_quantum_number}{self.representation()}>"
        else:
            return f"|{self.representation()}>"

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

    @abstractmethod
    def couple(self, other: State, new_quantum_number: Symbol) -> State:
        pass

    @abstractmethod
    def representation(self) -> str:
        pass


class CoupledState(State):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Symbol = None, factor=1,
                 substructure=None):
        if not substructure:
            raise AttributeError('[ERROR] A coupled state must have a substructure.')
        super().__init__(angular_quantum_number, other_quantum_number, factor, substructure)

    def couple(self, other, new_quantum_number: Symbol) -> CoupledState:
        new_factor = self.factor * other.factor
        new_other_quantum_number = ''
        if self.other_quantum_number:
            new_other_quantum_number = self.other_quantum_number
        if other.other_quantum_number:
            new_other_quantum_number += other.other_quantum_number
        if not new_other_quantum_number:
            new_other_quantum_number = None
        return CoupledState(new_quantum_number, new_other_quantum_number, new_factor, [self, other])

    def representation(self) -> str:
        return f"{self.angular_quantum_number}" \
               f"({self.substructure[0].representation()}{self.substructure[1].representation()})"

    def decouple(self) -> [State, State]:
        pass # TODO use cg to decouple the states


class BasicState(State):

    def __init__(self, angular_quantum_number: Symbol, other_quantum_number: Symbol = None, factor=1,
                 substructure=None):
        if substructure:
            raise AttributeError('[ERROR] A basic state cannot have a substructure.')
        super().__init__(angular_quantum_number, other_quantum_number, factor)

    def couple(self, other: State, new_quantum_number: Symbol) -> CoupledState:
        new_factor = self.factor * other.factor
        new_other_quantum_number = ''
        if self.other_quantum_number:
            new_other_quantum_number = self.other_quantum_number
        if other.other_quantum_number:
            new_other_quantum_number += other.other_quantum_number
        if not new_other_quantum_number:
            new_other_quantum_number = None
        return CoupledState(new_quantum_number, new_other_quantum_number, new_factor, [self, other])

    def representation(self) -> str:
        return f"{self.angular_quantum_number}"


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

