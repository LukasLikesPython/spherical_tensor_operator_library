from __future__ import annotations
from abc import ABC, abstractmethod
from sympy.physics.wigner import wigner_6j, wigner_9j


class SymbolicWigner(ABC):
    """
    An auxiliary construct that helps to hold the evaluation of wigner_6j and wigner_9j symbols.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1/other)


class SymbolicWignerComposite(SymbolicWigner):

    def __init__(self, instance_1, instance_2):
        self.children = []
        self.add_children(instance_1)
        self.add_children(instance_2)
        self.factor = instance_1.factor * instance_2.factor

    def __str__(self):
        return f"{self.factor} * (" + " * ".join([str(child) for child in self.children]) + ")"

    def __mul__(self, other):
        if isinstance(other, SymbolicWigner):
            self.factor *= other.factor
            self.add_children(other)
        else:
            self.factor *= other

    def evaluate(self, symbol_replace_dict):
        return self.factor.subs(symbol_replace_dict) # TODO

    def add_children(self, other):
        if isinstance(other, self.__class__):
            self.children.append(other.children)
        else:
            self.children.append(other)


class Symbolic6j(SymbolicWigner):

    def __init__(self, j1, j2, j3, j4, j5, j6, factor=1):
        self.j1 = j1
        self.j2 = j2
        self.j3 = j3
        self.j4 = j4
        self.j5 = j5
        self.j6 = j6
        self.factor = factor

    def __str__(self):
        return f"{self.factor} * SixJ({self.j1} {self.j2} {self.j3}; {self.j4} {self.j5} {self.j6})"

    def __mul__(self, other):
        if isinstance(other, SymbolicWigner):
            return SymbolicWignerComposite(self, other)
        else:
            self.factor *= other
            return self

    def evaluate(self, j1, j2, j3, j4, j5, j6):
        return self.factor * wigner_6j(j1, j2, j3, j4, j5, j6)


class Symbolic9j(SymbolicWigner):

    def __init__(self, j1, j2, j3, j4, j5, j6, j7, j8, j9, factor=1):
        self.j1 = j1
        self.j2 = j2
        self.j3 = j3
        self.j4 = j4
        self.j5 = j5
        self.j6 = j6
        self.j7 = j7
        self.j8 = j8
        self.j9 = j9
        self.factor = factor

    def __str__(self):
        return f"{self.factor} * NineJ({self.j1} {self.j2} {self.j3}; {self.j4} {self.j5} {self.j6}; {self.j7} " \
                + f"{self.j8} {self.j9})"

    def __mul__(self, other):
        if isinstance(other, SymbolicWigner):
            return SymbolicWignerComposite(self, other)
        else:
            self.factor *= other
            return self

    def evaluate(self, j1, j2, j3, j4, j5, j6, j7, j8, j9):
        return self.factor * wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)

