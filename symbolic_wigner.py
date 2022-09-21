from __future__ import annotations
from abc import ABC, abstractmethod
from sympy.physics.wigner import wigner_6j, wigner_9j


class SymbolicWigner(ABC):
    """
    An auxiliary construct that helps to hold the evaluation of wigner_6j and wigner_9j symbols.
    """

    def __init__(self):
        self.factor = None

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
        super().__init__()
        self.children = []
        self.add_children(instance_1)
        self.add_children(instance_2)
        self.factor = 1

    def __str__(self):
        return f"{self.factor} * (" + " * ".join([str(child) for child in self.children]) + ")"

    def __mul__(self, other):
        if isinstance(other, SymbolicWigner):
            self.factor *= other.factor
            self.add_children(other)
        else:
            self.factor *= other

    def evaluate(self, symbol_replace_dict):
        sub_symbol_value = 1
        for child in self.children:
            sub_symbol_value *= child.evaluate(symbol_replace_dict)
        factor = self.factor.subs(symbol_replace_dict) if isinstance(self.factor, Symbol) else self.factor
        return factor * sub_symbol_value

    def add_children(self, other):
        if isinstance(other, self.__class__):
            self.children.append(other.children)
        else:
            self.children.append(other)


class Symbolic6j(SymbolicWigner):

    six_j_cache = {}

    def __init__(self, j1, j2, j3, j4, j5, j6, factor=1):
        super().__init__()
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

    def evaluate(self, symbol_replace_dict):
        j1, j2, j3, j4, j5, j6 = [x.subs(symbol_replace_dict) if isinstance(x, Symbol) else x for x in
                                  (self.j1, self.j2, self.j3, self.j4, self.j5, self.j6, )]
        factor = self.factor.subs(symbol_replace_dict) if isinstance(self.factor, Symbol) else self.factor
        key = (j1, j2, j3, j4, j5, j6)
        if key in self.six_j_cache:
            six_j_value = self.six_j_cache[key]
        else:
            six_j_value = wigner_6j(j1, j2, j3, j4, j5, j6)
            self.six_j_cache[key] = six_j_value
        return factor * six_j_value


class Symbolic9j(SymbolicWigner):

    nine_j_cache = {}

    def __init__(self, j1, j2, j3, j4, j5, j6, j7, j8, j9, factor=1):
        super().__init__()
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

    def evaluate(self, symbol_replace_dict):
        j1, j2, j3, j4, j5, j6, j7, j8, j9 = [x.subs(symbol_replace_dict) if isinstance(x, Symbol) else x for x in
                                              (self.j1, self.j2, self.j3, self.j4, self.j5, self.j6, self.j7, self.j8,
                                               self.j9)]
        factor = self.factor.subs(symbol_replace_dict) if isinstance(self.factor, Symbol) else self.factor
        key = (j1, j2, j3, j4, j5, j6, j7, j8, j9)
        if key in self.nine_j_cache:
            nine_j_value = self.nine_j_cache[key]
        else:
            nine_j_value = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
            self.nine_j_cache[key] = nine_j_value
        return factor * nine_j_value


if __name__ == "__main__":
    from sympy import Symbol

    six_j = Symbolic6j(Symbol('j1'), Symbol('j2'), Symbol('j3'), Symbol('j4'), Symbol('j5'), 0)
    subsdict = {Symbol('j1'): 1, Symbol('j2'): 1, Symbol('j3'): 0, Symbol('j4'): 1, Symbol('j5'): 1,
                Symbol('ja'): 2, Symbol('jb'): 2, Symbol('jc'): 0, Symbol('jd'): 1, Symbol('je'): 1, Symbol('jf'): 2,
                Symbol('jh'): 1, Symbol('jk'): 1, Symbol('ji'): 2}

    print(six_j.evaluate(subsdict))
    print(six_j)

    nine_j = Symbolic9j(Symbol('ja'), Symbol('jb'), Symbol('jc'), Symbol('jd'), Symbol('je'), Symbol('jf'),
                        Symbol('jh'), Symbol('jk'), Symbol('ji'), factor=2)
    print(nine_j)
    print(nine_j.evaluate(subsdict))

    coupled = six_j * nine_j
    print(coupled)
    print(coupled.evaluate(subsdict))
