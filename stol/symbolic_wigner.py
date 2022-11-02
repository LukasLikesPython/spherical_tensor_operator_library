from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import Union
from sympy.physics.wigner import wigner_6j, wigner_9j
from sympy import Symbol


def factor_eval(factor: Union[int, float, Symbol], symbol_replace_dict: dict):
    """
    Auxiliary function to evaluate the factor. The function checks the type and substitutes potential symbols.
    :param factor: The factor from any of the objects below.
    :param symbol_replace_dict: A dictionary that contains any unresolved symbols and their replacement values as key-
           value pairs.
    :return: The evaluated form of the factor.
    """
    try:
        return factor.subs(symbol_replace_dict)
    except AttributeError:
        return factor


class SymbolicWigner(ABC):
    """
    An auxiliary construct that helps to hold the evaluation of wigner_6j and wigner_9j symbols.
    """

    def __init__(self):
        self.factor = None

    @abstractmethod
    def evaluate(self, symbol_replace_dict: dict):
        """
        Perform a substitution of the symbols in the object according to the input dictionary and evaluate the symbolic
        6j and or 9j symbols.

        :param symbol_replace_dict: A dictionary that contains symbols and their replacement value as key-value pairs.
        :return: the evaluated result.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Print the object's components in a symbolic way, e.g. SixJ(j1 j2 j3; j4 j5 j6).
        :return: String representation of the object.
        """
        pass

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other: Union[SymbolicWigner, int, float, Symbol]) -> SymbolicWigner:
        """
        Multiplication with another object. The other object needs to be of type
            - SymbolicWigner -> Return Value is SymbolicWignerComposite
            - int, float, Symbol -> Changes only the factor and returns a Symbolic6j/9j object
        :param other: SymbolicWigner, int, float, or Symbol type
        :return: Either SymbolicWignerComposite (other is of type SymbolicWigner) or Symbolic6j/9j (else)
        """
        if isinstance(other, SymbolicWigner):
            return SymbolicWignerComposite(self, other)
        else:
            new_self = copy.deepcopy(self)
            new_self.factor *= other
            return new_self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)


class SymbolicWignerComposite(SymbolicWigner):
    """
    Stores a collection of SymbolicWigner symbols. This allows to multiply 6j and 9j symbols and store them in a
    symbolic way.
    """

    def __init__(self, instance_1, instance_2):
        super().__init__()
        self.children = []
        self.add_children(instance_1)
        self.add_children(instance_2)
        self.factor = 1

    def __str__(self):
        return f"{self.factor} * (" + " * ".join([str(child) for child in self.children]) + ")"

    def __mul__(self, other: Union[SymbolicWigner, int, float, Symbol]) -> SymbolicWignerComposite:
        """
        Overrides the __mul__ function in the interface. In case the "other" input is of type SymbolicWigner
        :param other: SymbolicWigner object. Depending on the exact subtype, the add_children class extends the
                      children list or appends the object.
        :return: SymbolicWignerComposite with updated children or factor.
        """
        new_self = copy.deepcopy(self)
        if isinstance(other, SymbolicWigner):
            new_self.factor *= other.factor
            new_self.add_children(other)
        else:
            new_self.factor *= other
        return new_self

    def evaluate(self, symbol_replace_dict: dict):
        """
        Replace all symbols by numbers. If not all symbols. See also Symbolic6j and Symbolic9j. An error is raised in
        case the input of the 6j and/or 9j symbols in this collection are not completely replaced.

        :param symbol_replace_dict: dictionary that contains symbols as keys and their replacement values as values
        :return: the evaluated collection of 6j and/or 9j symbols.
        """
        sub_symbol_value = 1
        for child in self.children:
            sub_symbol_value *= child.evaluate(symbol_replace_dict)
        factor = factor_eval(self.factor, symbol_replace_dict)
        return factor * sub_symbol_value

    def add_children(self, other: SymbolicWigner) -> None:
        """
        Extends children in case other is of same type, otherwise append other object to children. The action happens
        in place.
        :param other: SymbolicWigner object
        :return: None
        """
        if isinstance(other, self.__class__):
            self.children.extend(other.children)
        else:
            self.children.append(other)


class Symbolic6j(SymbolicWigner):
    """
    Auxiliary construct to handle Wigner-6j symbols in a symbolic way.
    """

    six_j_cache = {}

    def __init__(self, j1, j2, j3, j4, j5, j6, factor: Union[int, float, Symbol] = 1):
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

    def evaluate(self, symbol_replace_dict: dict):
        """
        Replace all symbols by numbers. If not all symbols in the 6j symbol are provided, the wigner_6j function will
        raise an error.

        :param symbol_replace_dict: dictionary that contains symbols as keys and their replacement values as values
        :return: the evaluated 6j symbol
        """
        j1, j2, j3, j4, j5, j6 = [
            x.subs(symbol_replace_dict) if isinstance(x, Symbol) else x
            for x in (
                self.j1,
                self.j2,
                self.j3,
                self.j4,
                self.j5,
                self.j6,
            )
        ]
        factor = factor_eval(self.factor, symbol_replace_dict)
        key = (j1, j2, j3, j4, j5, j6)
        if key in self.six_j_cache:
            six_j_value = self.six_j_cache[key]
        else:
            six_j_value = wigner_6j(j1, j2, j3, j4, j5, j6)
            self.six_j_cache[key] = six_j_value
        return factor * six_j_value


class Symbolic9j(SymbolicWigner):
    """
    Auxiliary construct to handle Wigner-9j symbols in a symbolic way.
    """

    nine_j_cache = {}

    def __init__(self, j1, j2, j3, j4, j5, j6, j7, j8, j9, factor: Union[int, float, Symbol] = 1):
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
        return (
            f"{self.factor} * NineJ({self.j1} {self.j2} {self.j3}; {self.j4} {self.j5} {self.j6}; {self.j7} "
            + f"{self.j8} {self.j9})"
        )

    def evaluate(self, symbol_replace_dict: dict):
        """
        Replace all symbols by numbers. If not all symbols in the 9j symbol are provided, the wigner_9j function will
        raise an error.

        :param symbol_replace_dict: dictionary that contains symbols as keys and their replacement values as values
        :return: the evaluated 9j symbol
        """
        j1, j2, j3, j4, j5, j6, j7, j8, j9 = [
            x.subs(symbol_replace_dict) if isinstance(x, Symbol) else x
            for x in (self.j1, self.j2, self.j3, self.j4, self.j5, self.j6, self.j7, self.j8, self.j9)
        ]
        factor = factor_eval(self.factor, symbol_replace_dict)
        key = (j1, j2, j3, j4, j5, j6, j7, j8, j9)
        if key in self.nine_j_cache:
            nine_j_value = self.nine_j_cache[key]
        else:
            nine_j_value = wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
            self.nine_j_cache[key] = nine_j_value
        return factor * nine_j_value
