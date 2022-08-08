from sympy import Symbol
from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j
from abc import ABC, abstractmethod
import numpy as np


class TensorOperatorInterface(ABC):
    """

    """

    def __add__(self, other):
        self.add(other)

    def __str__(self):
        return self.to_expression()

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def to_latex(self):
        pass

    @abstractmethod
    def to_expression(self):
        pass

    @abstractmethod
    def add(self, other):
        pass

    @abstractmethod
    def couple(self, other, rank, factor):
        pass


class TensorOperatorComposite(TensorOperatorInterface):

    def __init__(self, *args):
        self.children = [arg for arg in args]

    def __mul__(self, other):
        for child in self.children:
            child *= other

    def to_latex(self):
        return " + ".join([child.to_latex() for child in self.children])

    def to_expression(self):
        return " + ".join([child.to_expression() for child in self.children])

    def add(self, other):
        self.children.append(other)

    def couple(self, other, rank, factor):
        if isinstance(other, TensorOperatorComposite):
            new_children = [child.couple(other_child, rank, factor) for child in self.children for other_child in
                            other.children]
        elif isinstance(other, TensorOperator):
            new_children = [child.couple(other, rank, factor) for child in self.children]
        else:
            raise (TypeError(f"Cannot couple type {type(other)} to an object of {type(self)}."))
        return TensorOperatorComposite(new_children)


class TensorOperator(TensorOperatorInterface):
    CompositeClass = TensorOperatorComposite

    def __init__(self, rank=0, factor=1, space='1', representation='X', tex_representation='X'):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._representation = representation
        self._tex_representation = tex_representation

    def __mul__(self, other):
        self._factor *= other.factor

    @property
    def rank(self):
        return self._rank

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        self._factor = value

    @property
    def representation(self):
        return self._representation

    @property
    def space(self):
        return self._space

    def add(self, other):
        pass

    def couple(self, other, rank, factor):
        return


"""    
representation = "{" + tensorOne + " x " + tensorTwo + "}"

        if tensorOne.space() == tensorTwo.space():
            space = tensorOne.space()
        else:
            space = "{" + tensorOne.space() + " x " + tensorTwo.space() + "}"

        super().__init__(rank, factor, space=space, representation=representation)
        self._tex_representation = "\left\lbrace" + tensorOne + " \otimes " + tensorTwo + "\right\rbrace"
        self.tensorOne = tensorOne
        self.tensorTwo = tensorTwo
"""
