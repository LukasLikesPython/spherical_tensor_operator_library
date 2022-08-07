from sympy import Symbol
from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j
from abc import ABC, abstractmethod


class TensorOperatorInterface(ABC):
    """

    """
    def __init__(self, rank=0, factor=1, space='1', representation='X'):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._representation = representation

    def __mul__(self, other):
        self._factor *= other

    def __add__(self, other):
        self.add(other)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_latex(self):
        pass

    @abstractmethod
    def recouple(self):
        pass

    @abstractmethod
    def add(self, other):
        pass

    @abstractmethod
    def remove(self, other):
        pass

    @abstractmethod
    def couple(self, other):
        pass

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




class TensorOperatorComposite(TensorOperatorInterface):
    pass


class TensorOperator(TensorOperatorInterface):
    pass




