from sympy import Symbol
from sympy.physics.wigner import (
                                  clebsch_gordan,
                                  wigner_6j,
                                  wigner_9j
                                  )
from abc import ABC


class TensorOperatorComponent(ABC):
    """

    """
    def __init__(self, rank=0, factor=1):
        self._rank = rank
        self._factor = factor

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        self._factor = value


class TensorOperatorComposite(TensorOperatorComponent):
    pass


class TensorOperator(TensorOperatorComponent):
    pass







if __name__ == "__main__":
    top = TensorOperatorComponent()
    top.rank(10)