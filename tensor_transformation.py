from __future__ import annotations
from typing import Optional
from sympy import sqrt, I

from tensor_operator import TensorOperator


class WrongOperatorTypeError(Exception):
    """
    This error gets raised when the Input type of the TensorOperator is not correct in the TensorsFromVector class.
    """


class TensorFromVectors(object):
    """
    In equations, we are usually confronted with vectors rather than tensors. The transition from Vectors to Tensors
    is accompanied by specific factors, depending on the type of vector coupling. This class pretends that the
    basic tensors we construct are still vectors, and offers methods to join them with the vector operations
    scalar_product and vector_product. The result is the tensor representation with the correct factors.
    """

    @classmethod
    def scalar_product(cls, operator_1: TensorOperator, operator_2: TensorOperator) -> Optional[TensorOperator]:
        """
        Uses TensorOperator objects as input, pretends they are still vectors that are coupled via a scalar product
        and returns the tensor product
        {A_1 x B_1}_0 with the correct factor -sqrt(3).

        :param operator_1: TensorOperator
        :param operator_2: TensorOperator
        :return: TensorOperator or None, depending on the constellation
        """
        if not(isinstance(operator_1, TensorOperator) and isinstance(operator_2, TensorOperator)):
            raise WrongOperatorTypeError(f"Input type {type(operator_1)} with {type(operator_2)} is not allowed.")
        return operator_1.couple(operator_2, 0, -sqrt(3))

    @classmethod
    def vector_product(cls, operator_1: TensorOperator, operator_2: TensorOperator) -> Optional[TensorOperator]:
        """
        Uses TensorOperator objects as input, pretends they are still vectors that are coupled via a vector product
        and returns the tensor product
        {A_1 x B_1}_1 with the correct factor -I sqrt(2).

        :param operator_1: TensorOperator
        :param operator_2: TensorOperator
        :return: TensorOperator or None, depending on the constellation
        """
        if not(isinstance(operator_1, TensorOperator) and isinstance(operator_2, TensorOperator)):
            raise WrongOperatorTypeError(f"Input type {type(operator_1)} with {type(operator_2)} is not allowed.")
        return operator_1.couple(operator_2, 1, -I * sqrt(2))

