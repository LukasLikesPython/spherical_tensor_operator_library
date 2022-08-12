from sympy import sqrt, I
from tensor_operator import TensorOperator


class VectorOperator(object):
    """
    Small auxiliary class in the transformation from vector operators to tensor operators
    The class needs the space in which the vector acts and a symbol for the vector
    """
    def __init__(self, space, symbol):
        self._space = space
        self._symbol = symbol

    @property
    def space(self):
        return self._space

    @property
    def symbol(self):
        return self._symbol

    def to_tensor(self) -> TensorOperator:
        return TensorOperator(rank=1, factor=1, space=self.space, symbol=self.symbol, substructure=None)


class TensorFromVectors(object):

    @classmethod
    def tensor_from_scalar_product(cls, vector_operator_1, vector_operator_2) -> TensorOperator:
        if isinstance(vector_operator_1, VectorOperator) and isinstance(vector_operator_2, VectorOperator):
            tensor_operator_1 = vector_operator_1.to_tensor()
            tensor_operator_2 = vector_operator_2.to_tensor()
        elif isinstance(vector_operator_1, TensorOperator) and isinstance(vector_operator_2, TensorOperator):
            tensor_operator_1 = vector_operator_1
            tensor_operator_2 = vector_operator_2
        else:
            raise ValueError(f"Input type {type(vector_operator_1)} with {type(vector_operator_2)} is not allowed.")
        return tensor_operator_1.couple(tensor_operator_2, 0, -sqrt(3))

    @classmethod
    def tensor_from_vector_product(cls, vector_operator_1, vector_operator_2) -> TensorOperator:
        if isinstance(vector_operator_1, VectorOperator) and isinstance(vector_operator_2, VectorOperator):
            tensor_operator_1 = vector_operator_1.to_tensor()
            tensor_operator_2 = vector_operator_2.to_tensor()
        elif isinstance(vector_operator_1, TensorOperator) and isinstance(vector_operator_2, TensorOperator):
            tensor_operator_1 = vector_operator_1
            tensor_operator_2 = vector_operator_2
        else:
            raise ValueError(f"Input type {type(vector_operator_1)} with {type(vector_operator_2)} is not allowed.")
        return tensor_operator_1.couple(tensor_operator_2, 1, -I * sqrt(2))


if __name__ == "__main__":
    vector_1 = VectorOperator("relative", "q")
    vector_2 = VectorOperator("relative", "k")
    print(TensorFromVectors.tensor_from_scalar_product(vector_1, vector_2))
    print(TensorFromVectors.tensor_from_vector_product(vector_1, vector_2))
    print(TensorFromVectors.tensor_from_vector_product(vector_1, vector_1))
