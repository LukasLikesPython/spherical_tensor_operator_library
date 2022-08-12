from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j


class TensorAlgebra(object):

    @classmethod
    def commute(cls, tensor_op):
        tensor_a, tensor_b = tensor_op.substructure
        new_factor = pow(-1, tensor_a.rank + tensor_b.rank - tensor_op.rank)
        new_top = tensor_b.couple(tensor_a, tensor_op.rank, new_factor * tensor_op.factor, order=False)
        return new_top

    @classmethod
    def recouple(cls, tensor_op):
        pass

    @classmethod
    def vector_product_to_tensor(cls):
        pass


