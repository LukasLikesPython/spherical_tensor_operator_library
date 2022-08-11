from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j

class TensorAlgebra(object):

    @classmethod
    def commute(cls, tensor_op):
        tensor_a, tensor_b = tensor_op.substructure
        tensor_a.factor = 1  # the outermost layer keeps track of the factor
        tensor_b.factor = 1
        new_factor = pow(-1., tensor_a.rank + tensor_b.rank - tensor_op.rank)
        new_top = tensor_b.couple(tensor_a, tensor_op.rank, new_factor * tensor_op.factor)
        return new_top

    @classmethod
    def vector_product_to_tensor(cls):
        pass


