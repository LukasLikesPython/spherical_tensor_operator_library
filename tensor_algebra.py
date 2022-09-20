from __future__ import annotations
from typing import Optional, Union
from sympy.physics.wigner import wigner_6j, wigner_9j
from sympy import sqrt
from math import prod

from tensor_operator import TensorOperator, TensorOperatorComposite

DEBUG_MODE = False


def jsc(*args):
    """
    Auxiliary function for a reappearing element during recoupling actions [j], where [j] = sqrt(2 * j + 1)

    :param args: either a single integer or multiple integers
    :return: the product of the input
    """
    return prod([sqrt(2 * arg + 1) for arg in args])


class TensorAlgebra(object):
    """

    Naming convention for tensor operator recoupling:
    The recoupling operations work for instance as follows (A x B) x (C x D) -> (A x C) x (B x D). We define our naming
    convention according to the initial space and final result. The functions all start with _recouple_initial_final.
    The example from above would require the function
    _recouple_ABxCD_ACxBD
    """

    @staticmethod
    def _can_be_recoupled_ABxAB_AAxBB(tensor_op: TensorOperator) -> bool:
        """
        Checks the subspaces of this operator. If the operator subspace structure returns two mixed states, with space
        one = A and space two = B coupled like (AxB) x (AxB) we can recouple it to (AxA) x (BxB) and the function
        returns True, otherwise it returns False.

        :param tensor_op: A TensorOperator for which we want to check whether it can be recoupled.
        :return: bool True or False. True -> Can be recoupled. False -> Cannot be recoupled.
        """
        if not tensor_op.substructure:
            return False
        first_pair, second_pair = tensor_op.substructure
        if not (first_pair.substructure and second_pair.substructure):
            return False
        tensor_a, tensor_b = first_pair.substructure
        tensor_c, tensor_d = second_pair.substructure
        if tensor_a.space == tensor_c.space and tensor_b.space == tensor_d.space and tensor_a.space != tensor_b.space:
            return True
        return False

    @staticmethod
    def _can_be_recoupled_AAxAB_AAAxB(tensor_op: TensorOperator) -> bool:
        """
        Checks the subspaces of this operator. If the operator subspace structure returns two mixed states, with space
        one = A and space two = B coupled like (AxA) x (AxB) we can recouple it to ((AxA) x A) x B and the function
        returns True, otherwise it returns False.

        :param tensor_op: A TensorOperator for which we want to check whether it can be recoupled.
        :return: bool True or False. True -> Can be recoupled. False -> Cannot be recoupled.
        """
        if not tensor_op.substructure:
            return False
        first_pair, second_pair = tensor_op.substructure
        if not (first_pair.substructure and second_pair.substructure):
            return False
        tensor_a, tensor_b = first_pair.substructure
        tensor_c, tensor_d = second_pair.substructure
        if tensor_a.space <= tensor_b.space and tensor_a.space <= tensor_c.space and \
                max([tensor_a.space.order, tensor_b.space.order, tensor_c.space.order]) < tensor_d.space.order:
            return True
        return False

    @staticmethod
    def _can_be_recoupled_ABxC_ACxB(tensor_op: TensorOperator) -> bool:
        """
        Checks the subspaces of this operator. If the operator subspace structure returns one mixed state and one pure
        state, with space one = AxB and space two = C coupled like (AxB) x C we can recouple it to (AxC) x B given that
        either space A = space C != space B or order space A >= order space C > order space B.

        :param tensor_op: A TensorOperator for which we want to check whether it can be recoupled.
        :return: bool True or False. True -> Can be recoupled. False -> Cannot be recoupled.
        """
        if not tensor_op.substructure:
            return False
        first_pair, tensor_c = tensor_op.substructure
        if not first_pair.substructure:
            return False
        tensor_a, tensor_b = first_pair.substructure
        if tensor_a.space == tensor_b.space or tensor_b.space == tensor_c.space \
                or tensor_b.space.contains(tensor_c.space):
            return False
        elif tensor_a.space == tensor_c.space or tensor_a.space <= tensor_c.space < tensor_b.space:
            return True
        return False

    @staticmethod
    def _can_be_recoupled_ABxC_AxBC(tensor_op: TensorOperator) -> bool:
        """
        Checks the subspaces of this operator. If the operator subspace structure returns one mixed state and one pure
        state, with space one = AxB and space two = C coupled like (AxB) x C we can recouple it to A x (BxC) given that
        either space A != space C and space C == space B.

        :param tensor_op: A TensorOperator for which we want to check whether it can be recoupled.
        :return: bool True or False. True -> Can be recoupled. False -> Cannot be recoupled.
        """
        if not tensor_op.substructure:
            return False
        first_pair, tensor_c = tensor_op.substructure
        if not first_pair.substructure:
            return False
        tensor_a, tensor_b = first_pair.substructure
        if tensor_a.space == tensor_b.space or tensor_a.space == tensor_c.space:
            return False
        elif tensor_a.space.order < min([tensor_c.space.order, tensor_b.space.order]) and \
                (tensor_b.space.contains(tensor_c.space) or tensor_c.space.contains(tensor_b.space)):
            return True
        return False

    @staticmethod
    def _can_be_recoupled_AxBC_ABxC(tensor_op: TensorOperator) -> bool:
        """
        Checks the subspaces of this operator. If the operator subspace structure returns one mixed state and one pure
        state, with space one = A and space two = BxC coupled like A x (BxC) we can recouple it to (AxB) x C given that
        either space B != space C and space A == space B.

        :param tensor_op: A TensorOperator for which we want to check whether it can be recoupled.
        :return: bool True or False. True -> Can be recoupled. False -> Cannot be recoupled.
        """
        if not tensor_op.substructure:
            return False
        tensor_a, second_pair = tensor_op.substructure
        if not second_pair.substructure:
            return False
        tensor_b, tensor_c = second_pair.substructure
        if tensor_b.space == tensor_c.space or tensor_a.space == tensor_c.space:
            return False
        elif tensor_c.space.order >= max([tensor_a.space.order, tensor_b.space.order]) and \
                (tensor_a.space.contains(tensor_b.space) or tensor_b.space.contains(tensor_a.space)):
            return True
        return False

    @classmethod
    def _perform_recoupling(cls, tensor_op: TensorOperator, factor=1) \
            -> Union[None, TensorOperator, TensorOperatorComposite]:
        out_tensor = tensor_op
        if cls._can_be_recoupled_ABxAB_AAxBB(out_tensor):
            out_tensor = cls._recouple_ABxCD_ACxBD(out_tensor, factor)
            return out_tensor
        if cls._can_be_recoupled_AAxAB_AAAxB(out_tensor):
            out_tensor = cls._recouple_ABxCD_ABCxD(out_tensor, factor)
            return out_tensor
        if cls._can_be_recoupled_ABxC_ACxB(out_tensor):
            out_tensor = cls._recouple_ABxC_ACxB(out_tensor, factor)
            return out_tensor
        if cls._can_be_recoupled_ABxC_AxBC(out_tensor):
            out_tensor = cls._recouple_ABxC_AxBC(out_tensor, factor)
            return out_tensor
        return out_tensor

    @classmethod
    def _perform_recoupling_composite(cls, tensor_op: TensorOperatorComposite, factor=1) \
            -> Union[None, TensorOperator, TensorOperatorComposite]:
        out_tensor = cls._perform_recoupling(tensor_op.children[0], factor)
        for child in tensor_op.children[1:]:
            res = cls._perform_recoupling(child, factor)
            if not res:
                continue
            out_tensor += res

        return out_tensor

    @classmethod
    def _recouple_substructure(cls, tensor_op: Union[TensorOperator, TensorOperatorComposite]) \
            -> Union[None, TensorOperator, TensorOperatorComposite]:
        if isinstance(tensor_op, TensorOperator):
            return cls._recouple_basic_substructure(tensor_op)
        elif isinstance(tensor_op, TensorOperatorComposite):
            out_tensor = cls._recouple_basic_substructure(tensor_op.children[0])
            for child in tensor_op.children[1:]:
                res = cls._recouple_basic_substructure(child)
                if not res:
                    continue
                out_tensor += res
            return out_tensor

    @classmethod
    def _recouple_basic_substructure(cls, tensor_op: Union[TensorOperator, TensorOperatorComposite]) \
            -> Union[None, TensorOperator, TensorOperatorComposite]:
        """
        Idea: Recoupling should take place starting from the innermost operators going to the outermost operators.
        A recursive call to the recouple method takes care of this.

        :param tensor_op: An operator of type TensorOperator or TensorOperatorComposite
        :return: None, TensorOperator, or TensorOperatorComposite
        """
        out_tensor = tensor_op
        if out_tensor.get_depth() > 2:  # The tensor operator has a substructure that can potentially be recoupled
            sub_tensor_1, sub_tensor_2 = out_tensor.substructure
            new_sub_tensor_1 = cls.recouple(sub_tensor_1, verbose=False, outer_loop=False)
            new_sub_tensor_2 = cls.recouple(sub_tensor_2, verbose=False, outer_loop=False)
            if new_sub_tensor_1 != sub_tensor_1 or new_sub_tensor_2 != sub_tensor_2:
                out_tensor = new_sub_tensor_1.couple(new_sub_tensor_2, out_tensor.rank, out_tensor.factor)
        return out_tensor

    @classmethod
    def _recouple_step(cls, tensor_op, factor):
        out_tensor = cls._recouple_substructure(tensor_op)  # First recouple innermost layers

        if isinstance(out_tensor, TensorOperator):
            out_tensor = cls._perform_recoupling(out_tensor, factor)
        elif isinstance(out_tensor, TensorOperatorComposite):
            out_tensor = cls._perform_recoupling_composite(out_tensor, factor)
        return out_tensor

    @classmethod
    def recouple(cls, tensor_op: Union[TensorOperator, TensorOperatorComposite], factor=1, verbose=True,
                 outer_loop=True) -> Union[TensorOperator, TensorOperatorComposite]:

        out_tensor = cls._recouple_step(tensor_op, factor)

        if out_tensor == tensor_op:
            if verbose:
                print(f'[INFO] Recoupling was not possible for tensor operator {tensor_op}')
        if out_tensor != tensor_op and outer_loop:
            prev_tensor = None
            while prev_tensor != out_tensor:  # Repeat recoupling steps until no further change happens
                prev_tensor = out_tensor
                out_tensor = cls.recouple(out_tensor, factor, verbose=False, outer_loop=False)
        return out_tensor

    @staticmethod
    def _recouple_ABxCD_ACxBD(tensor_op: TensorOperator, factor=1, debug=DEBUG_MODE) -> Optional[TensorOperator]:
        """
        Tensor recoupling of the form
        (A_a x B_b)_ab x (C_c x D_d)_cd -> (A_a x C_c)_ac x (B_b x D_d)_bd
        The assumption is, that these operators are already ordered according to their spaces within the brackets and
        that space A = space C and space B = space D.

        :param tensor_op: the tensor operator that should be recoupled
        :param factor: a factor that appears during the recoupling, default is 1
        :return: recoupled tensor operator of type TensorOperator or TensorOperatorComposite
        """
        first_pair, second_pair = tensor_op.substructure
        tensor_a, tensor_b = first_pair.substructure
        tensor_c, tensor_d = second_pair.substructure
        new_factor = tensor_op.factor * factor
        a = tensor_a.rank
        b = tensor_b.rank
        c = tensor_c.rank
        d = tensor_d.rank
        ab = first_pair.rank
        cd = second_pair.rank
        rank = tensor_op.rank

        out_tensor = None
        for ac in range(abs(a - c), a + c + 1):
            for bd in range(abs(b - d), b + d + 1):
                if not (abs(ac - bd) <= rank <= ac + bd):
                    continue
                operator_factor = new_factor * jsc(ab, cd, ac, bd) * wigner_9j(a, b, ab, c, d, cd, ac, bd, rank)
                if operator_factor != 0:
                    new_first_pair = tensor_a.couple(tensor_c, ac, 1)
                    new_second_pair = tensor_b.couple(tensor_d, bd, 1)
                    if not new_first_pair or not new_second_pair:
                        continue
                    new_tensor = new_first_pair.couple(new_second_pair, rank, operator_factor)
                    if not out_tensor:
                        out_tensor = new_tensor
                    else:
                        out_tensor += new_tensor
        if debug:
            print('[DEBUG] recouple ABxCD -> ACxBD')
            print(tensor_op, '->', out_tensor)
        return out_tensor

    @staticmethod
    def _recouple_ABxCD_ABCxD(tensor_op: TensorOperator, factor=1, debug=DEBUG_MODE) -> Optional[TensorOperator]:
        """
        Tensor recoupling of the form
        (A_a x B_b)_ab x (C_c x D_d)_cd -> ((A_a x B_b)_ab x C_c)_abc x D_d
        The assumption is, that these operators are already ordered according to their spaces within the brackets and
        that space A = space B and space C != space D

        :param tensor_op: the tensor operator that should be recoupled
        :param factor: a factor that appears during the recoupling, default is 1
        :return: recoupled tensor operator of type TensorOperator or TensorOperatorComposite
        """
        first_pair, second_pair = tensor_op.substructure
        tensor_c, tensor_d = second_pair.substructure
        new_factor = tensor_op.factor * factor
        c = tensor_c.rank
        d = tensor_d.rank
        ab = first_pair.rank
        cd = second_pair.rank
        rank = tensor_op.rank
        out_tensor = None
        for abc in range(abs(c - ab), c + ab + 1):
            operator_factor = jsc(cd, abc) * wigner_6j(c, ab, abc, rank, d, cd) * new_factor \
                              * pow(-1, d + c + ab + rank)
            if operator_factor != 0:
                triplet = first_pair.couple(tensor_c, abc, 1)
                if not triplet:
                    continue
                new_tensor = triplet.couple(tensor_d, rank, operator_factor)
                if not out_tensor:
                    out_tensor = new_tensor
                else:
                    out_tensor += new_tensor
        if debug:
            print('[DEBUG] recouple ABxCD -> ABCxD')
            print(tensor_op, '->', out_tensor)
        return out_tensor

    @staticmethod
    def _recouple_ABxC_ACxB(tensor_op: TensorOperator, factor=1, debug=DEBUG_MODE) -> Optional[TensorOperator]:
        """
        Tensor recoupling of the form
        (A_a x B_b)_ab x C_c -> (A_a x C_c)_ac x B_b
        The assumption is, that these operators are already ordered according to their spaces within the brackets.
        This recoupling scheme is applied when the order of space B is higher than that of space C and
        space A != space B.

        :param tensor_op: the tensor operator that should be recoupled
        :param factor: a factor that appears during the recoupling, default is 1
        :return: recoupled tensor operator of type TensorOperator or TensorOperatorComposite
        """
        first_pair, tensor_c = tensor_op.substructure
        tensor_a, tensor_b = first_pair.substructure
        new_factor = tensor_op.factor * factor
        a = tensor_a.rank
        b = tensor_b.rank
        c = tensor_c.rank
        ab = first_pair.rank
        rank = tensor_op.rank
        out_tensor = None
        for ac in range(abs(a - c), a + c + 1):
            operator_factor = jsc(ab, ac) * wigner_6j(a, b, ab, rank, c, ac) * new_factor * pow(-1, ac + c)
            if operator_factor != 0:
                new_pair = tensor_a.couple(tensor_c, ac, 1)
                if not new_pair:
                    continue
                new_tensor = new_pair.couple(tensor_b, rank, operator_factor)
                if not out_tensor:
                    out_tensor = new_tensor
                else:
                    out_tensor += new_tensor
        if debug:
            print('[DEBUG] recouple ABxC -> ACxB')
            print(tensor_op, '->', out_tensor)
        return out_tensor

    @staticmethod
    def _recouple_ABxC_AxBC(tensor_op: TensorOperator, factor=1, debug=DEBUG_MODE) -> Optional[TensorOperator]:
        """
        Tensor recoupling of the form
        (A_a x B_b)_ab x C_c -> A_a x (B_b x C_c)_bc
        The assumption is, that these operators are already ordered according to their spaces within the brackets.
        This recoupling scheme is applied when space B = space C and space A != space B.

        :param tensor_op: the tensor operator that should be recoupled
        :param factor: a factor that appears during the recoupling, default is 1
        :return: recoupled tensor operator of type TensorOperator or TensorOperatorComposite
        """

        first_pair, tensor_c = tensor_op.substructure
        tensor_a, tensor_b = first_pair.substructure
        a = tensor_a.rank
        b = tensor_b.rank
        c = tensor_c.rank
        ab = first_pair.rank
        rank = tensor_op.rank
        out_tensor = None
        new_factor = tensor_op.factor * factor * pow(-1, b + c - ab - rank)
        for bc in range(abs(b - c), b + c + 1):
            operator_factor = jsc(ab, bc) * wigner_6j(b, a, ab, rank, c, bc) * new_factor
            if operator_factor != 0:
                new_pair = tensor_b.couple(tensor_c, bc, 1)
                if not new_pair:
                    continue
                new_tensor = tensor_a.couple(new_pair, rank, operator_factor)
                if not out_tensor:
                    out_tensor = new_tensor
                else:
                    out_tensor += new_tensor
        if debug:
            print('[DEBUG] recouple ABxC -> AxBC')
            print(tensor_op, '->', out_tensor)
        return out_tensor


if __name__ == "__main__":
    from tensor_transformation import TensorFromVectors
    from tensor_space import TensorSpace
    rel_space = TensorSpace('rel', 0)
    spin_space = TensorSpace('spin', 1)
    cm_space = TensorSpace('cm', 2)

    q = TensorOperator(rank=1, symbol='q', space=rel_space)
    k = TensorOperator(rank=1, symbol='k', space=rel_space)
    P = TensorOperator(rank=1, symbol='P', space=cm_space)
    sig1 = TensorOperator(rank=1, symbol='sig1', space=spin_space)
    sig2 = TensorOperator(rank=1, symbol='sig2', space=spin_space)

    operator = TensorFromVectors.scalar_product(TensorFromVectors.vector_product(sig1, sig2),
                                                TensorFromVectors.vector_product(q, k)).\
                couple(TensorFromVectors.scalar_product(q, P), 0, 1)

    print('First')
    first_rec = TensorAlgebra.recouple(operator)
    print(first_rec)

    print('Second')
    second_rec = TensorAlgebra.recouple(first_rec)
    print(second_rec)
