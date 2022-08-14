from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j
from sympy import sqrt
from math import prod


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
    convention according to the initial space and final result. The functions all start with _recouple_ixj_kxl where
    i and j are the initial term settings and k and l the final one. The example from above would require the function
    _recouple_2x2_2x2
    """

    @classmethod
    def commute(cls, tensor_op):
        tensor_a, tensor_b = tensor_op.substructure
        new_factor = pow(-1, tensor_a.rank + tensor_b.rank - tensor_op.rank)
        new_top = tensor_b.couple(tensor_a, tensor_op.rank, new_factor * tensor_op.factor, order=False)
        return new_top

    @classmethod
    def recouple(cls, tensor_op):
        pass

    @staticmethod
    def _recouple_2x2_2x2(tensor_op, rank, factor):
        """
        Tensor recoupling of the form
        (A x B) x (C x D) -> (A x C) x (B x D)
        The assumption is, that these operators are already ordered according to their spaces within the brackets and
        that space A = space C and space B = space D.

        :param tensor_op: the tensor operator that should be recoupled
        :param rank: the new rank to which it should be coupled
        :param factor: a factor that appears during the recoupling
        :return: recoupled tensor operator
        """
        first_pair, second_pair = tensor_op.substructure
        tensor_a, tensor_b = first_pair.children
        tensor_c, tensor_d = second_pair.children
        if tensor_a.space != tensor_c.space or tensor_b.space != tensor_d.space:
            return None  # This is not the correct recoupling function for the given operator
        new_factor = tensor_op.factor * factor
        a = tensor_a.rank
        b = tensor_b.rank
        c = tensor_c.rank
        d = tensor_d.rank
        e = first_pair.rank
        f = second_pair.rank

        new_tensor_list = []
        for g in range(abs(a - c), a + c + 1):
            for h in range(abs(b - d), b + d + 1):
                new_factor *= jsc([e, f, g, h]) * wigner_9j(a, b, e, c, d, f, g, h, rank)
                if new_factor != 0:
                    new_first_pair = tensor_a.couple(tensor_c, g, 1)
                    new_second_pair = tensor_b.couple(tensor_d, h, 1)
                    new_tensor = new_first_pair.couple(new_second_pair, rank, new_factor)
                    new_tensor_list.append(new_tensor)
        return new_tensor_list

"""
        def tenrec431(self, other, rank, fac=1.):
            '''
            Recouple 431: ((AxB)x(AB x C)) => (((AxB)xAB)xC)
            '''
            print
            'recoupling (431)'
            obj1 = self.obj[0][0]
            obj2 = self.obj[0][1]
            obj3 = other.obj[0][0]
            obj4 = other.obj[0][1]
            attr1 = self.attr[0]
            attr2 = self.attr[1]
            attr3 = other.attr[0]
            attr4 = other.attr[1]
            c = self.rank
            f = other.rank
            a = obj1[2]
            b = obj2[2]
            d = obj3[2]
            e = obj4[2]
            i = rank
            o1 = TensorOp(a, attr1, obj1[1], obj1[0], fac=1.)
            o2 = TensorOp(b, attr2, obj2[1], obj2[0], fac=1.)
            o3 = TensorOp(d, attr3, obj3[1], obj3[0], fac=1.)
            o4 = TensorOp(e, attr4, obj4[1], obj4[0], fac=1.)
            prefac = self.fac * other.fac * fac
            outlist = []
            for h in range(abs(d - c), d + c + 1):
                fac = ljsc([f, h]) * sj(d, c, h, i, e, f) * prefac * pow(-1.,
                                                                         d + c + e + i)  # I am so happy that the six J symbols reads d(a) chief ... what am I doing with my life
                if fac != 0:
                    try:
                        co1 = o1.couple(o2, c, 1.)
                        co2 = co1.couple(o3, h, 1.)
                        oout = co2.couple(o4, i, fac)
                        outlist += [oout, ]
                    except:
                        print
                        'no combination for tensor combination of ranks (431)', h
            return TOPlist(outlist)

        def tenrec321(self, other, rank, fac=1.):
            '''
            input (AxB) x D
            output (A x D) x B
            '''
            print
            'recoupling (321)'
            obj1 = self.obj[0][0]
            obj2 = self.obj[0][1]
            obj3 = other.obj
            attr1 = self.attr[0]
            attr2 = self.attr[1]
            attr3 = other.attr
            c = self.rank
            a = obj1[2]
            b = obj2[2]
            d = obj3[2]
            i = rank
            o1 = TensorOp(a, attr1, obj1[1], obj1[0], fac=1.)
            o2 = TensorOp(b, attr2, obj2[1], obj2[0], fac=1.)
            o3 = TensorOp(d, attr3, obj3[1], obj3[0], fac=1.)
            print
            o1
            print
            o2
            print
            o3
            prefac = self.fac * other.fac * fac * jsc(c)
            outlist = []
            for f in range(abs(a - d), a + d + 1):
                fac = jsc(f) * sj(a, b, c, i, d, f) * prefac * pow(-1., f + b)
                if fac != 0:
                    try:
                        if kill_check(o1, o3, f):  # cross product of identical vectors vanishes
                            print
                            "Vector product vanishes"
                        else:
                            co1 = o1.couple(o3, f, 1.)
                            oout = co1.couple(o2, i, fac)
                            outlist += [oout, ]
                    except:
                        print
                        'no combination for tensor combination of rank (321)', f
            return TOPlist(outlist)

        def tenrec321_2(self, other, rank, fac=1.):
            '''
            input (AxB) x D
            output A  x (B x D)
            '''
            print
            'recoupling (321_2)'
            obj1 = self.obj[0][0]
            obj2 = self.obj[0][1]
            obj3 = other.obj
            attr1 = self.attr[0]
            attr2 = self.attr[1]
            attr3 = other.attr
            c = self.rank
            a = obj1[2]
            b = obj2[2]
            d = obj3[2]
            i = rank
            o1 = TensorOp(a, attr1, obj1[1], obj1[0], fac=1.)
            o2 = TensorOp(b, attr2, obj2[1], obj2[0], fac=1.)
            o3 = TensorOp(d, attr3, obj3[1], obj3[0], fac=1.)
            prefac = self.fac * other.fac * fac * jsc(c) * pow(-1., b + d - c - i)
            outlist = []
            for f in range(abs(b - d), b + d + 1):
                fac = jsc(f) * sj(b, a, c, i, d, f) * prefac
                if fac != 0:
                    try:
                        if kill_check(o2, o3, f):  # cross product of identical vectors vanishes
                            print
                            "Vector product vanishes"
                        else:
                            co2 = o2.couple(o3, f, 1.)
                            oout = o1.couple(co2, i, fac)
                            outlist += [oout, ]
                    except:
                        print
                        'no combination for tensor combination of rank (321_2)', f
            return TOPlist(outlist)

        def tenrec312(self, other, rank, fac=1.):
            '''
            input A x (BxD)
            output (A x B) x D
            '''
            # print 'recoupling (312)'
            obj1 = self.obj
            obj2 = other.obj[0][0]
            obj3 = other.obj[0][1]
            attr1 = self.attr
            attr2 = other.attr[0]
            attr3 = other.attr[1]
            f = other.rank
            a = obj1[2]
            b = obj2[2]
            d = obj3[2]
            i = rank
            o1 = TensorOp(a, attr1, obj1[1], obj1[0], fac=1.)
            o2 = TensorOp(b, attr2, obj2[1], obj2[0], fac=1.)
            o3 = TensorOp(d, attr3, obj3[1], obj3[0], fac=1.)
            prefac = self.fac * other.fac * fac * jsc(f) * pow(-1., f + b + a - i)
            outlist = []
            for c in range(abs(a - b), a + b + 1):
                fac = jsc(c) * sj(b, d, f, i, a, c) * prefac
                if fac != 0:
                    try:
                        if kill_check(o1, o2, c):  # cross product of identical vectors vanishes
                            print
                            "Vector product vanishes"
                        else:
                            co1 = o1.couple(o2, c, 1.)
                            oout = co1.couple(o3, i, fac)
                            outlist += [oout, ]
                    except:
                        print
                        'no combination for tensor combination of rank (312)', c
            return TOPlist(outlist)
"""


if __name__ == "__main__":
    from tensor_operator import TensorOperator
    from tensor_transformation import TensorFromVectors
    q = TensorOperator(rank=1, symbol="q", space="rel")
    sig1 = TensorOperator(rank=1, symbol="sig1", space="spin")
    sig2 = TensorOperator(rank=1, symbol="sig2", space="spin")
    tensor_op = TensorFromVectors.tensor_from_scalar_product(q, sig1).\
        couple(TensorFromVectors.tensor_from_scalar_product(q, sig2), 0, 1)
    print(tensor_op)