from sympy.physics.wigner import clebsch_gordan, wigner_6j, wigner_9j


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

        :param tensor_op:
        :param rank: the new rank to which it should be coupled
        :param factor:
        :return:
        """
        first_pair, second_pair = tensor_op.substructure
        tensor_a, tensor_b = first_pair.children
        tensor_c, tensor_d = second_pair.children
        if tensor_a.space != tensor_c.space or tensor_b.space != tensor_d.space


        for g in range(abs(a - d), a + d + 1):
            for h in range(abs(b - e), b + e + 1):
                fac = ljsc([c, f, g, h]) * nj(a, b, c, d, e, f, g, h, i) * prefac

        def tenrec422(self, other, rank, fac=1.):
            print
            'recoupling (422)'
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
            for g in range(abs(a - d), a + d + 1):
                for h in range(abs(b - e), b + e + 1):
                    fac = ljsc([c, f, g, h]) * nj(a, b, c, d, e, f, g, h, i) * prefac
                    if fac != 0:
                        try:
                            if kill_check(o1, o3, g) or kill_check(o2, o4,
                                                                   h):  # cross product of identical vectors vanishes
                                print
                                "Vector product vanishes"
                            else:
                                co1 = o1.couple(o3, g, 1.)
                                co2 = o2.couple(o4, h, 1.)
                                oout = co1.couple(co2, i, fac)
                                outlist += [oout, ]
                        except:
                            print
                            'no combination for tensor combination of ranks (422)', g, h
            return TOPlist(outlist)

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
