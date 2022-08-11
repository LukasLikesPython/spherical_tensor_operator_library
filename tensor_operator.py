from abc import ABC, abstractmethod
import tensor_algebra
from copy import deepcopy

class TensorOperatorInterface(ABC):
    """

    """

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return self.to_expression()

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __mul__(self, factor):
        pass

    def __rmul__(self, factor):
        return self.__mul__(factor)

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
    """

    """

    def __init__(self, *args):
        self.children = [arg for arg in args]

    def __mul__(self, factor):
        for child in self.children:
            child *= factor

    def to_latex(self):
        return " + ".join([child.to_latex() for child in self.children])

    def to_expression(self):
        return " + ".join([child.to_expression() for child in self.children])

    def add(self, other):
        if isinstance(other, self.__class__):
            args = self.children + other.children  # This keeps a flat hierarchy
        else:
            args = self.children + [other]
        new_object = self.__class__(*args)
        new_object._simplify()
        return new_object

    def couple(self, other, rank, factor):
        if isinstance(other, self.__class__):
            args = [child.couple(other_child, rank, factor) for child in self.children for other_child in
                    other.children]
        else:
            args = [child.couple(other, rank, factor) for child in self.children]

        new_object = self.__class__(*args)
        new_object._simplify()
        return new_object

    def _simplify(self):
        children = self.children
        new_children = []
        while len(children) > 0:
            child = children.pop(0)
            if not child:  # Remove bad couplings
                continue
            for i, other_child in enumerate(children):
                if child == other_child:
                    children.pop(i)
                    child += other_child
            new_children.append(child)
        self.children = new_children


class TensorOperator(TensorOperatorInterface):
    """

    """
    CompositeClass = TensorOperatorComposite
    TensorAlgebra = tensor_algebra.TensorAlgebra

    def __init__(self, rank=0, factor=1, space='1', representation=None, substructure=None):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._representation = representation
        self._substructure = substructure

    def __mul__(self, factor):
        new_factor = self.factor * factor
        return self.__class__(self.rank, new_factor, self.space, self.representation)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        elif self.space == other.space and self.representation == other.representation and self.rank == other.rank:
            return True
        else:
            return False

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

    @representation.setter
    def representation(self, value):
        self._representation = value

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, value):
        self._space = value

    @property
    def substructure(self):
        return self._substructure

    @substructure.setter
    def substructure(self, value):
        self._substructure = value

    def add(self, other):
        if isinstance(other, self.CompositeClass):
            return other + self
        elif self.representation == other.representation and self.rank == other.rank:
            new_factor = self.factor + other.factor
            return self.__class__(self.rank, new_factor, self.space, self.representation)
        else:
            return self.CompositeClass(self, other)

    def to_latex(self):
        return str(self.factor) + " * "\
               + str(self.representation).replace("[", "\\left\\lbrace").replace("]", "\\right\\rbrace").\
                   replace(', ', ' \\otimes ').replace("'", "") + "_" + "{" + str(self.rank) + "}"

    def to_expression(self):
        return str(self.factor) + " * " + self._to_expression_no_factor()

    def _to_expression_no_factor(self):
        return str(self.representation).replace("[", "{").replace("]", "}").replace(', ', ' x ').replace("'", "") \
               + "_" + str(self.rank)

    def couple(self, other, rank, factor):
        # Coupling two operators to the new rank is not possible
        if not abs(self.rank - other.rank) <= rank <= self.rank + other.rank:
            return None

        # Coupling two identical operators to rank 1 is a cross product, i.e., zero for parallel vectors
        if self == other and rank == 1:
            return None

        # Valid cases
        if isinstance(other, self.CompositeClass):
            args = [self.couple(other_child, rank, factor) for other_child in other.children]
            return self.CompositeClass(*args)
        elif isinstance(other, self.__class__):
            new_factor = self.factor * other.factor * factor
            if self.space == other.space:
                new_space = self.space
            else:
                new_space = [self.space, other.space]
            new_representation = [self._to_expression_no_factor(), other._to_expression_no_factor()]
            # keep track of factors on outermost layer
            tensor_a = deepcopy(self)
            tensor_b = deepcopy(other)
            tensor_a.factor = 1
            tensor_b.factor = 1
            substructure = [tensor_a, tensor_b]
            new_object = self.__class__(rank, new_factor, new_space, new_representation, substructure)
            new_object.order()
            print(new_object.substructure)
            return new_object

    def order(self):
        """
        Order the tensor operators according to
            1st) their space
            2nd) their representation
        The idea is to have a consistent represenation. The order itself does not matter.
        :return: None
        """
        new_top = None
        if self.substructure:  # ordering only makes sense if there is a substructure
            tensor_a, tensor_b = self.substructure
            if tensor_a.space > tensor_b.space:
                new_top = self.TensorAlgebra.commute(self)
            else:
                if isinstance(tensor_a.representation, str) \
                        and tensor_a.representation > tensor_b.representation:
                    new_top = self.TensorAlgebra.commute(self)
        if new_top:
            self.factor = new_top.factor
            self.representation = new_top.representation
            self.space = new_top.space
            self.substructure = new_top.substructure


if __name__ == "__main__":
    top = TensorOperator(rank=1, factor=1, space='rel', representation="q")
    ktop = TensorOperator(rank=1, factor=1, space='rel', representation="k")
    qsq = top.couple(top, 0, 3)
    print(qsq)
    print(top)
    print('Mutliplication')
    tlist = qsq + 2 * top
    print(tlist)
    print(top)
    print('Addition')
    tlist = tlist + qsq + top + top + qsq
    print(tlist)

    tlist = tlist + (tlist + ktop)
    print(tlist)
    print('Coupling')
    ntlist = tlist.couple(ktop, 1, 1)
    print(ntlist)
    ntlist = ntlist.couple(ktop, 1, 1)
    print(ntlist)
    print(tlist)

