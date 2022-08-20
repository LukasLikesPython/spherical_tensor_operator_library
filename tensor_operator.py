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

    def __sub__(self, other):
        return self.add(-1 * other)

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
        self.children = [arg for arg in args if arg]
        self._simplify()

    def __mul__(self, factor):
        args = []
        for child in self.children:
            args.append(child * factor)
        return self.__class__(*args)


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
        return new_object

    def couple(self, other, rank, factor):
        if isinstance(other, self.__class__):
            args = [child.couple(other_child, rank, factor) for child in self.children for other_child in
                    other.children]
        else:
            args = [child.couple(other, rank, factor) for child in self.children]

        new_object = self.__class__(*args)
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
            if child:
                new_children.append(child)
        self.children = new_children


class TensorOperator(TensorOperatorInterface):
    """

    """
    CompositeClass = TensorOperatorComposite
    TensorAlgebra = tensor_algebra.TensorAlgebra

    def __init__(self, rank=0, factor=1, space='1', symbol=None, substructure=None):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._symbol = symbol
        self._substructure = substructure

    def __mul__(self, factor):
        new_factor = self.factor * factor
        if new_factor != 0:
            return self.__class__(self.rank, new_factor, self.space, self.symbol, self.substructure)
        else:
            return None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        elif self.space == other.space and self.symbol == other.symbol and self.rank == other.rank:
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
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value

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

    def get_depth(self):
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth

    def add(self, other):
        if isinstance(other, self.CompositeClass):
            return other + self
        elif self.symbol == other.symbol and self.rank == other.rank:
            new_factor = self.factor + other.factor
            if new_factor != 0:
                return self.__class__(self.rank, new_factor, self.space, self.symbol, self.substructure)
            else:
                return None
        else:
            return self.CompositeClass(self, other)

    def to_latex(self):
        return str(self.factor) + " * " \
               + str(self.symbol).replace("[", "\\left\\lbrace").replace("]", "\\right\\rbrace"). \
                   replace(', ', ' \\otimes ').replace("'", "") + "_" + "{" + str(self.rank) + "}"

    def to_expression(self):
        return str(self.factor) + " * " + self._to_expression_no_factor()

    def _to_expression_no_factor(self):
        return str(self.symbol).replace("[", "{").replace("]", "}").replace(', ', ' x ').replace("'", "") \
               + "_" + str(self.rank)

    def couple(self, other, rank, factor, order=True):
        """
        Couple two Tensor Operators together to a new Tensor Operator of a given rank.
        There are two simplifications. First, the new rank has to fulfill
        |self.rank - other.rank| <= rank <= self.rank + other.rank
        Furthermore, Tensor Operators of rank 1 are vectors and the tensor product of two vectors to rank 1 is
        basically a vector product. The tensor product of two parallel vectors is zero. This means, e.g.,
        {q_1 x q_1}_1 = 0. We set such terms to zero.

        The new operator keeps track of its constituents in the self._substructure variable, all factors are carried
        to the outermost level.

        :param other: Either TensorOperator or TensorOperatorComposite
        :param rank:  The new rank to which the two entities shall be coupled to
        :param factor: Any factors that need to be added to the coupling
        :param order: default True, flag that indicates whether the coupled result shall be ordered
        :return: A new, coupled instance of TensorOperator or TensorOperatorComposite (depending on "other")
        """
        if isinstance(other, self.CompositeClass):
            args = [self.couple(other_child, rank, factor) for other_child in other.children]
            return self.CompositeClass(*args)
        elif isinstance(other, self.__class__):
            # Coupling two operators to the new rank is not possible
            if not abs(self.rank - other.rank) <= rank <= self.rank + other.rank:
                return None

            # Coupling two identical operators to rank 1 is a cross product, i.e., zero for parallel vectors
            if self == other and rank == 1:
                return None

            # Valid Cases
            new_factor = self.factor * other.factor * factor
            if self.space == other.space:
                new_space = self.space
            else:
                new_space = self.space + other.space
            new_symbol = [self._to_expression_no_factor(), other._to_expression_no_factor()]
            # keep track of factors on outermost layer
            tensor_a = deepcopy(self)
            tensor_b = deepcopy(other)
            tensor_a.factor = 1
            tensor_b.factor = 1
            substructure = [tensor_a, tensor_b]
            new_object = self.__class__(rank, new_factor, new_space, new_symbol, substructure)
            if order:
                new_object.order()
            return new_object

    def order(self):
        """
        Order the tensor operators according to
            1st) their space (alphabetically)
            2nd) their structure depth (greater depth first), e.g., {q_1 x q_1}_2 x k_1 should stay that way
            3rd) their symbol (alphabetically)
        The idea is to have a consistent representation. The order itself does not matter.

        :return: None
        """
        new_top = None
        if self.substructure:  # ordering only makes sense if there is a substructure
            tensor_a, tensor_b = self.substructure
            commute = False

            if tensor_a.space > tensor_b.space:
                commute = True
            elif tensor_a.space == tensor_b.space:
                if tensor_b.get_depth() > tensor_a.get_depth():  # depth
                    commute = True
                elif tensor_b.get_depth() == tensor_a.get_depth() \
                        and str(tensor_a.symbol) > str(tensor_b.symbol):  # alphabetically
                    commute = True

            if commute:
                new_top = self.TensorAlgebra.commute(self)

        if new_top:
            self.factor = new_top.factor
            self.symbol = new_top.symbol
            self.space = new_top.space
            self.substructure = new_top.substructure


if __name__ == "__main__":
    top = TensorOperator(rank=1, factor=1, space='rel', symbol="q")
    ktop = TensorOperator(rank=1, factor=1, space='rel', symbol="k")
    other_space = TensorOperator(rank=1, symbol='sig', space='spin')
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

    print(qsq.get_depth())
    qsig = top.couple(other_space, 1, 1)
    print(qsig.get_depth())
    print(ntlist.children[0].get_depth())
