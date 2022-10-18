from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Optional
from copy import deepcopy

from tensor_space import TensorSpace, default_space


class TensorOperatorInterface(ABC):
    """
    Interface for the TensorOperator class and the TensorOperatorComposite class
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
        """
        Multiply a float/integer/symbol/etc. to the factor of this tensor operator / composite

        :param factor: float/integer/symbol/etc.
        :return: A new tensor operator / composite with the new factor = existing factor * factor
        """
        pass

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __truediv__(self, other):
        return self * (1/other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    @abstractmethod
    def to_expression(self) -> str:
        """
        Return a string expression for the given object. The expression for a Tensor Operator should show the factor
        times an expression for the tensor operator, e.g., 15 * {A_a x B_b}_c. For composites, it should return the
        string expression of all children, connected with a + sign.

        :return: String
        """
        pass

    @abstractmethod
    def to_expression_no_factor(self) -> str:
        """
        Return a string expression similar to to_expression, but without explicitly printing the factor.

        :return: String
        """
        pass

    @abstractmethod
    def add(self, other: TensorOperatorInterface) -> TensorOperatorInterface:
        """
        Add two TensorOperators/Composites.
        In case we add two TensorOperators -> TensorOperator with new factor if the operators are identical up to the
        factor. The new factor is just the sum of the old factors. Otherwise, return a TensorOperatorComposite with the
        two TensorOperators as Children. In all other Cases (e.g. adding Two TensorOperatorComposites) return a new
        composite with an updated list of children.

        :param other: TensorOperator or TensorOperatorComposite
        :return: TensorOperator or TensorOperatorComposite
        """
        pass

    @abstractmethod
    def couple(self, other: TensorOperatorInterface, rank, factor) -> TensorOperatorInterface:
        """
        Allows to couple TensorOperators and Composites with each other.
        E.g., A_a.couple(B_b, c, 15) -> 15 * {A_a x B_b}_c. Similar one can couple each instance in a Composite to an
        Operator or another Composite.

        :param other: TensorOperator or TensorOperatorComposite
        :param rank: The new rank to which the operators are coupled, checks for triangular equations
        :param factor: A new factor that is mutliplied to existing factors
        :return: TensorOperator or TensorOperatorComposite depending on the input or None if triangular equations are
        violated
        """
        pass


class TensorOperatorComposite(TensorOperatorInterface):
    """
    This class allows to create objects that hold a sum of TensorOperators, and still allows to perform calculations
    with its children.

    The class roughly follows a Composite Pattern, where this class is the composite and TensorOperators are the leafs.
    However, this class is flat. It cannot hold instances of itself.
    """

    def __init__(self, *args):
        self._children = [arg for arg in args if arg]
        self._simplify()

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        self._children = children
        self._simplify()

    def __mul__(self, factor):
        """
        Multiply each of the children with a factor

        :param factor: integer/float/symbol
        :return: A new instance of this class with an updated list of children
        """
        args = []
        for child in self.children:
            args.append(child * factor)
        return self.__class__(*args)

    def __eq__(self, other: TensorOperatorInterface) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if len(self.children) != len(other.children):
            return False
        return all([child1 == child2 for child1, child2 in zip(self.children, other.children)])

    def __ne__(self, other: TensorOperatorInterface) -> bool:
        return not self.__eq__(other)

    def to_expression(self) -> str:
        """
        Returns a string expression for the sum of all children.
        :return: String
        """
        return " + ".join([child.to_expression() for child in self.children])

    def to_expression_no_factor(self) -> str:
        """
        Returns a string expression for the sum of all children without explicitly showing the factors.
        :return: String
        """
        return " + ".join([child.to_expression_no_factor() for child in self.children])

    def add(self, other: TensorOperatorInterface) -> TensorOperatorComposite:
        """
        Extends the list of children depending on the input and returns a new instance of this class with the updated
        list.

        :param other: TensorOperator or TensorOperatorComposite
        :return: TensorOperatorComposite
        """
        if isinstance(other, self.__class__):
            args = self.children + other.children  # This keeps a flat hierarchy
        else:
            args = self.children + [other]
        new_object = self.__class__(*args)
        return new_object

    def couple(self, other: TensorOperatorInterface, rank, factor) -> TensorOperatorComposite:
        """
        Couple a TensorOperator or TensorOperatorComposite to all children of this object and return a new instance
        of this class with an updated list of children.

        :param other: TensorOperator or TensorOperatorComposite
        :param rank: The new rank to which the tensor operator should be coupled
        :param factor: A factor that is applied during the recoupling
        :return: A new TensorOperatorComposite
        """
        if isinstance(other, self.__class__):
            args = [child.couple(other_child, rank, factor) for child in self.children for other_child in
                    other.children]
        else:
            args = [child.couple(other, rank, factor) for child in self.children]

        new_object = self.__class__(*args)
        return new_object

    def _simplify(self):
        """
        Auxiliary function that simplifies the list of children in place.
        It removes zero entries and combines matching entries to one.
        This action is called whenever the children setter is used or a new instance of this object is instanciated.

        :return: None
        """
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
        self._children = new_children


class TensorOperator(TensorOperatorInterface):
    """
    This class holds basic (no substructure) and coupled components (substructure) of tensor operators.

    The class roughly follows a Composite Pattern, where the TensorOperatorComposite class is the composite and
    this class are the leafs.
    """

    def __init__(self, rank, factor=1, space: TensorSpace = default_space, symbol: Union[None, str, list[str]] = None,
                 substructure=None):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._symbol = symbol
        self._substructure = substructure

    def __mul__(self, factor) -> Optional[TensorOperator]:
        """
        Allows to multiply a factor to the tensor operator. In case the factor is 0, the method returns one. Otherwise
        a new instance of this class with an updated factor is returned.

        :param factor: inst/float/symbol
        :return: TensorOperator or None
        """
        new_factor = self.factor * factor
        if new_factor != 0:
            return self.__class__(self.rank, new_factor, self.space, self.symbol, self.substructure)
        else:
            return None

    def __eq__(self, other: Union[TensorOperatorComposite, TensorOperator]) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif self.space == other.space and self.symbol == other.symbol and self.rank == other.rank:
            return True
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

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

    def get_depth(self) -> int:
        """
        Auxiliary function to get the maximum coupling depth of the operator. This is done by going through the
        substructure recursively.

        :return: Integer value
        """
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth

    def add(self, other: Union[TensorOperator, TensorOperatorComposite]) \
            -> Union[TensorOperator, TensorOperatorComposite, None]:
        """
        Allows the addition of two tensor operators. If the tensor operators are identical up to the factor, return an
        updated instance of this object with the new factor, which is the sum of both factors. If this new factor is
        zero, return None. In all other cases, the method returns a TensorOperatorComposite.

        :param other: TensorOperator or TensorOperatorComposite
        :return: TensorOperator or TensorOperatorComposite or None
        """
        if isinstance(other, TensorOperatorComposite):
            return other + self
        elif isinstance(other, TensorOperator) and self.symbol == other.symbol and self.rank == other.rank:
            new_factor = self.factor + other.factor
            if new_factor != 0:
                return self.__class__(self.rank, new_factor, self.space, self.symbol, self.substructure)
            else:
                return None
        else:
            return TensorOperatorComposite(self, other)

    def to_expression(self) -> str:
        """
        A string expression of the operator at hand.

        :return: String
        """
        return str(self.factor) + " * " + self.to_expression_no_factor()

    def to_expression_no_factor(self) -> str:
        """
        A string expression of the operator at hand, without explicitly listing the factor.

        :return: String
        """
        return str(self.symbol).replace("[", "{").replace("]", "}").replace(', ', ' x ').replace("'", "") \
               + "_" + str(self.rank)

    def couple(self, other: TensorOperatorInterface, rank, factor, order=True) -> Optional[TensorOperatorInterface]:
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
        if isinstance(other, TensorOperatorComposite):
            args = [self.couple(other_child, rank, factor) for other_child in other.children]
            return TensorOperatorComposite(*args)
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
            new_symbol = [self.to_expression_no_factor(), other.to_expression_no_factor()]
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

    def commute(self) -> Optional[TensorOperator]:
        """
        In case this tensor operator has a substructure, it commutes the substructure and applies the correct factor to
        it. A new instance of this class is returned.
        :return: TensorOperator or None
        """
        if self.substructure:
            tensor_a, tensor_b = self.substructure
            new_factor = pow(-1, tensor_a.rank + tensor_b.rank - self.rank)
            new_top = tensor_b.couple(tensor_a, self.rank, new_factor * self.factor, order=False)
            return new_top
        else:
            return None

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
                new_top = self.commute()

        if new_top:
            self.factor = new_top.factor
            self.symbol = new_top.symbol
            self.space = new_top.space
            self.substructure = new_top.substructure

    def get_space_structure(self):
        """
        Goes recursively through all operators and obtains the space object of each operator. The function returns a
        (nested) List with all spaces.

        :return: (nested) list containing all spaces for each substructure operator.
        """
        depth = self.get_depth()
        if depth == 0:
            return self.space
        elif depth == 1:
            tensor_a, tensor_b = self.substructure
            return [tensor_a.space, tensor_b.space]
        else:
            tensors_a, tensors_b = self.substructure
            return [tensors_a.get_space_structure(), tensors_b.get_space_structure()]
