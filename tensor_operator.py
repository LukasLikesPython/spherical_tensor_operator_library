from abc import ABC, abstractmethod


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
        if isinstance(other, TensorOperatorComposite):
            new_children = [child.couple(other_child, rank, factor) for child in self.children for other_child in
                            other.children]
        elif isinstance(other, TensorOperator):
            new_children = [child.couple(other, rank, factor) for child in self.children]
        else:
            raise (TypeError(f"Cannot couple type {type(other)} to an object of {type(self)}."))
        new_object = TensorOperatorComposite(new_children)
        new_object._simplify()
        return new_object

    def _simplify(self):
        children = self.children[::-1]
        new_children = []
        while len(children) > 0:
            child = children.pop(0)
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

    def __init__(self, rank=0, factor=1, space='1', representation=None):
        self._rank = rank
        self._factor = factor
        self._space = space
        self._representation = representation

    def __mul__(self, factor):
        new_factor = self.factor * factor
        return self.__class__(self.rank, new_factor, self.space, self.representation)

    def __eq__(self, other):
        if self.space == other.space and self.representation == other.representation and self.rank == other.rank:
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

    @property
    def space(self):
        return self._space

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
        if not abs(self.rank - other.rank) <= rank <= self.rank + other.rank:
            return 0  # Coupling those two operators to the new rank is not possible

        new_factor = self.factor * other.factor * factor

        if self.space == other.space:
            new_space = self.space
        else:
            new_space = [self.space, other.space]
            # TODO possibly I have to reorder here and use commutator relations
        new_representation = [self._to_expression_no_factor(), other._to_expression_no_factor()]

        return self.__class__(rank, new_factor, new_space, new_representation)


if __name__ == "__main__":
    top = TensorOperator(rank=1, factor=1, space='rel', representation="q")
    qsq = top.couple(top, 0, 3)
    print(qsq)
    print(top)

    tlist = qsq + 2 * top
    print(tlist)
    print(top)

    tlist = tlist + qsq + top
    print(tlist)
    tlist._simplify()
    print(tlist)
