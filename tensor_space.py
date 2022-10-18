from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Optional, List


class DuplicateSpaceError(Exception):
    """
    Thrown when a name for a space is already taken, but the user tries to define a new space with the same name and
    different properties.
    """


class SpaceIterator(Iterator):
    """
    Iterator to pass through Space objects and their subspaces.
    Following the example from https://refactoring.guru/design-patterns/iterator/python/example

    We want to have a standardized way to iterate through space couplings. Here is one example how we want the traversal
    to go for three spaces, labeled 1, 2, and 3
    {{1 x 2} x {1 x 3}}
    0. {{1 x 2} x {1 x 3}}
    1. {1 x 2}
    2. 1
    3. 2
    4. {1 x 3}
    5. 1
    6. 3
    """

    def __init__(self, space: TensorSpace):
        """
        Construct a _collection over which we will iterate later on.
        :param space: TensorSpace
        """
        self._position = 0
        self._substructure = space.substructure
        self._collection = []  # the list that is traversed during iterations, needs to be constructed
        element_list = [0]  # keeps track of the structure and is constructed during the initialization
        while len(element_list) > 0:
            collection_element = self._get_substructure(element_list)
            if collection_element:
                self._collection.append(collection_element)
                element_list.append(0)
            else:
                element_list = element_list[:-1]
                if element_list[-1] == 1:
                    # We have to find the last 0 element in the list and switch this to a one. The rest must be removed.
                    try:
                        index = element_list[::-1].index(0)
                    except ValueError:
                        # We reached the last element
                        break
                    element_list = element_list[:-index]
                element_list[-1] = 1

    def _get_substructure(self, element_list: list[int]) -> Optional[TensorSpace]:
        """
        Auxiliary function for the iterator initialization.
        The idea is to loop through all subspaces of the space to construct the _collection list, which is later used
        in the iteration steps.
        
        :param element_list: A list of integers, which provides a map of the sub-spaces of the space.
        :return: Sub-Space or None
        """
        out_space = self._substructure
        for pos in element_list[:-1]:
            out_space = out_space[pos].substructure
        if out_space:
            return out_space[element_list[-1]]
        else:
            return None

    def __next__(self) -> TensorSpace:
        """
        Proceed to the next element.

        :return: The next-in-line TensorSpace element
        """
        try:
            value = self._collection[self._position]
            self._position += 1
        except IndexError:
            raise StopIteration()
        return value

    def get_next(self) -> Optional[TensorSpace]:
        """
        Same as the __next__ function but without raising an exception.

        :return:
        """
        try:
            value = self._collection[self._position]
            self._position += 1
        except IndexError:
            return None
        return value


class TensorSpace(Iterable):
    """
    An iterable class used to define the space of operators and states. Objects of this class can be compared and their
    subspace can be traversed with the SpaceIterator.
    """

    space_dict = {}  # Keep track which spaces already exist.

    def __init__(self, name: str, order: int, substructure: Optional[List[TensorSpace]] = None):
        self._name = name
        self._order = order
        self._substructure = substructure
        self._pure_space = not substructure
        if name not in self.space_dict.keys():
            self.space_dict[name] = self
        else:
            other = self.space_dict[name]
            if not(other.order == self.order and other.substructure == self.substructure):
                raise DuplicateSpaceError(f'A space with the name "{name}" and different properties already exists.')

    @property
    def name(self):
        return self._name

    @property
    def pure_space(self):
        return self._pure_space

    @property
    def order(self):
        return self._order

    @property
    def substructure(self):
        return self._substructure

    def __str__(self) -> str:
        return f"{self.name}-space: order = {self.order}"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: TensorSpace) -> TensorSpace:
        """
        Combine two spaces, if they are identical, return the space. Else, construct a new space, which contains the
        two spaces as a substructure

        :param other: TensorSpace
        :return: TensorSpace
        """
        if self == other:
            return self
        else:
            new_name = '{' + self.name + ' x ' + other.name + '}'
            new_order = min(self.order, other.order)
            return self.__class__(new_name, new_order, substructure=[self, other])

    def __eq__(self, other: TensorSpace) -> bool:
        """
        TensorSpaces cannot share a name if they have different properties. For equality, it is enough to check the name

        :param other: TensorSpace
        :return: True or False
        """
        if self.name == other.name:
            return True
        else:
            return False

    def __gt__(self, other: TensorSpace):
        """
        Order spaces according to their definition. The order is the first criteria. If there are coupled spaces, we
        further prefer shallower space to deep spaces. If those are the same we iterate through the spaces and repeat
        these criteria for each subspace. See also the SpaceIterator class. In case only one object has a substructure,
        we want this one on the first position.

        :param other: TensorSpace
        :return: True or False
        """
        if self.order > other.order:
            return True  # Those should be switched
        elif self.order == other.order:
            if self.substructure and other.substructure:
                if self.get_depth() < other.get_depth():
                    # both have identical order now the deepest substructure loses
                    return True
                elif self.get_depth() == other.get_depth():
                    self_iterator = SpaceIterator(self)
                    other_iterator = SpaceIterator(other)
                    next_self = self_iterator.get_next()
                    other_self = other_iterator.get_next()
                    while next_self and other_self:
                        if next_self.order > other_self.order:
                            return True
                        next_self = self_iterator.get_next()
                        other_self = other_iterator.get_next()
            elif self.substructure:
                return False  # Objects with substructure have higher rank by choice
            elif other.substructure:
                return True  # Objects with substructure have higher rank by choice
        return False

    def __ge__(self, other: TensorSpace):
        if self == other:
            return True
        elif self > other:
            return True
        return False

    def __lt__(self, other: TensorSpace):
        return not self >= other

    def __le__(self, other: TensorSpace):
        return not self > other

    def __ne__(self, other: TensorSpace):
        return not self == other

    def __iter__(self) -> SpaceIterator:
        return SpaceIterator(self)

    def get_depth(self) -> int:
        """
        Get the depth of the TensorSpace, by going recursively through the substructure.

        :return: Integer
        """
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth

    def contains(self, other: TensorSpace) -> bool:
        """
        Used to check if this TensorSpace contains the other space within it.

        :param other: TensorSpace
        :return: True or False
        """
        if self == other:
            return True
        if self.get_depth() == 0:
            return False
        for space in self:
            if space == other:
                return True
        return False

    def get_flat_basic_states(self) -> list[TensorSpace]:
        """
        Finds all basic TensorSpace objects (i.e., those without a substructure) and returns them as a list.
        :return: list of basic states
        """
        basic_states = []
        flat_basic_states = []
        if self.pure_space:
            basic_states.append(self)
        else:
            subspace_1, subspace_2 = self.substructure
            basic_states.extend(subspace_1.get_flat_basic_states())
            basic_states.extend(subspace_2.get_flat_basic_states())
        for bs in basic_states:
            if bs not in flat_basic_states:
                flat_basic_states.append(bs)   # Need to do it this way since the entries are unhashable
        if flat_basic_states:
            flat_basic_states.sort(key=lambda x: x.order)
        return flat_basic_states
