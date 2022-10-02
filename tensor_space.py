from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Optional, List


class SpaceIterator(Iterator):
    """
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
        self._position = 0
        self._substructure = space.substructure
        self._collection = []
        element_list = [0]
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

    def _get_substructure(self, element_list):
        out_space = self._substructure
        for pos in element_list[:-1]:
            out_space = out_space[pos].substructure
        if out_space:
            return out_space[element_list[-1]]
        else:
            return None

    def __next__(self):
        try:
            value = self._collection[self._position]
            self._position += 1
        except IndexError:
            raise StopIteration()
        return value

    def get_next(self):
        try:
            value = self._collection[self._position]
            self._position += 1
        except IndexError:
            return None
        return value


class TensorSpace(Iterable):

    space_dict = {}

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
                raise AttributeError(f'A space with the name "{name}" and different properties already exists.')

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

    @staticmethod
    def _create_new(space_1, space_2):
        new_name = f"[{space_1.name} x {space_2.name}]"
        new_order = space_1.order
        substructure = [space_1, space_2]
        return new_name, new_order, substructure

    def __str__(self):
        return f"{self.name}-space: order = {self.order}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: TensorSpace):
        if self == other:
            return self
        else:
            new_name = '{' + self.name + ' x ' + other.name + '}'
            new_order = min(self.order, other.order)
            return self.__class__(new_name, new_order, substructure=[self, other])

    def __eq__(self, other: TensorSpace):
        if self.name == other.name:
            return True
        else:
            return False

    def __gt__(self, other: TensorSpace):
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

    def get_depth(self):
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth

    def contains(self, other: TensorSpace) -> bool:
        if self == other:
            return True
        if self.get_depth() == 0:
            return False
        for space in self:
            if space == other:
                return True
        return False

    def get_flat_basic_states(self):
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


default_space = TensorSpace('DEFAULT', 99)


if __name__ == "__main__":
    relative_space = TensorSpace('relative', 0)
    spin_space = TensorSpace('spin', 1)
    cm_space = TensorSpace('center of mass', 2)

    so_space = relative_space + relative_space
    soo_space = relative_space + spin_space + relative_space
    sso_space = spin_space + relative_space + spin_space
    socm_space = relative_space + spin_space + cm_space

    print(so_space > soo_space)
    print(soo_space > so_space)
    print(socm_space > soo_space)
    print(sso_space)

    print(sso_space.contains(cm_space))
    print(sso_space.contains(spin_space))
    print(sso_space.contains(relative_space))

    print(sso_space.get_flat_basic_states())
