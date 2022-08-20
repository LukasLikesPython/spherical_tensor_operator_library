
class TensorSpace(object):

    space_dict = {}

    def __init__(self, name, order, substructure=None):
        self._name = name
        self._order = order
        self._substructure = substructure
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

    def __add__(self, other):
        if self == other:
            return self
        else:
            new_name = '{' + self.name + ' x ' + other.name + '}'
            new_order = min(self.order, other.order)
            return self.__class__(new_name, new_order, substructure=[self, other])

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.order > other.order:
            return True  # Those should be switched
        elif self.order == other.order:
            if self.substructure and other.substructure:
                if max([sub.order for sub in self.substructure]) > max([sub.order for sub in other.substructure]):
                    return True
                elif self.get_depth() < other.get_depth():
                    # both have identical order now the deepest substructure loses
                    return True
            elif self.substructure:
                return False  # Objects with substructure have higher rank by choice
            elif other.substructure:
                return True  # Objects with substructure have higher rank by choice
        return False

    def __ge__(self, other):
        if self == other:
            return True
        elif self > other:
            return True
        return False

    def __lt__(self, other):
        return not self >= other

    def __le__(self, other):
        return not self > other

    def __ne__(self, other):
        return not self == other

    def get_depth(self):
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth





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
    print(sso_space > so_space)

