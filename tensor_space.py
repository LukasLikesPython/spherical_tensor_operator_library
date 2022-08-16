
class TensorSpace(object):

    order_name_dict = {}

    def __init__(self, name, order, substructure=None):
        self._name = name
        self._order = order
        self._substructure = substructure
        if name in self.order_name_dict.keys():
            print('[WARN] a state with that name already exists. This action is overriding the state in existence.')
        self.order_name_dict[name] = self

    def __str__(self):
        return f"{self.name}-space: order = {self.order}"

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

    def get_depth(self):
        if not self.substructure:
            return 0
        else:
            sub_structure_depth = max([sub_tensor.get_depth() for sub_tensor in self.substructure]) + 1
            return sub_structure_depth

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.order < other.order:
            return True
        elif self.order == other.order:
            if self.substructure:
                pass

    def __mul__(self, other):
        if self == other:
            return self
        else:
            if self.order == other.order:
                # coupled space before uncoupled
                if other.substructure:
                    new_name, new_order, substructure = self._create_new(other, self)
                else:
                    new_name, new_order, substructure = self._create_new(self, other)
            elif self.order < other.order:
                new_name, new_order, substructure = self._create_new(self, other)
            else:
                new_name, new_order, substructure = self._create_new(other, self)

            if new_name in self.order_name_dict.keys():
                return self.order_name_dict[new_name]
            return self.__class__(new_name, new_order, substructure)




if __name__ == "__main__":
    relative_space = TensorSpace('relative', 0)
    spin_space = TensorSpace('spin', 1)
    cm_space = TensorSpace('center of mass', 2)

    print(relative_space * relative_space)
    print(relative_space * spin_space * relative_space)
    print(relative_space * spin_space * cm_space)
