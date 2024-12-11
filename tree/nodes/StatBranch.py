from __future__ import annotations
from river.tree.nodes.branch import *


class StatDTBranch(DTBranch):
    def __init__(self, stats, *children, **attributes):
        super().__init__(stats, *children, **attributes)
        self.creation_instance = None

    @property
    def total_weight(self):
        return sum(child.total_weight for child in filter(None, self.children))

    @abc.abstractmethod
    def branch_no(self, x):
        pass

    def next(self, x):
        return self.children[self.branch_no(x)]

    @abc.abstractmethod
    def max_branches(self):
        pass

    @abc.abstractmethod
    def repr_branch(self, index: int, shorten=False):
        """Return a string representation of the test performed in the branch at `index`.

        Parameters
        ----------
        index
            The branch index.
        shorten
            If True, return a shortened version of the performed test.
        """
        pass


class StatNumericBinaryBranch(StatDTBranch, NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)

    def branch_no(self, x):
        if x[self.feature] <= self.threshold:
            return 0
        return 1

    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return f"≤ {round(self.threshold, 4)}"
            return f"> {round(self.threshold, 4)}"
        else:
            if index == 0:
                return f"{self.feature} ≤ {self.threshold}"
            return f"{self.feature} > {self.threshold}"

    @property
    def repr_split(self):
        return f"{self.feature} ≤ {self.threshold}"


class StatNominalBinaryBranch(StatDTBranch, NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)

    def branch_no(self, x):
        if x[self.feature] == self.value:
            return 0
        return 1

    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return str(self.value)
            else:
                return f"not {self.value}"
        else:
            if index == 0:
                return f"{self.feature} = {self.value}"
            return f"{self.feature} ≠ {self.value}"

    @property
    def repr_split(self):
        return f"{self.feature} {{=, ≠}} {self.value}"


class StatNumericMultiwayBranch(StatDTBranch, NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **attributes):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **attributes)

    def branch_no(self, x):
        slot = math.floor(x[self.feature] / self.radius)

        return self._mapping[slot]

    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(range(len(self.children)), key=lambda i: self.children[i].total_weight)

        return pos, self.children[pos]

    def add_child(self, feature_val, child):
        slot = math.floor(feature_val / self.radius)

        self._mapping[slot] = len(self.children)
        self._r_mapping[len(self.children)] = slot
        self.children.append(child)

    def repr_branch(self, index: int, shorten=False):
        lower = self._r_mapping[index] * self.radius
        upper = lower + self.radius

        if shorten:
            return f"[{round(lower, 4)}, {round(upper, 4)})"

        return f"{lower} ≤ {self.feature} < {upper}"

    @property
    def repr_split(self):
        return f"{self.feature} ÷ {self.radius}"


class StatNominalMultiwayBranch(StatDTBranch, NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)

    def branch_no(self, x):
        return self._mapping[x[self.feature]]

    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(range(len(self.children)), key=lambda i: self.children[i].total_weight)

        return pos, self.children[pos]

    def add_child(self, feature_val, child):
        self._mapping[feature_val] = len(self.children)
        self._r_mapping[len(self.children)] = feature_val
        self.children.append(child)

    def repr_branch(self, index: int, shorten=False):
        feat_val = self._r_mapping[index]

        if shorten:
            return str(feat_val)

        return f"{self.feature} = {feat_val}"

    @property
    def repr_split(self):
        return f"{self.feature} in {set(self._mapping.keys())}"