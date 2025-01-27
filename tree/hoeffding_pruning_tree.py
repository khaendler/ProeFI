import math
import collections
import functools
import typing

from river import base
from river.utils.norm import normalize_values_in_dict

from river.metrics import Accuracy
from river.drift.adwin import ADWIN

from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch
from river.tree.splitter import Splitter
from river.tree import HoeffdingTreeClassifier

from ipfi.explainer import IncrementalPFI
from pruner.complete_pruner import CompletePruner
from pruner.selective_pruner import SelectivePruner


from tree.nodes.HTLeafWithPruneInfo import (
    HTLeafWithPruneInfo,
    LeafMajorityClassWithPruneInfo,
    LeafNaiveBayesWithPruneInfo,
    LeafNaiveBayesAdaptiveWithPruneInfo
)

from tree.nodes.StatBranch import (
    StatNumericBinaryBranch,
    StatNominalBinaryBranch,
    StatNumericMultiwayBranch,
    StatNominalMultiwayBranch
)

import matplotlib.pyplot as plt
import numpy as np


class HoeffdingPruningTree(HoeffdingTreeClassifier):
    """Hoeffding Pruning Tree using the VFDT classifier with incremental PFI to prune the tree.
    HPT uses the ADWIN estimation as the importance threshold to determine whether a feature is important
    enough to retain the nodes split on it.

    Note: This works with river versions 0.16.0. Later versions may not be supported.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    pruner
        The Pruner to use.</br>
        - 'selective' - SelectivePruner</br>
        - 'complete' - CompletePruner</br>
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        pruner: str = "complete",
        seed: int = None
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.incremental_pfi: typing.Optional[IncrementalPFI] = None
        self.importance_threshold = 0.02  # standard value for the beginning

        self.feature_names = None
        self.important_features = None
        self.last_important_features = None

        self.pruner = self._set_pruner(pruner)
        self.seed = seed

        # For threshold
        self.importance_adwin = ADWIN()

        # {feature: (creation, death)}
        self.branch_lifetimes = None

    @property
    def root(self):
        return self._root

    @property
    def importance_values(self):
        return self.incremental_pfi.importance_values

    @property
    def collected_importance_values(self):
        return self.incremental_pfi.collected_importance_values

    @property
    def collected_normalized_importance_values(self):
        return self.incremental_pfi.collected_normalized_importance_values

    def set_new_root(self, node: HTLeaf | DTBranch):
        self._root = node

    def _branch_selector(self, numerical_feature=True, multiway_split=False) -> type[DTBranch]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return StatNumericBinaryBranch
            else:
                return StatNumericMultiwayBranch
        else:
            if not multiway_split:
                return StatNominalBinaryBranch
            else:
                return StatNominalMultiwayBranch

    def create_new_leaf(self, initial_stats: dict | None = None, parent: HTLeafWithPruneInfo | DTBranch | None = None,
                        prune_info: dict | None = None):
        return self._new_leaf(initial_stats, parent, prune_info)

    def _new_leaf(self, initial_stats=None, parent=None, prune_info=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LeafMajorityClassWithPruneInfo(initial_stats, depth, self.splitter, prune_info=prune_info)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LeafNaiveBayesWithPruneInfo(initial_stats, depth, self.splitter, prune_info=prune_info)
        else:  # Naives Bayes Adaptive (default)
            return LeafNaiveBayesAdaptiveWithPruneInfo(initial_stats, depth, self.splitter, prune_info=prune_info)

    def _update_leaf_counts(self):
        leaves = [leaf for leaf in self._root.iter_leaves()]
        self._n_active_leaves = sum(1 for leaf in leaves if leaf.is_active())
        self._n_inactive_leaves = sum(1 for leaf in leaves if not leaf.is_active())

    def _set_pruner(self, pruner):
        if pruner == "selective":
            return SelectivePruner(self)
        elif pruner == "complete":
            return CompletePruner(self)
        else:
            raise ValueError(f"Invalid pruner type: {pruner}. Valid options are 'selective' or 'complete'.")

    def learn_one(self, x, y, *, w=1.0):
        # Initialize the incremental PFI instance.
        if self.incremental_pfi is None:
            self.feature_names = list(x.keys())
            self.important_features = self.feature_names
            self.last_important_features = set(self.feature_names)
            self._create_ipfi()

            self.branch_lifetimes = {feature: {} for feature in self.feature_names}

        self._update_ipfi(x, y)
        if self.importance_values:
            self._update_importance_threshold()

        # Prune the tree if the set of important features has changed.
        if self.last_important_features != set(self.important_features):
            self.pruner.prune_tree()
            self._update_leaf_counts()

        self.last_important_features = set(self.important_features)

        # learning
        #super().learn_one(x, y, sample_weight=1.0)

        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, HTLeaf):
            node.learn_one(x, y, w=w, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, HTLeaf):
                    break
            # Learn from the sample
            node.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch, parent_branch: int, **kwargs):
        """Attempt to split a leaf.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node (branch node).
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attributes. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The leaf to evaluate.
        parent
            The leaf's parent.
        parent_branch
            Parent leaf's branch index.
        kwargs
            Other parameters passed to the new branch.
        """
        if not leaf.observed_class_distribution_is_pure():  # type: ignore
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions = self._include_fi_in_merit(best_split_suggestions)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )
                    new_split.creation_instance = int(self._train_weight_seen_by_model)
                    self.branch_lifetimes[split_decision.feature][int(self._train_weight_seen_by_model)] = math.inf

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                self._enforce_size_limit()

    def _create_ipfi(self):
        self.incremental_pfi = IncrementalPFI(
            model_function=self.predict_one,
            loss_function=Accuracy(),
            smoothing_alpha=0.001,
            seed=self.seed
        )

    def _update_ipfi(self, x, y):
        """Updates iPFI, PFI plotter and the list of current important features based on the importance threshold."""

        # explaining
        self.incremental_pfi.explain_one(x, y)

        # Check if any feature's importance value meets or exceeds the threshold.
        # If so, select those features as important. Otherwise, consider all features as important.
        if any(value >= self.importance_threshold for value in self.importance_values.values()):
            self.important_features = [k for k, v in self.importance_values.items()
                                       if v >= self.importance_threshold]
        else:
            self.important_features = self.feature_names

    def _update_importance_threshold(self):
        average_importance_value = sum(self.importance_values.values()) / len(self.importance_values)
        self.importance_adwin.update(average_importance_value)
        self.importance_threshold = self.importance_adwin.estimation

    def _include_fi_in_merit(self, split_suggestions):
        """ Nothing. Currently only for HPTMerit """
        return split_suggestions

    def plot_feature_importance(self, names_to_highlight, normalized=False):
        self.incremental_pfi.plot(names_to_highlight=names_to_highlight, normalized=normalized)

    def draw(self, max_depth: int | None = None):
        """Draw the tree using the `graphviz` library. Includes prune_info of leaves.

        Since the tree is drawn without passing incoming samples, classification trees
        will show the majority class in their leaves, whereas regression trees will
        use the target mean.

        Parameters
        ----------
        max_depth
            Only the root will be drawn when set to `0`. Every node will be drawn when
            set to `None`.

        Notes
        -----
        Currently, Label Combination Hoeffding Tree Classifier (for multi-label
        classification) is not supported.

        """
        try:
            import graphviz
        except ImportError as e:
            raise ValueError("You have to install graphviz to use the draw method.") from e
        counter = 0

        def iterate(node=None):
            if node is None:
                yield None, None, self._root, 0, None
                yield from iterate(self._root)

            nonlocal counter
            parent_no = counter

            if isinstance(node, DTBranch):
                for branch_index, child in enumerate(node.children):
                    counter += 1
                    yield parent_no, node, child, counter, branch_index
                    if isinstance(child, DTBranch):
                        yield from iterate(child)

        if max_depth is None:
            max_depth = -1

        dot = graphviz.Digraph(
            graph_attr={"splines": "ortho", "forcelabels": "true", "overlap": "false"},
            node_attr={
                "shape": "box",
                "penwidth": "1.2",
                "fontname": "trebuchet",
                "fontsize": "11",
                "margin": "0.1,0.0",
            },
            edge_attr={"penwidth": "0.6", "center": "true", "fontsize": "7  "},
        )

        if isinstance(self, base.Classifier):
            n_colors = len(self.classes)  # type: ignore
        else:
            n_colors = 1

        # Pick a color palette which maps classes to colors
        new_color = functools.partial(next, iter(_color_brew(n_colors)))
        palette: typing.DefaultDict = collections.defaultdict(new_color)

        for parent_no, parent, child, child_no, branch_index in iterate():
            if child.depth > max_depth and max_depth != -1:
                continue

            if isinstance(child, DTBranch):
                text = f"{child.feature}"  # type: ignore
            else:
                # Checks whether the leaf was created due to pruning.
                if child.prune_info is None:
                    text = f"{repr(child)}\nsamples: {int(child.total_weight)}\n"
                else:
                    text = (f"{repr(child)}\nsamples: {int(child.total_weight)}\n"
                            f"pruned {child.prune_info[1]} split node\n"
                            f"at instance: {child.prune_info[0]}\n")

            # Pick a color, the hue depends on the class and the transparency on the distribution
            if isinstance(self, base.Classifier):
                class_proba = normalize_values_in_dict(child.stats, inplace=False)
                mode = max(class_proba, key=class_proba.get)
                p_mode = class_proba[mode]
                try:
                    alpha = (p_mode - 1 / n_colors) / (1 - 1 / n_colors)
                    fillcolor = str(transparency_hex(color=palette[mode], alpha=alpha))
                except ZeroDivisionError:
                    fillcolor = "#FFFFFF"
            else:
                fillcolor = "#FFFFFF"

            dot.node(f"{child_no}", text, fillcolor=fillcolor, style="filled")

            if parent_no is not None:
                dot.edge(
                    f"{parent_no}",
                    f"{child_no}",
                    xlabel=parent.repr_branch(branch_index, shorten=True),
                )

        return dot

    def plot_active_nodes_with_feature(self, feature):
        """ Plots the active nodes over time with the importance values for a given feature.

        Parameters:
        - data: List of dictionaries, where each dictionary represents {creation_time: elimination_time}.
        - importance_values: List of dictionaries containing importance values over time.
        - feature: The feature to plot the importance values for.
        """
        # Extract events from creation and elimination times
        events = []
        for creation, elimination in self.branch_lifetimes[feature].items():
            events.append((creation, True))
            events.append((elimination, False))

        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Calculate active instances over time
        active_instances = 0
        times = []
        active_count = []

        for event in events:
            if event[0] != math.inf:
                times.append(event[0])
                if event[1]:
                    active_instances += 1
                else:
                    active_instances -= 1
                active_count.append(active_instances)

        times = [0] + times + [len(self.collected_importance_values["importance_values"])]
        active_count = [0] + active_count + [active_count[-1]]
        print(times, active_count)

        # Extract importance values for the specified feature
        feature_values = np.array([entry.get(feature, 0.0) for entry in self.collected_importance_values["importance_values"]])
        time_steps = np.arange(len(feature_values))

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time')
        color_left = 'tab:blue'
        ax1.set_ylabel('Node Count', color=color_left)
        ax1.step(times, active_count, color=color_left, label='Nodes')
        ax1.tick_params(axis='y', labelcolor=color_left)

        ax2 = ax1.twinx()

        color_right = 'tab:red'
        ax2.set_ylabel('Normalized Feature Importance', color=color_right)
        ax2.plot(time_steps, feature_values, color=color_right, label=f'Feature Importance')
        ax2.tick_params(axis='y', labelcolor=color_right)

        plt.title(f'Node Count and Feature Importance of "{feature}" Over Time')
        fig.tight_layout()
        plt.show()


def _color_brew(n: int) -> list[tuple[int, int, int]]:
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n
        The number of required colors.

    Returns
    -------
        List of n tuples of form (R, G, B) being the components of each color.
    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_export.py
    """
    colors = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in [i for i in range(25, 385, int(360 / n))]:
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))

        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]

        # Shift the initial RGB values to match value and store
        colors.append(((int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))))

    return colors


# Utility adapted from the original creme's implementation
def transparency_hex(color: tuple[int, int, int], alpha: float) -> str:
    """Apply alpha coefficient on hexadecimal color."""
    return "#{:02x}{:02x}{:02x}".format(
        *tuple([int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color])
    )
