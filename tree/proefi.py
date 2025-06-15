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


from tree.nodes.HTLeafWithPruneInfo import (
    HTLeafWithPruneInfo,
    LeafMajorityClassWithPruneInfo,
    LeafNaiveBayesWithPruneInfo,
    LeafNaiveBayesAdaptiveWithPruneInfo
)


class ProeFI(HoeffdingTreeClassifier):
    """Pruning Hoeffding Trees by the Importance of Features (ProeFI) using the VFDT classifier with incremental PFI
    to prune the tree. ProeFI uses the ADWIN estimation as the importance threshold to determine whether a feature
    is important enough to retain the nodes split on it.

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

        self.pruner = CompletePruner(self)
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
        super().learn_one(x, y, w=w)

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

    def plot_feature_importance(self, top_k=4, names_to_highlight=None, normalized=False):
        self.incremental_pfi.plot(top_k=top_k, names_to_highlight=names_to_highlight, normalized=normalized)

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
            graph_attr={"center": "true", "splines": "ortho", "forcelabels": "true", "overlap": "false"},
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
                            f"pruned '{child.prune_info[1]}' split node\n"
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
