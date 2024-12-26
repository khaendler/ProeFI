from river.drift.adwin import ADWIN
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch
from river.tree.splitter import Splitter
from river.metrics import Accuracy
from river.tree import HoeffdingTreeClassifier

from reproducible_ipfi.ipfi import IPFI
from ixai.utils.wrappers import RiverWrapper
from ixai.visualization import FeatureImportancePlotter

from scaler.MinMaxScaler import MinMaxScaler


class HTMerit(HoeffdingTreeClassifier):
    """ Hoeffding tree to experiment with the merit for splitting.

    Note: This works with river versions 0.16.0. Later versions may not be supported.
    """
    def __init__(
            self,
            grace_period: int = 200,
            max_depth: int | None = None,
            split_criterion: str = "info_gain",
            delta: float = 1e-7,
            tau: float = 0.05,
            leaf_prediction: str = "nba",
            nb_threshold: int = 0,
            nominal_attributes: list | None = None,
            splitter: Splitter | None = None,
            binary_split: bool = False,
            max_size: float = 100.0,
            memory_estimate_period: int = 1000000,
            stop_mem_management: bool = False,
            remove_poor_attrs: bool = False,
            merit_preprune: bool = True,
            seed: int = 42
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
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.incremental_pfi = None
        self.pfi_plotter = None
        self.feature_names = None
        self.seed = seed

        # For threshold
        self.importance_adwin = ADWIN()
        self.scaler = MinMaxScaler()

    @property
    def root(self):
        return self._root

    @property
    def importance_values(self):
        return self.incremental_pfi.importance_values

    @property
    def collected_importance_values(self):
        return self.pfi_plotter.y_data

    def learn_one(self, x, y, *, sample_weight=1.0):
        # Initialize the incremental PFI instance.
        if self.incremental_pfi is None:
            self.feature_names = list(x.keys())
            self._create_ipfi()

        self._update_ipfi(x, y)
        super().learn_one(x, y, sample_weight=1.0)

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

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                self._enforce_size_limit()

    def _create_ipfi(self):
        self.incremental_pfi = IPFI(
            model_function=RiverWrapper(self.predict_one),
            loss_function=Accuracy(),
            feature_names=self.feature_names,
            smoothing_alpha=0.001,
            n_inner_samples=5,
            seed=self.seed
        )

        self.pfi_plotter = FeatureImportancePlotter(feature_names=self.feature_names)

    def _update_ipfi(self, x, y):
        """Updates iPFI, PFI plotter and the list of current important features based on the importance threshold."""
        inc_fi_pfi = self.incremental_pfi.explain_one(x, y)
        self.pfi_plotter.update(inc_fi_pfi)
        self.scaler.learn_one(self.importance_values)

    def _include_fi_in_merit(self, split_suggestions):
        """ Includes scaled FI in the merit of each feature. """
        self.scaled_fi = self.scaler.transform_one(self.incremental_pfi.importance_values)
        for branch in split_suggestions:
            if branch.feature in self.incremental_pfi.importance_values.keys():
                branch.merit *= self.scaled_fi[branch.feature]
        return split_suggestions
