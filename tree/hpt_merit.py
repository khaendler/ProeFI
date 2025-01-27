from river.tree.splitter import Splitter

from tree.hoeffding_pruning_tree import HoeffdingPruningTree


class HPTMerit(HoeffdingPruningTree):
    """ HPT to experiment with the merit for pruning. """

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
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
            pruner=pruner,
            seed=seed
        )


    def _include_fi_in_merit(self, split_suggestions):
        """ Includes scaled FI in the merit of each feature. """
        self.scaled_fi = self.incremental_pfi.normalized_importance_values
        for branch in split_suggestions:
            if branch.feature in self.incremental_pfi.importance_values.keys():
                branch.merit *= self.scaled_fi[branch.feature]
        return split_suggestions

