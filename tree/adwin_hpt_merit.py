from river.tree.splitter import Splitter
from tree.adwin_hpt import AdwinHPT


class AdwinHPTMerit(AdwinHPT):

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
            importance_threshold: float = 0.02,
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
            importance_threshold=importance_threshold,
            pruner=pruner,
            seed=seed
        )

    def _include_fi_in_merit(self, split_suggestions):
        """ Includes scaled FI in the merit of each feature. """
        self.scaled_fi = self.scaler.transform_one(self.incremental_pfi.importance_values)
        for branch in split_suggestions:
            if branch.feature in self.incremental_pfi.importance_values.keys():
                branch.merit *= self.scaled_fi[branch.feature]
        return split_suggestions

