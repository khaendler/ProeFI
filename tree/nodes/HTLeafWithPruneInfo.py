from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive


class HTLeafWithPruneInfo(HTLeaf):
    """ Leaf that stores information about the pruning which resulted into it being created.
    Stores instance and feature. """
    def __init__(self, stats, depth, splitter, prune_info=None, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        # (instance, feature)
        self.prune_info: tuple[int, str] = prune_info


class LeafMajorityClassWithPruneInfo(HTLeafWithPruneInfo, LeafMajorityClass):
    """Leaf that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, prune_info=None, **kwargs):
        super().__init__(stats, depth, splitter, prune_info, **kwargs)


class LeafNaiveBayesWithPruneInfo(HTLeafWithPruneInfo, LeafNaiveBayes):
    """Leaf that uses Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, prune_info=None, **kwargs):
        super().__init__(stats, depth, splitter, prune_info, **kwargs)


class LeafNaiveBayesAdaptiveWithPruneInfo(HTLeafWithPruneInfo, LeafNaiveBayesAdaptive):
    """Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, prune_info=None, **kwargs):
        super().__init__(stats, depth, splitter, prune_info, **kwargs)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0
