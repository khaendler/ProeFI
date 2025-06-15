from river.metrics.base import Metric


class RiverMetricToLossFunction:
    """Wrapper that transforms a river.metrics.base.Metric into a loss function.

    This Wrapper turns metrics that expect a single value as predictions (e.g. river.metrics.MAE, or
    river.metrics.Accuracy) or metrics that expect a dictionary as predictions (e.g. river.metrics.CrossEntropy) into
    a similar interface.
    """

    def __init__(self, river_metric: Metric):
        """
        Args:
            river_metric (river.Metric): The river metric to be used as a loss function.
        """
        self._river_metric = river_metric

        if river_metric.bigger_is_better:
            self._sign = -1.
        else:
            self._sign = 1.

    def __call__(self, y_true, y_prediction):
        """Calculates the loss given for a single prediction given its true (expected) value.

        Args:
            y_true (Any): The true labels.
            y_prediction (Any): The predicted values.

        Returns:
            The loss value given the true and predicted labels.
        """
        try:
            self._river_metric.update(y_true=y_true, y_pred=y_prediction)
            loss_i = self._river_metric.get()
            self._river_metric.revert(y_true=y_true, y_pred=y_prediction)

        except Exception:
            raise AttributeError("Provided model function does not fit the given loss function.")

        return loss_i * self._sign
