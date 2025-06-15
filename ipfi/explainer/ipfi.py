import random
import itertools
from typing import Union, Dict, Callable, Any, Optional

import matplotlib.pyplot as plt
from river.metrics.base import Metric

from ipfi.scaler import MinMaxScaler
from wrapper import RiverMetricToLossFunction
from ipfi.storage import BaseReservoirStorage, GeometricReservoirStorage, UniformReservoirStorage


class IncrementalPFI:
    """Incremental PFI Explainer

    Computes PFI importance values incrementally by applying exponential smoothing.
    For each input instance tuple x_i, y_i one update of the explanation procedure is performed.

    Parts of this code are adapted from https://github.com/mmschlk/iXAI.

    Args:
        model_function (Callable[[Any], Any]): The Model function to be explained (e.g.
            model.predict_one (river), model.predict_proba (sklearn)).
        loss_function (Union[Metric, Callable[[Any, Dict], float]]): The loss function for which
            the importance values are calculated. This can either be a callable function or a
            predefined river.metric.base.Metric.<br>
            - river.metric.base.Metric: Any Metric implemented in river (e.g.
                river.metrics.CrossEntropy() for classification or river.metrics.MSE() for
                regression).<br>
            - callable function: The loss_function needs to follow the signature of
                loss_function(y_true, y_pred) and handle the output dimensions of the model
                function. Smaller values are interpreted as being better if not overriden with
                `loss_bigger_is_better=True`. `y_pred` is passed as a dict.
        smoothing_alpha (float, optional): The smoothing parameter for the exponential smoothing
            of the importance values. Should be in the interval between ]0,1].
            Defaults to 0.001.
        storage (BaseStorage, optional): Optional incremental data storage Mechanism.
            Defaults to `GeometricReservoirStorage(size=100)` for dynamic modelling settings
            (`dynamic_setting=True`) and `UniformReservoirStorage(size=100)` in static modelling
            settings (`dynamic_setting=False`).
        dynamic_setting (bool): Flag to indicate if the modelling setting is dynamic `True`
            (changing model, and adaptive explanation) or a static modelling setting `False`
            (all observations contribute equally to the final importance). Defaults to `True`.
        collect_values (bool): Whether to store the feature importance values or not.
        seed (int, optional): Random seed for reproducibility.
    """
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            smoothing_alpha: float = 0.001,
            storage: Optional[BaseReservoirStorage] = None,
            dynamic_setting: bool = True,
            collect_values: bool = True,
            seed: Optional[int] = None
    ):

        self._model_function = model_function
        self._loss_function = RiverMetricToLossFunction(loss_function)
        self.smoothing_alpha = smoothing_alpha

        if storage is not None:
            self._storage = storage
        elif dynamic_setting:
            self._storage = GeometricReservoirStorage(size=100, seed=seed)
        else:
            self._storage = UniformReservoirStorage(size=100, seed=seed)

        self.collect_values = collect_values
        self.collected_importance_values = []

        self.importance_values = {}
        self.feature_names = []
        self.scaler = MinMaxScaler()
        self.seen_samples = 0

        self._rng = random.Random(seed)

    @property
    def normalized_importance_values(self):
        return self.scaler.transform_one(self.importance_values)

    @property
    def collected_normalized_importance_values(self):
        collected_normalized_importance_values = [
            self.scaler.transform_one(importance_values)
            for importance_values in self.collected_importance_values
        ]
        return collected_normalized_importance_values

    def explain_one(self, x: dict, y: dict):
        if self.seen_samples > 0:
            original_prediction = self._model_function(x)
            original_loss = self._loss_function(original_prediction, y)
            pfi = {}
            for feature in self.feature_names:
                # sample from reservoir
                sampled_feature = self.sample(feature)
                # predict and evaluate on instance using sampled_feature
                prediction = self._model_function({**x, **sampled_feature})
                loss = self._loss_function(y, prediction)
                # calculate incremental pfi with exponential smoothing
                pfi[feature] = (
                        (1 - self.smoothing_alpha) * self.importance_values[feature]
                        + self.smoothing_alpha * (loss - original_loss))

            self.importance_values = pfi

            if self.collect_values:
                self.collected_importance_values.append(self.importance_values)

            self.update_scaler()

        else:
            self.feature_names = list(x.keys())
            self.importance_values = {feature: 0 for feature in self.feature_names}

        self._storage.update(x)
        self.seen_samples += 1

        return self.importance_values

    def sample(self, feature):
        rand_idx = self._rng.randrange(len(self._storage))
        sampled_instance = self._storage[rand_idx].copy()
        return {feature: sampled_instance[feature]}

    def update_scaler(self):
        self.scaler.learn_one(self.importance_values)

    def plot(self, top_k=4, names_to_highlight=None, title="Feature Importance over time", normalized=False):

        if names_to_highlight is None:
            names_to_highlight = []

        plt.style.use("bmh")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor('#f8f8f8')

        x = range(len(self.collected_importance_values))
        if normalized:
            fi_values = self.collected_normalized_importance_values
        else:
            fi_values = self.collected_importance_values

        all_keys = list(fi_values[0].keys())

        highlight_set = set(names_to_highlight)
        if 0 < top_k < len(all_keys) + 1:
            totals = {key: sum(abs(row[key]) for row in fi_values) for key in all_keys}
            top_keys = sorted(totals, key=totals.get, reverse=True)[:top_k]
            highlight_set.update(top_keys)

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = itertools.cycle(default_colors)
        key_to_color = {key: next(color_cycle) for key in highlight_set}

        for key in all_keys:
            y = [row[key] for row in fi_values]
            if key in key_to_color:
                color = key_to_color[key]
                label = key
                z = 3
            else:
                color = 'lightgrey'
                label = None
                z = 1

            ax.plot(x, y, color=color, label=label, linewidth=1, zorder=z)

        plt.legend()
        plt.xlabel("Instances")
        plt.ylabel("Feature importance")
        plt.title(title)
        plt.show()
