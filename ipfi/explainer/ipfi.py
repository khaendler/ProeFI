import random
import itertools
from typing import Union, Dict, Callable, Any, Optional

import matplotlib.pyplot as plt
from river.metrics.base import Metric

from ipfi.scaler import MinMaxScaler
from wrapper import RiverMetricToLossFunction
from ipfi.storage import BaseReservoirStorage, GeometricReservoirStorage, UniformReservoirStorage


class IncrementalPFI:

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

    def plot(self, names_to_highlight, title="Feature Importance over time", normalized=False):

        plt.style.use("bmh")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor('#f8f8f8')

        x = range(len(self.collected_importance_values))
        if normalized:
            fi_values = self.collected_normalized_importance_values
        else:
            fi_values = self.collected_importance_values

        all_keys = list(fi_values[0].keys())

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = itertools.cycle(default_colors)

        key_to_color = {}
        for key in names_to_highlight:
            key_to_color[key] = next(color_cycle)

        for key in all_keys:
            y = [row[key] for row in fi_values]

            if key in names_to_highlight:
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
