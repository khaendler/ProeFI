import sys
import numpy as np
from river.metrics import RollingROCAUC, Accuracy, CohenKappa
from river.utils import Rolling

from river.tree import (HoeffdingTreeClassifier,
                        HoeffdingAdaptiveTreeClassifier,
                        ExtremelyFastDecisionTreeClassifier)

from river.datasets import Elec2

from tree import HoeffdingPruningTree, HPTMerit, HTMerit, HPTConvexMerit
from data.datasets.experiment_datasets import *
from utils.io_helpers import save


# Evaluates each model on a chosen dataset. Collects number of nodes, accuracy values, kappa scores and
# feature importance values (if available) for each instance.
if __name__ == '__main__':

    models = [("ht", HoeffdingTreeClassifier()),
              ("ht_merit", HTMerit(seed=40)),
              ("ht_merit", HTMerit(seed=41)),
              ("ht_merit", HTMerit(seed=42)),
              ("hat_seed40", HoeffdingAdaptiveTreeClassifier(seed=40)),
              ("hat_seed41", HoeffdingAdaptiveTreeClassifier(seed=41)),
              ("hat_seed42", HoeffdingAdaptiveTreeClassifier(seed=42)),
              ("efdt", ExtremelyFastDecisionTreeClassifier()),
              ("hpt_seed40", HoeffdingPruningTree(max_depth=None, seed=40)),
              ("hpt_seed41", HoeffdingPruningTree(max_depth=None, seed=41)),
              ("hpt_seed42", HoeffdingPruningTree(max_depth=None, seed=42)),
              ("hpt_merit_seed40", HPTMerit(max_depth=None, seed=40)),
              ("hpt_merit_seed41", HPTMerit(max_depth=None, seed=41)),
              ("hpt_merit_seed42", HPTMerit(max_depth=None, seed=42)),
              ("hpt_convex_merit_seed40", HPTConvexMerit(alpha=0.5, max_depth=None, seed=40)),
              ("hpt_convex_merit_seed41", HPTConvexMerit(alpha=0.5, max_depth=None, seed=41)),
              ("hpt_convex_merit_seed42", HPTConvexMerit(alpha=0.5, max_depth=None, seed=42))
              ]

    datasets = [("airlines", lambda: Airlines()),
                ("electricity", lambda: Elec2()),
                ("kdd99", lambda: KDD99()),
                ("wisdm", lambda: WISDM()),
                ("covtype", lambda: CovType()),
                ("nomao", lambda: Nomao()),
                ("agr_a", lambda: DriftingAgrawal(width=50).take(10**6)),
                ("agr_g", lambda: DriftingAgrawal(width=50000).take(10**6)),
                ("rbf_f", lambda: DriftingRBF(change_speed=0.001).take(10**6)),
                ("rbf_m", lambda: DriftingRBF(change_speed=0.0001).take(10**6)),
                ("led_a", lambda: DriftingLED(width=50).take(10**6)),
                ("led_g", lambda: DriftingLED(width=50000).take(10**6))
                ]

    data_name, data_generator = datasets[6]  # choose the dataset, 0-11
    for (model_name, model) in models:
        data = data_generator()

        acc_values = []
        kappa_values = []
        n_nodes = []

        accuracy = Accuracy()
        kappa = CohenKappa()

        print(f"Starting evaluation for {model_name} on {data_name}...\n")
        for i, (x, y) in enumerate(data, start=1):

            y_pred = model.predict_one(x)
            # Updates metrics and stores the values.
            accuracy.update(y, y_pred)
            acc_values.append(accuracy.get())
            kappa.update(y, y_pred)
            kappa_values.append(kappa.get())

            model.learn_one(x, y)
            n_nodes.append(model.n_nodes)

            if i % 10000 == 0:
                print(
                    f"Iteration: {i}\n"
                    f"Model: {model_name} | Data: {data_name}\n"
                    f"Accuracy: {accuracy.get():.3f} | Kappa: {kappa.get():.3f} | Node count: {model.n_nodes}\n"
                    "------------------------------------------------------------"
                )

        # Stores collected stats.
        save(acc_values, f"results/acc_values/{model_name}_{data_name}.npy")
        save(kappa_values, f"results/kappa_values/{model_name}_{data_name}.npy")
        save(n_nodes, f"results/n_nodes/{model_name}_{data_name}.npy")

        # Stores feature importance values if the model collected any.
        if isinstance(model, HoeffdingPruningTree):
            save(model.collected_importance_values, f"results/fi_values/{model_name}_{data_name}.npy")

        avg_total_acc = np.mean(acc_values)
        avg_total_kappa = np.mean(kappa_values)
        avg_total_nodes = np.mean(n_nodes)

        print(
            f"Summary for {model_name} on {data_name}:\n"
            f"  - Total avg. node count: {avg_total_nodes}\n"
            f"  - Avg. AUC: {avg_total_kappa:.3f}\n"
            f"  - Avg. Accuracy: {avg_total_acc:.3f}\n"
            "------------------------------------------------------------"
        )

