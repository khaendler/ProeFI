import numpy as np
from river.metrics import Accuracy, CohenKappa
from river.tree import (HoeffdingTreeClassifier,
                        HoeffdingAdaptiveTreeClassifier,
                        ExtremelyFastDecisionTreeClassifier)
from river.datasets import Elec2
from tree import HoeffdingPruningTree, HPTMerit, HTMerit, HPTConvexMerit, HPTFixedThreshold
from data.datasets.experiment_datasets import *
from utils.io_helpers import save
from pathlib import Path
from sklearn.metrics import roc_auc_score
import random

Path("./results").mkdir(parents=True, exist_ok=True)
Path("./results/acc_values").mkdir(parents=True, exist_ok=True)
Path("./results/kappa_values").mkdir(parents=True, exist_ok=True)
Path("./results/n_nodes").mkdir(parents=True, exist_ok=True)
Path("./results/fi_values").mkdir(parents=True, exist_ok=True)
Path("./results/summary").mkdir(parents=True, exist_ok=True)

datasets_n_classes = {"airlines": 2, "electricity": 2, "kdd99": 23, "wisdm": 6, "covtype": 7, "nomao": 2,
                      "agr_a": 2, "agr_g": 2, "rbf_f": 5,  "rbf_m": 5, "led_a": 10, "led_g": 10 }


# Evaluates each model on a chosen dataset. Collects number of nodes, accuracy values, kappa scores and
# feature importance values (if available) for each instance.
def run_evaluation(data_name, seed):
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    taus = [0.01 * i for i in range(1, 11)]
    taus.extend([0.2, 0.3, 0.4, 0.5])
    models = [
              #("ht", HoeffdingTreeClassifier()),
              # (f"ht_merit_seed{seed}", HTMerit(seed=seed)),
              # (f"hat_seed{seed}", HoeffdingAdaptiveTreeClassifier(seed=seed)),
              # ("efdt", ExtremelyFastDecisionTreeClassifier()),
              # (f"hpt_seed{seed}", HoeffdingPruningTree(max_depth=None, seed=seed)),
              # (f"hpt_merit_seed{seed}", HPTMerit(max_depth=None, seed=seed)),
              ]
    # for alpha in alphas:
    #     models.append((f"hpt_convex_merit_{alpha}_seed{seed}", HPTConvexMerit(alpha=alpha, max_depth=None, seed=seed)))
    for tau in taus:
        models.append((f"hpt_tau_{tau}_seed{seed}", HPTFixedThreshold(importance_threshold=tau, seed=seed)))

    datasets = {"airlines": lambda: Airlines(),
                "electricity": lambda: Elec2(),
                # "kdd99": lambda: KDD99(),
                "wisdm": lambda: WISDM(),
                "covtype": lambda: CovType(),
                "nomao": lambda: Nomao(),
                "agr_a": lambda: DriftingAgrawal(width=50).take(10 ** 6),
                "agr_g": lambda: DriftingAgrawal(width=50000).take(10 ** 6),
                "rbf_f": lambda: DriftingRBF(change_speed=0.001).take(10 ** 6),
                "rbf_m": lambda: DriftingRBF(change_speed=0.0001).take(10 ** 6),
                "led_a": lambda: DriftingLED(width=50).take(10 ** 6),
                "led_g": lambda: DriftingLED(width=50000).take(10 ** 6)
                }

    data_generator = datasets[data_name]
    for (model_name, model) in models:
        data = data_generator()
        evaluator = EvaluatorTree(n_classes=datasets_n_classes[data_name])

        print(f"Starting evaluation for {model_name} on {data_name}...\n")
        for i, (x, y) in enumerate(data, start=1):
            y_pred = model.predict_proba_one(x)
            model.learn_one(x, y)
            evaluator.update(y_pred=y_pred, y_target=y, n_nodes=model.n_nodes)
            # if i == 10000: break

        # save(evaluator.acc_values, f"results/acc_values/{model_name}_{data_name}.npy")
        # save(evaluator.kappa_values, f"results/kappa_values/{model_name}_{data_name}.npy")
        # save(evaluator.n_nodes, f"results/n_nodes/{model_name}_{data_name}.npy")

        # Stores feature importance values if the model collected any.
        if isinstance(model, HoeffdingPruningTree):
            save(model.collected_importance_values, f"results/fi_values/{model_name}_{data_name}.npy")

        metrics = evaluator.get_final()
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(f"results/summary/{model_name}_{data_name}.csv", mode="w", index=False, header=True)
        print(metrics)


class EvaluatorTree:
    def __init__(self, n_classes):
        self.acc_values = []
        self.kappa_values = []
        self.n_nodes = []
        self.y_pred_list = []
        self.y_target_list = []
        self.accuracy = Accuracy()
        self.kappa = CohenKappa()
        self.positive_class = 1
        self.n_classes = n_classes

    def normalize(self, y_pred):
        if sum(y_pred) == 0:
            y_pred[random.randint(0, len(y_pred) - 1)] = 1.
        return [float(y_i) / sum(y_pred) for y_i in y_pred]

    def get_auroc(self):
        if self.n_classes > 2:
            auroc = roc_auc_score(self.y_target_list, self.y_pred_list, multi_class='ovr')
        else:
            auroc = roc_auc_score(self.y_target_list, np.array(self.y_pred_list)[:, self.positive_class])
        return auroc

    def update(self, y_pred, y_target, n_nodes):
        if y_pred:
            y_pred_label = max(y_pred, key=y_pred.get)
        else:
            y_pred_label = None
        y_pred = list(y_pred.values())
        if len(y_pred) != self.n_classes:
            while len(y_pred) != self.n_classes:
                y_pred.append(0)
        # Normalize y_pred if not yet done (e.g., avoid precision errors)
        if sum(y_pred) != 1:
            y_pred = self.normalize(y_pred)
        # Predictive performance
        self.accuracy.update(y_target, y_pred_label)
        self.acc_values.append(self.accuracy.get())
        self.kappa.update(y_target, y_pred_label)
        self.kappa_values.append(self.kappa.get())
        self.y_pred_list.append(y_pred)
        self.y_target_list.append(y_target)
        # Nodes after training
        self.n_nodes.append(n_nodes)

    def get_final(self):
        return {
            "Auroc": self.get_auroc(),
            "Avg Node Count": np.mean(self.n_nodes),
            "Avg Accuracy": np.mean(self.acc_values),
            "Avg Kappa": np.mean(self.kappa_values)
        }
