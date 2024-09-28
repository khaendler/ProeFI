import sys

import numpy as np
from river.metrics import RollingROCAUC
import matplotlib.pyplot as plt

from utils.io_helpers import save
from utils.compute_averages import compute_fi_avgs
from utils.plotting import plot_feature_importance
from data.utils import load_thesis_datasets
from tree.hoeffding_pruning_tree import HoeffdingPruningTree
from tree.experimental_hpt import ExperimentalHPT


perf = [[], [], [], [], []]
n_nodes = [[], [], [], [], []]
fis = []
for i, seed in enumerate([40, 41, 42, 43, 44]):
    model = ExperimentalHPT(seed=42)
    metric = RollingROCAUC()
    data = load_thesis_datasets(pertubation=0.2, seed=seed)
    data = data[1]
    for j, (x, y) in enumerate(data.take(1000000), start=1):
        y_pred = model.predict_proba_one(x)
        metric.update(y, y_pred)
        perf[i].append(metric.get())
        model.learn_one(x, y)

        n_nodes[i].append(model.n_nodes)

        if j % 10000 == 0:
            print(f"{j}: AUC: {metric.get():.3f}, Node count: {model.n_nodes}")

    fis.append(model.pfi_plotter.y_data)

sys.exit()
np_nodes = np.array(n_nodes)
avg_n_nodes = np.mean(np_nodes, axis=0)
avg_total_nodes = np.mean(np_nodes)
save(avg_n_nodes, "results/n_nodes/ehpt_nodes_agr_m.npy")

np_perf = np.array(perf)
avg_perf = np.mean(np_perf, axis=0)
avg_total_perf = np.mean(np_perf)
save(avg_perf, "results/metric_values/ehpt_auc_agr_m.npy")

avg_fi = compute_fi_avgs(fis)
save(avg_fi, "results/fi_values/ehpt_fi_agr_m.npy")

print(f"Total avg. node count: {avg_total_nodes}, Avg. perf.: {avg_total_perf}")

x_values = list(range(0, len(avg_n_nodes)))
plt.figure(figsize=(10, 6))
plt.plot(x_values, avg_n_nodes)
plt.title("Tree growth of HPT with windowed avg. threshold (AGRm)")
plt.xlabel("Number of instances")
plt.ylabel("Node count")
plt.legend()
plt.grid(True)
plt.savefig("ehpt_node_count_agr_m")
print(f"Plot saved as ehpt_node_count_agr_m")
plt.show()

x_values = list(range(1000, len(avg_perf)))
plt.figure(figsize=(10, 6))
plt.plot(x_values, avg_perf[1000:])
plt.title("Preq. AUC of HPT with windowed avg. threshold (AGRm)")
plt.xlabel("Number of instances")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.savefig("ehpt_auc_agr_m")
print(f"Plot saved as ehpt_auc_agr_m")
plt.show()

plot_feature_importance(avg_fi, ["salary", "commission", "age", "elevel"], "FI values of HPT with windowed avg. threshold (AGRm)",
                        save_name="ehpt_fi_agr_m")
