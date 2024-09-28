from utils.io_helpers import load
from data.utils import load_thesis_datasets
from utils.plotting import plot_differences
from utils.evaluate_multiple import evaluate_multiple
from utils.compute_averages import compute_total_avg

from tree.EFDT import EFDT
from tree.hoeffding_pruning_tree import HoeffdingPruningTree

from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier

model_names = ["base_ht", "efdt", "hatc",
               "hpt_cpl_05",
               "hpt_cpl_02",
               "experimental HPT"
               ]

data_name = "SEA"
rnodes = {}
for name in model_names[:5]:
    rnodes[name] = load(f"C:/Users/kevin/PycharmProjects/bachelorarbeit/results/collected_stats/{name}/n_nodes/sea.npy")
rnodes["experimental HPT"] = load("results/n_nodes/ehpt_nodes_sea.npy")
print(compute_total_avg(rnodes["experimental HPT"]))
rauc = {}
for name in model_names[:5]:
    rauc[name] = load(f"C:/Users/kevin/PycharmProjects/bachelorarbeit/results/collected_stats/{name}/metric_values/sea.npy")[1000:]
rauc["experimental HPT"] = load("results/metric_values/ehpt_auc_sea.npy")[1000:]
print(compute_total_avg(rauc["experimental HPT"]))

plot_differences(rnodes, f"Growth of n_nodes in models over time ({data_name})", "Number of nodes",
                 model_names)

plot_differences(rauc, f"Preq. AUC of models over time ({data_name})", "preq. AUC",
                 model_names)

data_name = "AGRm"
rnodes = {}
for name in model_names[:5]:
    rnodes[name] = load(f"C:/Users/kevin/PycharmProjects/bachelorarbeit/results/collected_stats/{name}/n_nodes/agrawal_mixed.npy")
rnodes["experimental HPT"] = load("results/n_nodes/ehpt_nodes_agr_m.npy")
print(compute_total_avg(rnodes["experimental HPT"]))
rauc = {}
for name in model_names[:5]:
    rauc[name] = load(f"C:/Users/kevin/PycharmProjects/bachelorarbeit/results/collected_stats/{name}/metric_values/agrawal_mixed.npy")[1000:]
rauc["experimental HPT"] = load("results/metric_values/ehpt_auc_agr_m.npy")[1000:]
print(compute_total_avg(rauc["experimental HPT"]))
plot_differences(rnodes, f"Growth of n_nodes in models over time ({data_name})", "Number of nodes",
                 model_names)

plot_differences(rauc, f"Preq. AUC of models over time ({data_name})", "preq. AUC",
                 model_names)