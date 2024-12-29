import numpy as np

from utils.io_helpers import load
from data.utils import load_thesis_datasets
from utils.plotting import plot_differences
from utils.evaluate_multiple import evaluate_multiple
from utils.compute_averages import compute_total_avg, compute_stats_avgs


# TODO: Load all results and average if mulitple seeds were used.
# load (into one list)
# compute_stats_avgs

# TODO: Plot only the two HT variants for a comparison.
# plot_differences
...

# TODO: Plot only the two AdwinHPT variants for a comparison.
# plot_differences
...

# TODO: PLOT HT, HAT, EFDT, AdwinHPT and potentially the merit versions.
# plot_differences
...


model_names = [#"HT",
               #"efdt",
               #"HAT",
               #"HAT (depth 4)",
               #"HAT (depth 5)",
               #"HAT (depth 6)",
               #"hpt_cpl_05",
               #"hpt_cpl_02",,
               "adwin HPT",
               "adwin HPT (convex half)",
               "adwin HPT (convex node_count)",
               "adwin HPT (merit * log2 * norm_FI)",
               "adwin HPT (merit * norm_FI)"
               ]

data_names = ["agrm", "hyp", "elec", "creditcard"]
for data_name in ["agrm100k"]:
    rnodes = {}

    rnodes["adwin HPT"] = load(f"results/n_nodes/adwin_hpt_nodes_{data_name}.npy")
    rnodes["adwin HPT (convex half)"] = load(f"results/n_nodes/adwin_hpt_convexhalf_nodes_{data_name}.npy")
    rnodes["adwin HPT (convex node_count)"] = load(f"results/n_nodes/adwin_hpt_convexnnode_nodes_{data_name}.npy")
    rnodes["adwin HPT (merit * log2 * norm_FI)"] = load(f"results/n_nodes/adwin_hpt_timeslog_nodes_{data_name}.npy")
    rnodes["adwin HPT (merit * norm_FI)"] = load(f"results/n_nodes/adwin_hpt_times_nodes_{data_name}.npy")

    print(f"Avg node count; \n"
          f" adwin HPT {compute_total_avg(rnodes['adwin HPT'])}, \n"
          f" adwin HPT (convex half) {compute_total_avg(rnodes['adwin HPT (convex half)'])}, \n"
          f" adwin HPT (convex node_count) {compute_total_avg(rnodes['adwin HPT (convex node_count)'])}, \n"
          f" adwin HPT (merit * log2 * norm_FI) {compute_total_avg(rnodes['adwin HPT (merit * log2 * norm_FI)'])}, \n"
          f" adwin HPT (merit * norm_FI) {compute_total_avg(rnodes['adwin HPT (merit * norm_FI)'])}")

    rauc = {}
    rauc["adwin HPT"] = load(f"results/metric_values/adwin_hpt_auc_{data_name}.npy")
    rauc["adwin HPT (convex half)"] = load(f"results/metric_values/adwin_hpt_convexhalf_auc_{data_name}.npy")
    rauc["adwin HPT (convex node_count)"] = load(f"results/metric_values/adwin_hpt_convexnnode_auc_{data_name}.npy")
    rauc["adwin HPT (merit * log2 * norm_FI)"] = load(f"results/metric_values/adwin_hpt_timeslog_auc_{data_name}.npy")
    rauc["adwin HPT (merit * norm_FI)"] = load(f"results/metric_values/adwin_hpt_times_auc_{data_name}.npy")

    print(f"Avg auc; \n"
          f" adwin HPT {compute_total_avg(rauc['adwin HPT'])}, \n"
          f" adwin HPT (convex half) {compute_total_avg(rauc['adwin HPT (convex half)'])}, \n"
          f" adwin HPT (convex node_count) {compute_total_avg(rauc['adwin HPT (convex node_count)'])}, \n"
          f" adwin HPT (merit * log2 * norm_FI) {compute_total_avg(rauc['adwin HPT (merit * log2 * norm_FI)'])}, \n"
          f" adwin HPT (merit * norm_FI) {compute_total_avg(rauc['adwin HPT (merit * norm_FI)'])}")

    plot_differences(rnodes, f"Growth of n_nodes in models over time ({data_name})", "Number of nodes",
                     model_names)

    plot_differences(rauc, f"Preq. AUC of models over time ({data_name})", "preq. AUC",
                     model_names)