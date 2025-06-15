import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as ss
from critdd import Diagram
import seaborn as sns
from river import metrics

from tree import ProeFI
from utils.io_helpers import load
from ipfi.scaler import MinMaxScaler
from data.datasets.experiment_datasets import DriftingAgrawal


model_names = ["ht",
               "hat",
               "efdt",
               "proefi",
               ]

data_names = ["airlines", "electricity", "covtype", "nomao", "kdd99", "wisdm",
              "agr_a", "agr_g", "rbf_f", "rbf_m", "led_a", "led_g"
              ]
seeds = [40, 41, 42, 43, 44]

plot_dir = "./results/plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


def get_df_summary(metric, data_dir="./results"):
    if metric != "tradeoff":
        table = pd.DataFrame(index=data_names, columns=model_names)
        for model in model_names:
            for data_name in data_names:
                value = 0
                seeds_seen = 0
                for seed in seeds:
                    file_path = f"{data_dir}/summary/{model}_seed{seed}_{data_name}.csv"
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        value += df.iloc[0][metric]
                        seeds_seen += 1
                    else:
                        try:
                            df = pd.read_csv(f"{data_dir}/summary/{model}_{data_name}.csv")
                            value += df.iloc[0][metric]
                            seeds_seen += 1
                            break
                        except:
                            pass

                if seeds_seen > 0:
                    value /= seeds_seen
                    table.at[data_name, model] = value
        table.index = table.index.str.replace("_", " ")
    else:
        table1 = get_df_summary(metric="Auroc", data_dir=data_dir)
        table2 = get_df_summary(metric="Avg Node Count", data_dir=data_dir)
        table2 = table2.apply(pd.to_numeric)
        table2 = np.log2(table2) + 1
        table = table1 / table2

    return table


def make_latex_table(metric, highlight_max=True, precision=3, data_dir="./results"):
    table = get_df_summary(metric=metric, data_dir=data_dir)
    # Add mean ranks
    ranks = table.rank(axis=1, method='average', ascending=False if highlight_max else True)
    mean_ranks = ranks.mean()
    table.loc['Mean Rank'] = mean_ranks
    if highlight_max:
        print(table.style.highlight_max(axis=1, props="textbf:--rwrap;").format(precision=precision).to_latex())
    else:
        print(table.style.highlight_min(axis=1, props="textbf:--rwrap;").format(precision=precision).to_latex())


# https://mirkobunse.github.io/critdd/
# Install with: pip install 'critdd @ git+https://github.com/mirkobunse/critdd'
def get_cridd(metric, data_dir="./results"):
    df = get_df_summary(metric=metric, data_dir=data_dir)
    df = df.rename_axis('dataset_name', axis=0)
    df = df.rename_axis('classifier_name', axis=1)
    df = df.astype(np.float64)

    # create a CD diagram from the Pandas DataFrame
    diagram = Diagram(
        df.to_numpy(),
        treatment_names=df.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    diagram.average_ranks  # the average rank of each treatment
    diagram.get_groups(alpha=.05, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        f"./results/plots/critdd-{metric}.tex",
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
    )


def plot_tau_values(metric, data_dir="./results"):
    taus = ['$\\tau_t$']
    taus.extend([0.01 * i for i in range(1, 11)])
    taus.extend([0.2, 0.3, 0.4, 0.5])
    # Read files
    table = pd.DataFrame(index=data_names, columns=taus)
    for tau in taus:
        if tau == "$\\tau_t$":
            model_name = "proefi"
        else:
            model_name = f"proefi_tau_{tau}"
        for data_name in data_names:
            value = 0
            seeds_seen = 0
            for seed in seeds:
                file_path = f"{data_dir}/summary/{model_name}_seed{seed}_{data_name}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    value += df.iloc[0][metric]
                    seeds_seen += 1

            if seeds_seen > 0:
                value /= seeds_seen
                table.at[data_name, tau] = value
    table.index = table.index.str.replace("_", " ")

    table = table.apply(pd.to_numeric, errors='coerce')
    plt.rcParams.update({'font.size': 16})
    sns.boxplot(table)
    if metric == "Avg Node Count":
        plt.yscale("log")
        plt.ylabel("#nodes")
    else:
        plt.ylabel("AUROC")
    plt.xlabel("$\\tau$ values")
    plt.tick_params("x", rotation=45)
    plt.tight_layout()
    metric = metric.replace(" ", "_")
    plt.savefig(f"./results/plots/boxplot_tau_threshold_{metric}.pdf", format="pdf", bbox_inches="tight")


def plot_model_behavior():
    """ This first trains ProeFI on Agrawal with an abrupt concept drift.
    Then it plots the feature importance, AUROC and node count over time.
    """
    model = ProeFI(seed=40)
    data = DriftingAgrawal(width=50, seed=40)
    metric = metrics.ROCAUC()

    auc_list = []
    node_list = []
    for i, (x, y) in enumerate(data.take(1000000), start=1):
        y_pred = model.predict_proba_one(x)
        metric.update(y, y_pred)
        auc_list.append(metric.get())
        node_list.append(model.n_nodes)

        model.learn_one(x, y)

    fi_values = model.collected_importance_values

    plot_feature_importance(fi_values)
    plot_auroc_and_nodes(auc_list, node_list)


def plot_feature_importance(fi_values):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(fi_values)))
    scaler = MinMaxScaler()
    for row in fi_values:
        scaler.learn_one(row)
    fi_values = [scaler.transform_one(row) for row in fi_values]

    highlight_set = set()
    all_keys = list(fi_values[0].keys())
    totals = {key: sum(abs(row[key]) for row in fi_values) for key in all_keys}
    top_keys = sorted(totals, key=totals.get, reverse=True)[:4]
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

    for xv in [250000, 500000, 750000]:
        ax.axvline(x=xv, color='grey', linestyle='--', linewidth=1)

    ax.legend(loc='upper right')
    plt.xlabel("Instances")
    plt.ylabel("Feature importance")
    plt.tight_layout()
    plt.savefig(f"./results/plots/plot_fi_over_time.pdf", format="pdf", bbox_inches="tight")


def plot_auroc_and_nodes(auc_list, node_list):
    x = np.arange(1, len(auc_list) + 1)
    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_auc = 'tab:blue'
    ax1.plot(x[100:], auc_list[100:], color=color_auc, label='AUROC')
    ax1.set_xlabel('Instances')
    ax1.set_ylabel('AUROC')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    color_nodes = 'tab:red'
    ax2.plot(x[100:], node_list[100:], color=color_nodes, label='#nodes')
    ax2.set_ylabel('#nodes')
    ax2.set_yscale("log")
    ax2.tick_params(axis='y')

    for xv in [250000, 500000, 750000]:
        ax1.axvline(x=xv, color='grey', linestyle='--', linewidth=1)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    plt.tight_layout()
    plt.savefig(f"./results/plots/auroc_nodecount_plot.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    data_dir = "./results"
    # plot_performances_all_data()
    # plot_fi_importance_all_data(data_dir=data_dir)
    make_latex_table(metric='Auroc', precision=2, data_dir=data_dir)
    make_latex_table(metric='Avg Node Count', highlight_max=False, precision=2, data_dir=data_dir)
    make_latex_table(metric='tradeoff', highlight_max=True, precision=3, data_dir=data_dir)
    get_cridd(metric='tradeoff', data_dir=data_dir)
    plot_tau_values(metric='Auroc', data_dir=data_dir)
    plot_tau_values(metric='Avg Node Count', data_dir=data_dir)
    plot_model_behavior()
