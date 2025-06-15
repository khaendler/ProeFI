import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as ss
from critdd import Diagram
import seaborn as sns
from utils.io_helpers import load
from utils.plotting import plot_differences, plot_feature_importance


model_names = ["ht",
               "hat",
               "efdt",
               "hpt",
               ]

data_names = ["airlines", "electricity", "covtype", "nomao", "kdd99", "wisdm",
              "agr_a", "agr_g", "rbf_f", "rbf_m", "led_a", "led_g"
              ]
seeds = [40, 41, 42, 43, 44]

plot_dir = "./results/plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


def ttest(
        metric: str,
        data_name: str,
        model_names: tuple[str, str] = ("hat", "hpt"),
        data_dir: str = "./results"):
    """
    Performs a ttest between two given models for a given dataset.
    :param metric: The metric to evaluate.
    :param data_name: Name of the dataset
    :param model_names: Names of models to compare.
    :param data_dir: Results directory.
    :return: p_value
    """

    model_dfs = [[] for _ in range(2)]
    for i, model in enumerate(model_names):
        summary_dir = os.path.join(data_dir, "summary")
        pattern = re.compile(rf"{model}_seed(\d+)_{data_name}.csv")

        for filename in os.listdir(summary_dir):
            if pattern.match(filename):
                file_path = os.path.join(summary_dir, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    model_dfs[i].append(df)

    model_results = [pd.concat(frames)[metric].values for frames in model_dfs]
    _, p_value = ss.ttest_rel(model_results[0], model_results[1])

    return p_value


def ttest_for_all_data(
        metric: str,
        model_names: tuple[str, str] = ("hat", "hpt"),
        data_dir: str = "./results"):
    """
    Performs a ttest between two given models for all datasets.
    :param metric: The metric to evaluate.
    :param model_names: Names of models to compare.
    :param data_dir: Results directory.
    :return: {data_name, p_value}
    """

    summary_dir = os.path.join(data_dir, "summary")
    data_names = set()
    pattern = re.compile(rf"(?:{'|'.join(model_names)})_seed\d+_(\w+)\.csv")

    for filename in os.listdir(summary_dir):
        match = pattern.match(filename)
        if match:
            data_names.add(match.group(1))

    results = {}
    for data_name in data_names:
        significant = ttest(metric, data_name, model_names, data_dir)
        results[data_name] = significant

    return results


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


def get_metrics_from_all_data():
    metrics = {}
    for data_name in data_names:
        nnodes = {}
        kappas = {}
        accuracies = {}
        metrics[data_name] = {}
        for model_name in model_names:
            average_nnodes, average_kappas, average_accuracies = None, None, None
            if model_name not in ['ht', 'efdt']:  # HT and EFDT are deterministic
                for i, seed in enumerate(seeds):
                    current_nnodes = np.array(load(f"results/n_nodes/{model_name}_seed{seed}_{data_name}.npy"))
                    current_kappas = np.array(load(f"results/kappa_values/{model_name}_seed{seed}_{data_name}.npy"))
                    current_accuracies = np.array(load(f"results/acc_values/{model_name}_seed{seed}_{data_name}.npy"))
                    if i == 0:
                        average_nnodes = current_nnodes
                        average_kappas = current_kappas
                        average_accuracies = current_accuracies
                    else:
                        average_nnodes += current_nnodes
                        average_kappas += current_kappas
                        average_accuracies += current_accuracies
                nnodes[model_name] = average_nnodes / len(seeds)
                kappas[model_name] = average_kappas / len(seeds)
                accuracies[model_name] = average_accuracies / len(seeds)
            else:
                nnodes[model_name] = load(f"results/n_nodes/{model_name}_{data_name}.npy")
                kappas[model_name] = load(f"results/kappa_values/{model_name}_{data_name}.npy")
                accuracies[model_name] = load(f"results/acc_values/{model_name}_{data_name}.npy")

            metrics[data_name][model_name] = {'nnodes': nnodes, 'kappas': kappas, 'accuracies': accuracies}
    return metrics


def plot_performances_all_data():
    metrics = get_metrics_from_all_data()
    for data_name in data_names:
        for model_name in model_names:
            plot_differences(metrics[data_name][model_name]['nnodes'],
                             f"Growth of n_nodes in models over time ({data_name})", "Number of nodes",
                             model_names, filename=f"{plot_dir}/n_nodes_{data_name}")
            plot_differences(metrics[data_name][model_name]['kappas'],
                             f"Kappa values over time ({data_name})", "Kappa", model_names,
                             filename=f"{plot_dir}/kappa_{data_name}")
            plot_differences(metrics[data_name][model_name]['accuracies'],
                             f"Accuracy over time ({data_name})", "Accuracy", model_names,
                             filename=f"{plot_dir}/accuracy_{data_name}")


def plot_fi_importance_all_data():
    for data_name in data_names:
        # Plot feature importance
        for model_name in ["hpt", "hpt_tau_0.1", "hpt_tau_0.2", "hpt_tau_0.3", "hpt_tau_0.4", "hpt_tau_0.5",
                           "hpt_tau_0.01", "hpt_tau_0.02", "hpt_tau_0.03", "hpt_tau_0.04", "hpt_tau_0.05",
                           "hpt_tau_0.06", "hpt_tau_0.07", "hpt_tau_0.08", "hpt_tau_0.09"]:
            for i, seed in enumerate(seeds[:1]):
                fi_values = load(f"results/fi_values/{model_name}_seed{seed}_{data_name}.npy")
                names_to_highlight = []
                plot_feature_importance(fi_values=fi_values, names_to_highlight=names_to_highlight, top_k=4,
                                        title=f"Feature Importance on {data_name} using {model_name}",
                                        save_name=f"{plot_dir}/fi_value_{data_name}_{model_name}.pdf")


def plot_tau_values(metric, data_dir="./results"):
    taus = ['$\\tau_t$']
    taus.extend([0.01 * i for i in range(1, 11)])
    taus.extend([0.2, 0.3, 0.4, 0.5])
    # Read files
    table = pd.DataFrame(index=data_names, columns=taus)
    for tau in taus:
        if tau == "$\\tau_t$":
            model_name = "hpt"
        else:
            model_name = f"hpt_tau_{tau}"
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
