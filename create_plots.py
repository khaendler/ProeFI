import numpy as np

from utils.io_helpers import load
from data.utils import load_thesis_datasets
from utils.plotting import plot_differences, plot_feature_importance
from utils.evaluate_multiple import evaluate_multiple
from utils.compute_averages import compute_total_avg, compute_stats_avgs
from pathlib import Path
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import os
from critdd import Diagram

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

model_names = ["ht",
               "ht_merit",
               "hat",
               "efdt",
               "hpt",
               "hpt_merit",
               "hpt_convex_merit"
               ]

data_names = ["airlines", "electricity",  "covtype", "nomao",
              "kdd99",
              "wisdm",
              "agr_a", "agr_g", "rbf_f", "rbf_m", "led_a", "led_g"]
seeds = [40, 41, 42]

plot_dir = "./results/plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


def get_df_summary(metric, data_dir="./results"):
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
    return table


def make_latex_table(metric, highlight_max=True, precision=3):
    table = get_df_summary(metric=metric)
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
def get_cridd(metric):
    df = get_df_summary(metric=metric)
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
        # axis_options={"title": f"{base_learner}-{args}"},
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
        for model_name in ["adwin_hpt", "adwin_hpt_merit"]:
            for i, seed in enumerate(seeds[:1]):
                fi_values = load(f"results/fi_values/{model_name}_seed{seed}_{data_name}.npy")
                names_to_highlight = None
                plot_feature_importance(fi_values=fi_values, names_to_highlight=names_to_highlight,
                                        title=f"Feature Importance on {data_name} using {model_name}",
                                        save_name=f"{plot_dir}/fi_value_{data_name}_{model_name}")


if __name__ == '__main__':
    # plot_performances_all_data()
    # plot_fi_importance_all_data()
    # make_latex_table(metric='Auroc')
    make_latex_table(metric='Avg Node Count', highlight_max=False, precision=2)
    get_cridd(metric='Auroc')
