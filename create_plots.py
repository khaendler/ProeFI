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
               "adwin_hpt",
               "adwin_hpt_merit",
               ]

data_names = ["airlines", "electricity", "wisdm", "covtype", "nomao",
              # "kdd99",
              "agr_a", "agr_g", "rbf_f", "rbf_m", "led_a", "led_g"]
seeds = [40, 41, 42]

plot_dir = "./results/plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


def get_metrics_from_all_data():
    metrics = {}
    for data_name in data_names:
        nnodes = {}
        kappas = {}
        accuracies = {}
        metrics[data_name] = {}
        for model_name in model_names:
            average_nnodes, average_kappas, average_accuracies = None, None, None
            if model_name not in ['ht', 'efdt', 'ht_merit']:  # todo remove ht_merit from here
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


def create_latex_table(metric_name='accuracies', mark_min=False):
    print(f"---- {metric_name} ----")
    metrics = get_metrics_from_all_data()
    column_names = None
    table_str = ""
    for data_name in metrics.keys():
        data_n = data_name.replace('_', '\\_')
        table_str += f"{data_n} & "
        if column_names is None:
            column_names = [col.replace('_', ' ') for col in metrics[data_name].keys()]
            column_names.insert(0, 'Data stream')
        values = []
        for model_name in metrics[data_name].keys():
            values.append(np.mean(metrics[data_name][model_name][metric_name][model_name]))
        if mark_min:
            marked_val = round(min(values), 2)
        else:
            marked_val = round(max(values), 2)
        for acc in values:
            if not mark_min and round(acc, 2) >= marked_val or mark_min and round(acc, 2) <= marked_val:
                table_str += f"\\textbf{{{acc:.2f}}} &"
            else:
                table_str += f"{acc:.2f} &"
        table_str = table_str[:-1]  # Remove last "&"
        table_str += "\\\\\n"
    print(" & ".join(column_names), "\\\\\n\\hline")
    print(table_str)


# https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html
def critical_difference_diagram():
    metrics = get_metrics_from_all_data()
    data = []
    data_dict = {}
    for data_name in metrics.keys():
        values = []
        for model_name in metrics[data_name].keys():
            acc = np.mean(metrics[data_name][model_name]['accuracies'][model_name])
            if model_name in data_dict:
                data_dict[model_name].append(acc)
            else:
                data_dict[model_name] = [acc]
            values.append(acc)
        data.append(values)
    # data = np.array(data)
    print(data_dict)

    data_df = (
        pd.DataFrame(data_dict)
        .rename_axis('cv_fold')
        .melt(
            var_name='estimator',
            value_name='score',
            ignore_index=False,
        )
        .reset_index()
    )
    print(data_df)
    avg_rank = data_df.groupby('cv_fold').score.rank(pct=True).groupby(data_df.estimator).mean()
    print(avg_rank)
    ss.friedmanchisquare(*data_dict.values())

    data_df['block_id'] = data_df.groupby(['cv_fold', 'estimator']).ngroup()
    test_results = sp.posthoc_conover_friedman(
        data_df,
        melted=True,
        block_col='cv_fold',
        group_col='estimator',
        y_col='score',
        block_id_col='block_id'
    )
    sp.sign_plot(test_results)

    plt.figure(figsize=(11, 4), dpi=100)
    plt.title('Critical difference diagram of average accuracy ranks')
    sp.critical_difference_diagram(avg_rank, test_results)

    plt.show()


def critical_difference_diagram_nnodes():
    metrics = get_metrics_from_all_data()
    data = []
    data_dict = {}
    for data_name in metrics.keys():
        values = []
        for model_name in metrics[data_name].keys():
            acc = np.mean(metrics[data_name][model_name]['nnodes'][model_name])
            if model_name in data_dict:
                data_dict[model_name].append(acc)
            else:
                data_dict[model_name] = [acc]
            values.append(acc)
        data.append(values)

    data_df = (
        pd.DataFrame(data_dict)
        .rename_axis('cv_fold')
        .melt(
            var_name='estimator',
            value_name='score',
            ignore_index=False,
        )
        .reset_index()
    )
    print(data_df)
    avg_rank = data_df.groupby('cv_fold').score.rank(pct=True, ascending=False).groupby(data_df.estimator).mean()
    print(avg_rank)
    ss.friedmanchisquare(*data_dict.values())

    data_df['block_id'] = data_df.groupby(['cv_fold', 'estimator']).ngroup()
    test_results = sp.posthoc_conover_friedman(
        data_df,
        melted=True,
        block_col='cv_fold',
        group_col='estimator',
        y_col='score',
        block_id_col='block_id'
    )
    sp.sign_plot(test_results)

    plt.figure(figsize=(13, 4), dpi=100)
    plt.title('Critical difference diagram of average number of nodes ranks')
    sp.critical_difference_diagram(avg_rank, test_results)

    plt.show()


if __name__ == '__main__':
    # plot_performances_all_data()
    # plot_fi_importance_all_data()
    # create_latex_table(metric_name='accuracies')
    # create_latex_table(metric_name='kappas')
    # create_latex_table(metric_name='nnodes', mark_min=True)
    # critical_difference_diagram()
    critical_difference_diagram_nnodes()
