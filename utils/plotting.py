import matplotlib.pyplot as plt
from ixai.visualization import FeatureImportancePlotter

import numpy as np

def plot_feature_importance(fi_values, names_to_highlight, title=None,
                            model_performances=None, metric_name="Perf.", save_name=None):

    feature_names = list(fi_values['importance_values'][1].keys())
    plotter = FeatureImportancePlotter(feature_names=feature_names)
    plotter.y_data = fi_values
    performance_kw = {
        "y_min": 0, "y_max": 1, "y_label": metric_name
    }

    fi_kw = {
        "names_to_highlight": names_to_highlight,
        "legend_style": {
            "fontsize": "small", 'title': 'features', "ncol": 1,
            "loc": 'upper left', "bbox_to_anchor": (0, 1)},
        "title": title
    }
    model_performances = None if model_performances is None else {"Perf.": model_performances}
    plotter.plot(
        save_name=save_name,
        model_performances=model_performances,
        performance_kw=performance_kw,
        **fi_kw
    )


def plot_differences(stats, title, ylabel, model_names, filename=None):
    stats = list(stats.values())
    x_values = list(range(0, len(stats[0])))
    plt.figure(figsize=(10, 6))
    for i, y_values in enumerate(stats):
        plt.plot(x_values, y_values, label=model_names[i])

    plt.title(title)
    plt.xlabel("Number of instances")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if filename is not None:
        plot_filename = filename
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    plt.show()


def plot_differences_comparison(stats1, stats2, title1, title2, ylabel1, ylabel2, model_names, filename=None):
    stats1 = list(stats1.values())
    stats2 = list(stats2.values())
    x_values = list(range(0, 1000000))

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    # Increase font size globally
    label_fontsize = 14
    legend_fontsize = 14
    tick_labelsize = 12

    # Plot the first set of stats
    for i, y_values in enumerate(stats1):
        axs[0].plot(x_values, y_values, label=model_names[i])
    axs[0].set_xlabel("Instances", fontsize=label_fontsize)
    axs[0].set_ylabel(ylabel1, fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Plot the second set of stats
    x_values = list(range(10001, 1000001))
    for i, y_values in enumerate(stats2):
        axs[1].plot(x_values, y_values, label=model_names[i])
    axs[1].set_xlabel("Instances", fontsize=label_fontsize)
    axs[1].set_ylabel(ylabel2, fontsize=label_fontsize)
    axs[1].legend(fontsize=legend_fontsize, ncols=3)
    axs[1].grid(True)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_labelsize)

    axs[0].annotate(f"(a) {title1}", xy=(0.5, -0.25), xycoords='axes fraction', ha='center', fontsize=20, fontname='DejaVu Serif')
    axs[1].annotate(f"(b) {title2}", xy=(0.5, -0.25), xycoords='axes fraction', ha='center', fontsize=20, fontname='DejaVu Serif')

    if filename is not None:
        plot_filename = filename
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")

    for ax in axs:
        ax.set_facecolor('#f8f8f8')

    plt.subplots_adjust(wspace=1.0)
    plt.tight_layout()
    plt.show()