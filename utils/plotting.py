import matplotlib.pyplot as plt
import itertools

import numpy as np

from ipfi.scaler import MinMaxScaler


def plot_feature_importance(fi_values, names_to_highlight, title="Feature Importance over time",
                            metric_name="Perf.", normalized=False, save_name=None):
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#f8f8f8')

    x = range(len(fi_values))
    if normalized:
        scaler = MinMaxScaler()
        for x in fi_values:
            scaler.learn_one(x)

        fi_values = [scaler.transform_one(x) for x in fi_values]

    all_keys = list(fi_values[0].keys())

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = itertools.cycle(default_colors)

    key_to_color = {}
    for key in names_to_highlight:
        key_to_color[key] = next(color_cycle)

    for key in all_keys:
        y = [row[key] for row in fi_values]

        if key in names_to_highlight:
            color = key_to_color[key]
            label = key
            z = 3
        else:
            color = 'lightgrey'
            label = None
            z = 1

        ax.plot(x, y, color=color, label=label, linewidth=1, zorder=z)

    plt.legend()
    plt.xlabel("Instances")
    plt.ylabel("Feature importance")
    plt.title(title)
    plt.show()


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