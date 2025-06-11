import matplotlib.pyplot as plt
import itertools

import numpy as np

from ipfi.scaler import MinMaxScaler


def plot_feature_importance(fi_values,
                            names_to_highlight=(),
                            top_k=None,
                            title="Feature Importance over time",
                            metric_name="Perf.",
                            normalized=False,
                            save_name=None):
    """
    Plot feature importance trajectories.

    Parameters
    ----------
    fi_values : list of dict
        A sequence of per-instance feature-importance mappings.
    names_to_highlight : iterable of str, optional
        Specific feature names to color/highlight.
    top_k : int or None, default=None
        If an integer, highlight the top k features by total importance.
    title : str
        Plot title.
    metric_name : str
        Label for the y-axis (unused in plotting code but kept for extensibility).
    normalized : bool
        If True, min/max normalize each feature’s time series before plotting.
    save_name : str or None
        If given, will save the figure to this filename.
    """
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#f8f8f8')

    x = list(range(len(fi_values)))

    # Optional normalization
    if normalized:
        scaler = MinMaxScaler()
        for row in fi_values:
            scaler.learn_one(row)
        fi_values = [scaler.transform_one(row) for row in fi_values]

    # All feature keys
    all_keys = list(fi_values[0].keys())

    # Determine which to highlight
    highlight_set = set(names_to_highlight)

    if top_k is not None and 0 < top_k < len(all_keys)+1:
        # Sum absolute importance across all instances
        totals = {key: sum(abs(row[key]) for row in fi_values) for key in all_keys}
        # Pick top_k keys by total importance
        top_keys = sorted(totals, key=totals.get, reverse=True)[:top_k]
        highlight_set.update(top_keys)

    # Prepare colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = itertools.cycle(default_colors)
    key_to_color = {key: next(color_cycle) for key in highlight_set}

    # Plot every feature
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

    plt.legend()
    plt.xlabel("Instances")
    plt.ylabel("Feature importance")
    plt.title(title)
    plt.show()

    if save_name:
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
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