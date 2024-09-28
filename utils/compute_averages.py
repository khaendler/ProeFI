import numpy as np
import pandas as pd


def compute_total_avg(stats):
    array = np.array(stats)
    column_averages = np.mean(array, axis=0)
    overall_average = np.mean(column_averages)
    avg_stats = overall_average.tolist()

    return avg_stats


def compute_stats_avgs(stats):
    array = np.array(stats)
    column_averages = np.mean(array, axis=0)

    return column_averages.tolist()


def compute_fi_avgs(dicts):
    num_dicts = len(dicts)
    if num_dicts == 0:
        return {'importance_values': []}

    # Get the list of importance_values DataFrames, ensuring the keys are sorted
    importance_lists = [
        pd.DataFrame([{k: v for k, v in sorted(d.items())} for d in dict['importance_values']])
        for dict in dicts
    ]

    # Concatenate all DataFrames along a new axis and compute the mean
    concatenated = np.stack([df.values for df in importance_lists])
    averages_array = np.mean(concatenated, axis=0)

    average_importance_values = pd.DataFrame(averages_array, columns=importance_lists[0].columns)
    average_importance_values_list = average_importance_values.to_dict(orient='records')
    average_importance_values_list[0] = {}

    return {'importance_values': average_importance_values_list}

