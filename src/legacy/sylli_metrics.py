"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS

Note: I do not claim any of the code written here. This file is just a cleaner
    version of the code found in the TSB-UAD repository, just for 
    my experiments. To have clean execution times per metric. 
"""

from .utils.metrics import metricor
from .basic_metrics import basic_metricor
from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from utils.utils import auc_pr_wrapper, time_it

import numpy as np


@time_it
def sylli_get_metrics(label, score, metric, slope_size=None):
    """
    Compute the selected evaluation metric given the labels, anomaly scores,
    and slope_size when required.

    Args:
        label (np.ndarray): Binary ground truth labels (0s and 1s).
        score (np.ndarray): Anomaly score vector.
        metric (str): The name of the metric to compute. Supported values:
                      'vus_pr', 'rf', 'affiliation', 'range_auc_pr'.
        slope_size (int): Size of the sliding window for range-based metrics.

    Returns:
        float: Computed value of the selected metric.
    """
    if metric == 'vus_pr' and slope_size is not None:
        compute_legacy = metricor()
        _, _, _, _, metric_value = compute_legacy.RangeAUC_volume_opt_mem(label, score, slope_size)
    elif metric == 'rf':
        metric_value = basic_metricor().sylli_RF(label, score)
    elif metric == 'affiliation':
        metric_value = sylli_affiliation(label, score)
    elif metric == 'range_auc_pr' and slope_size is not None:
        metric_value = metricor().sylli_RangeAUC(label, score, window=slope_size)
    elif metric == 'auc_pr':
        metric_value = auc_pr_wrapper(label, score)
    else:
        raise ValueError(f"Argument {metric} for metric is not valid or window size is required!")
    
    return metric_value

def sylli_affiliation(label, score):
    discrete_score = np.array(score > 0.5, dtype=np.float32)
    events_pred = convert_vector_to_events(discrete_score)
    events_gt = convert_vector_to_events(label)
    Trange = (0, len(discrete_score))
    
    affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
    
    pr = affiliation_metrics['Affiliation_Precision']
    rec = affiliation_metrics['Affiliation_Recall']

    f1_score = 2 * ((pr * rec) / (pr + rec))
    return f1_score