"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

from sklearn import metrics
import time
import functools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def time_it(func):
    """Wrapper to measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper


def auc_pr_wrapper(label, score, interpolation='stepwise'):
    if interpolation == 'stepwise':
        auc_pr = metrics.average_precision_score(label, score)    
    elif interpolation == 'linear':
        pr, rec, _ = metrics.precision_recall_curve(label, score)
        auc_pr = metrics.auc(rec, pr)
    else:
        raise ValueError(f"Unknown argument for interpolation: {interpolation}")
    
    return auc_pr


def compare_vectors(v1, v2):
    """
    Compare two vectors and return True if their min, max, and mean differences are below a threshold.
    """
    diff = np.abs(v1 - v2)
    return np.min(diff), np.max(diff), np.mean(diff)

def visualize_differences(labels, slopes_func, slopes_pre, tol):
    """
    Plot the labels and slopes to visualize differences.
    """
    slopes_diff_mask = slopes_func != slopes_pre
    different_indexes = np.where(np.any(slopes_diff_mask, axis=1))

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 6))

    ax[0].plot(labels, label="Original Labels")
    ax[0].legend()

    sns.lineplot(data=slopes_func.T, ax=ax[1])
    ax[1].set_title("Computed Slopes (Function)")
    ax[1].legend()

    sns.lineplot(data=slopes_pre.T, ax=ax[2])
    ax[2].set_title("Computed Slopes (Precomputed)")
    ax[2].legend()

    for slope_func, slope_pre in zip(slopes_func, slopes_pre):
        different_indexes = np.where(np.abs(slope_func - slope_pre) > tol)[0]
        ax[1].scatter(different_indexes, slope_func[different_indexes], color='red', label='Different values', alpha=0.3)
        ax[2].scatter(different_indexes, slope_pre[different_indexes], color='red', label='Different values', alpha=0.3)

    plt.show()

def visualize_differences_1d(labels, slopes_func, slopes_pre, tol):
    """
    Visualize differences between two slope computations.
    
    Args:
        labels (np.ndarray): The original label sequence.
        slopes_func (np.ndarray): Slopes computed using the function.
        slopes_pre (np.ndarray): Slopes computed using the precomputed method.
        tol (float): Tolerance for considering differences.
    """
    slopes_diff_mask = np.abs(slopes_func - slopes_pre) > tol
    different_indexes = np.where(slopes_diff_mask)[0]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

    # Plot original labels
    ax[0].plot(labels, label="Original Labels", color='blue')
    ax[0].legend()

    # Plot computed slopes (Function)
    ax[1].plot(slopes_func, label="Computed Slopes (Function)", color='green')
    ax[1].scatter(different_indexes, slopes_func[different_indexes], color='red', label="Differences", alpha=0.6)
    ax[1].legend()

    # Plot computed slopes (Precomputed)
    ax[2].plot(slopes_pre, label="Computed Slopes (Precomputed)", color='orange')
    ax[2].scatter(different_indexes, slopes_pre[different_indexes], color='red', label="Differences", alpha=0.6)
    ax[2].legend()

    plt.show()

def compute_slopes_and_compare(labels, compute_slope_func, compute_slope_precomputed, tol=1e-12):
    """
    Compute slopes using two different methods and compare them.
    If they differ, visualize the differences.
    """
    slopes_func, func_time = time_it(compute_slope_func)(labels)
    slopes_pre, pre_time = time_it(compute_slope_precomputed)(labels)

    # slopes_pre = np.average(slopes_pre, axis=0)
    # sns.lineplot(slopes_pre.T)
    # plt.show()

    min_dif, max_dif, mean_dif = compare_vectors(slopes_func, slopes_pre)

    # print(f"Min difference: {min_dif}, Max difference: {max_dif}, Mean difference: {mean_dif}")
    # Compute speedup
    if func_time < pre_time:
        speedup = pre_time / func_time
        print(f"Function compute_slope_func is {speedup:.2f} times faster. Absolute gain: {(pre_time - func_time):.5f} secs")
    else:
        speedup = func_time / pre_time
        print(f"Function compute_slope_precomputed is {speedup:.2f} times faster. Absolute gain: {(func_time - pre_time):.5f} secs")

    if min_dif > tol or max_dif > tol or mean_dif > tol :
        visualize_differences(labels, slopes_func, slopes_pre, tol)
    
def get_anomalies_coordinates(label, include_edges=True):
    '''
    Return the starting and ending points of all anomalies in label
    If edges is True, then return the first and last point,
    only if anomalies exist in the very beggining or very end.
    '''
    diff = np.diff(label)
    start_points = np.where(diff == 1)[0] + 1
    end_points = np.where(diff == -1)[0]

    if include_edges:
        if label[-1]:
            end_points = np.append(end_points, len(label) - 1)
        if label[0]:
            start_points = np.append([0], start_points)
        if start_points.shape != end_points.shape:
            raise ValueError(f'The number of start and end points of anomalies does not match, {start_points} != {end_points}')
        
    return start_points, end_points

def analyze_label(label):
    """
    Analyze a binary time series anomaly label vector
    and return basic metrics.

    Args:
        label (np.ndarray): a 1D numpy array of 0s and 1s
    Return:
        length (int): total length of the label
        n_anomalies (int): number of anomalies in the label
        anomalies_avg_length (int): the average length of the anomalies
    """
    length = len(label)

    start_points, end_points = get_anomalies_coordinates(label)
    end_points += 1     # end points are exclusive, so this is necessary
    
    n_anomalies = start_points.shape[0]
    anomaly_lengths = np.array([e - s for s, e in zip(start_points, end_points)])
    anomalies_avg_length = np.mean(anomaly_lengths)
    
    return length, n_anomalies, anomalies_avg_length