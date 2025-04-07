"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


from src.utils.utils import time_it

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma
from tqdm import tqdm
import os


def generate_synthetic_labels(length=10000, n_anomalies=10, avg_anomaly_length=100):
    """
    Generate a binary label sequence (0 for normal, 1 for anomaly).

    Parameters:
    - length: total length of the time series
    - n_anomalies: number of anomalies 
    - avg_anomaly_length: average length of an anomalous segment
    
    Returns:
    - labels: np.array of shape (length,), binary values
    """
    label = np.zeros(length, dtype=int)
    
    anomalies_index = np.random.choice(length, size=n_anomalies, replace=False)
    anomalies_length = np.abs(np.random.normal(loc=avg_anomaly_length, scale=avg_anomaly_length//4, size=n_anomalies)).astype(int)

    start_points = anomalies_index - (anomalies_length // 2)
    end_points = start_points + anomalies_length
    for i, s, e in zip(anomalies_index, start_points, end_points):
        label[s: e] = 1
    
    return label, start_points, end_points

def generate_score_from_labels(label, start_points, end_points, detection_prob=0.9, lag_ratio=10, noise=0.03, false_positive_strength=0.05):
    """
    Simulate an anomaly score based on binary labels with possible detection errors.

    Returns:
    - score: array of shape (len(labels),) with float scores in [0, 1]
    """
    length = len(label)
    n_anomalies = len(start_points)
    anomaly_lengths = []
    score = np.abs(np.random.normal(loc=0.0, scale=noise, size=length))

    detection = np.random.uniform(size=n_anomalies) < detection_prob
    
    for i in range(n_anomalies):
        curr_length = end_points[i] - start_points[i]
        anomaly_lengths.append(curr_length)
        if detection[i]:
            half_length = curr_length // 2

            # Apply random lag
            lag = curr_length / lag_ratio
            lag_shift = np.random.randint(-lag, lag + 1)
            curr_start = np.clip(start_points[i] - half_length + lag_shift, 0, length - 1)
            curr_end = np.clip(end_points[i] + half_length + lag_shift, curr_start + 1, length)

            # Create a gamma-shaped bump centered in the window
            x = np.arange(curr_start, curr_end)
            a = np.random.randint(5, 11)
            scale = (curr_end - curr_start) / (a * 2)
            gamma_values = gamma.pdf(x, a=a, loc=curr_start, scale=scale)
            gamma_values = (gamma_values - np.min(gamma_values)) / (np.max(gamma_values) - np.min(gamma_values))

            detection_scale = np.random.random_sample(1)[0] * (0.8 - 0.4) + 0.5
            score[x] += (gamma_values * detection_scale)

    # Add false positives
    n_false_positives = np.random.randint(0, 3)
    false_positive_lengths = np.random.randint(0, np.max(anomaly_lengths), size=n_false_positives)
    false_positive_indexes = np.random.randint(0, length, size=n_false_positives)
    
    for idx, curr_len in zip(false_positive_indexes, false_positive_lengths):
        curr_fp = np.random.normal(loc=0.5, scale=false_positive_strength, size=curr_len)
        
        if (idx + curr_len) - length < 0:
            score[idx: idx + curr_len] += curr_fp
            
    return np.clip(score, 0, 1)


def main():
    n_labels = 10
    length = 1000
    n_anomalies = 10
    avg_anomaly_length = 100
    
    labels = np.zeros((n_labels, length))
    scores = np.zeros((n_labels, length))
    labels_times = np.zeros((n_labels))
    scores_times = np.zeros((n_labels))

    os.makedirs("generated_data", exist_ok=True)
    
    for i in tqdm(range(n_labels), desc="Generating labels"):
        context, labels_times[i] = time_it(generate_synthetic_labels)(length=length, n_anomalies=n_anomalies)
        labels[i], start_points, end_points = context
        
        scores[i] = generate_score_from_labels(labels[i], start_points, end_points, detection_prob=0.9, lag_ratio=10, noise=0.03, false_positive_strength=0.05)
    
    param_str = f"n_labels_{n_labels}_length_{length}_n_anomalies_{n_anomalies}_avg_anomaly_length_{avg_anomaly_length}"
    np.savetxt(f"generated_data/labels_scores_{param_str}.csv", np.vstack((labels, scores)))

    # Optionally, save the time measurements as well
    # np.savetxt(f"generated_data/times_{param_str}.csv", labels_times=labels_times, scores_times=scores_times)

    print("Data generation completed and saved successfully!")

if __name__ == "__main__":
    main()