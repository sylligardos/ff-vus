"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from tqdm import tqdm
import os
import time
from datetime import datetime


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
    label = np.zeros(length, dtype=np.int8)
    
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
    score = np.abs(np.random.normal(loc=0.0, scale=noise, size=length), dtype=np.float16)

    detection = np.random.uniform(size=n_anomalies) < detection_prob
    
    for i in range(n_anomalies):
        curr_length = end_points[i] - start_points[i]
        anomaly_lengths.append(curr_length)
        if detection[i]:
            half_length = max(1, curr_length // 2)

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


def generate_synthetic(
    n_timeseries=10,
    ts_length=1000,
    n_anomalies=10,
    avg_anomaly_length=100,
    testing=False
):
    if n_anomalies * avg_anomaly_length > 0.6 * ts_length:
        raise ValueError(f"The total anomaly ratio cannot be more than 60%, current {(((n_anomalies * avg_anomaly_length) / ts_length) * 100):.2f}%")

    total_start = time.time()
    dir_path = os.path.join("data", "synthetic", f"synthetic_length_{ts_length}_n_anomalies_{n_anomalies}_avg_anomaly_length_{avg_anomaly_length}")
    os.makedirs(dir_path, exist_ok=True)
    ts_template_name = f"syn_{ts_length}_{n_anomalies}_{avg_anomaly_length}"
    
    # labels = np.zeros((n_timeseries, ts_length))
    # scores = np.zeros((n_timeseries, ts_length))
    times = np.zeros((n_timeseries))
    
    for i in tqdm(range(n_timeseries), desc="Generating labels"):
        tic = time.time()
        label, start_points, end_points = generate_synthetic_labels(length=ts_length, n_anomalies=n_anomalies, avg_anomaly_length=avg_anomaly_length)
        score = generate_score_from_labels(label, start_points, end_points, detection_prob=0.9, lag_ratio=10, noise=0.03, false_positive_strength=0.05)
        toc = time.time()

        times[i] = toc - tic
        ts_name = f"{ts_template_name}_{i}"
        
        # Uncomment if you want to see the generated labels and scores
        print(testing)
        if testing:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
            ax[0].plot(label)
            ax[1].plot(score)
            fig.suptitle(ts_name)
            plt.tight_layout()
            plt.show()

        if not testing:
            np.savetxt(os.path.join(dir_path, f"{ts_name}.csv"), np.vstack((label, score)).T, fmt=["%d", "%.2f"], delimiter=",")

    total_time = time.time() - total_start
    total_time_min = total_time / 60

    print("Data generation completed, parameters:")
    print(f"- Number of time series: {n_timeseries}")
    print(f"- Time series length: {ts_length}")
    print(f"- Number of anomalies: {n_anomalies}")
    print(f"- Average length of anomalies: {avg_anomaly_length}")
    print(f"- Total generation time: {total_time:.2f} seconds ({total_time_min:.2f} min)")
    print(f"- Saved at: {dir_path}")

    # Save this info to file
    if not testing:
        date_str = "10_04_2025"
        info_dir = os.path.join("experiments", date_str, "synthetic_info")
        os.makedirs(info_dir, exist_ok=True)
        info_path = os.path.join(info_dir, f"{ts_template_name}.csv")

        with open(info_path, "w") as f:
            f.write(f"date,{date_str}\n")
            f.write(f"n_timeseries,{n_timeseries}\n")
            f.write(f"ts_length,{ts_length}\n")
            f.write(f"n_anomalies,{n_anomalies}\n")
            f.write(f"avg_anomaly_length,{avg_anomaly_length}\n")
            f.write(f"output_directory,{dir_path}\n")
            f.write(f"total_generation_time_seconds,{total_time}\n")
            f.write(f"total_generation_time_minutes,{total_time_min}\n")
            # f.write(f"avg_generation_time_per_ts,{np.mean(times)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Generate synthetic labels and scores to evaluate time series anomaly detection metrics'
    )
    parser.add_argument('-t', '--n_timeseries', type=int, help='the number of synthetic pairs of labels and scores to produce')
    parser.add_argument('-l', '--ts_length', type=int, help='the length of the generated time series')
    parser.add_argument('-n', '--n_anomalies', type=int, help='the number of anomalies per label')
    parser.add_argument('-a', '--avg_anomaly_length', type=int, help='the average length of the induced anomalies')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode (plots the data and does not save them)')

    args = parser.parse_args()

    generate_synthetic(
        n_timeseries=args.n_timeseries,
        ts_length=args.ts_length,
        n_anomalies=args.n_anomalies,
        avg_anomaly_length=args.avg_anomaly_length,
        testing=args.testing
    )