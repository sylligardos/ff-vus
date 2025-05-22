"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch

from utils.scoreloader import Scoreloader
from utils.dataloader import Dataloader
from legacy.sylli_metrics import sylli_get_metrics
from utils.utils import analyze_label, natural_keys

import argparse
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch


def load_tsb(testing=False):
    # Load the TSB-UAD benchmark
    dataloader = Dataloader(raw_data_path='data/raw')
    datasets = ['MITDB'] if testing else dataloader.get_dataset_names()
    _, labels, filenames = dataloader.load_raw_datasets(datasets)
    
    if testing:
        labels = labels[:10]
        filenames = filenames[:10]

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()
    scores, idx_failed = scoreloader.load_parallel(filenames)
    labels, filenames = scoreloader.clean_failed_idx(labels, idx_failed), scoreloader.clean_failed_idx(filenames, idx_failed)
    if len(scores) != len(labels) or len(scores) != len(filenames):
        raise ValueError(f'Size of scores and labels is not the same, scores: {len(scores)}, labels: {len(labels)}, filenames: {len(filenames)}')

    # Pick a random detector score for each label
    if testing:
        detectors_idx = np.zeros(len(labels)).astype(int)
    else:    
        detectors_idx = np.random.randint(0, len(detectors), size=len(labels))
    scores = [score[:, idx] for score, idx in zip(scores, detectors_idx)]

    detectors_selected = [detectors[idx] for idx in detectors_idx]

    return filenames, labels, scores, detectors_selected

def load_synthetic(dataset, testing=False):
    # Load dataset
    dataset_path = os.path.join('data', 'synthetic', dataset)
    csv_files = [x for x in os.listdir(dataset_path) if '.csv' in x or '.npy' in x]

    labels = []
    scores = []

    for file in tqdm(csv_files, desc="Loading synthetic"):
        if '.csv' in file:
            data = np.loadtxt(os.path.join(dataset_path, file), delimiter=",")
        else:
            data = np.load(os.path.join(dataset_path, file))
        label = data[:, 0]
        score = data[:, 1]
        labels.append(label)
        scores.append(score)
        if testing: break

    labels = np.array(labels)
    scores = np.array(scores).round(2)

    return csv_files, labels, scores

def compute_metric(
        filenames,
        labels, 
        scores, 
        metric, 
        slope_size=None, 
        step=None, 
        slopes=None, 
        existence=None, 
        conf_matrix=None, 
):
    # Compute metric
    metric_name = metric.replace('_', '-').upper()
    results = []
    
    if metric == 'ff_vus_pr':
        ff_vus = VUSNumpy(
            slope_size=slope_size, 
            step=step,  
            slopes=slopes,
            existence=existence,
            conf_matrix=conf_matrix,
        )
    elif metric == 'ff_vus_pr_gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ff_vus = VUSTorch(
            slope_size=slope_size, 
            step=step,
            conf_matrix=conf_matrix,
            device=device,
        )

    for filename, label, score in tqdm(zip(filenames, labels, scores), desc=f'Computing {metric}', total=len(labels)):
        length, n_anomalies, anomalies_avg_length = analyze_label(label)
        results.append({
            "Time series": filename,
            "Length": length,
            "Number of anomalies": n_anomalies,
            "Anomalies average length": float(anomalies_avg_length),
        })

        if metric == 'ff_vus_pr' or metric == 'ff_vus_pr_gpu': 
            results[-1].update({
                'Slope size': slope_size,
                'Step': step,  
                'Slopes': slopes,
                'Existence': existence,
                'Confusion matrix': conf_matrix,
            })

            if metric == 'ff_vus_pr_gpu':
                label, score = torch.tensor(label, device=device), torch.tensor(score, device=device)
                results[-1]['Slopes'], results[-1]['Existence'] = 'function', 'matrix'

            (metric_value, ff_vus_time_analysis), metric_time = ff_vus.compute(label, score)
            results[-1].update(ff_vus_time_analysis)
        else:
            if metric == 'range_auc_pr' or metric == 'vus_pr':
                results[-1].update({
                    'Slope size': slope_size
                })

            metric_value, metric_time = sylli_get_metrics(label, score, metric, slope_size)

        results[-1].update({
            'Metric': metric_name,
            'Metric value': float(metric_value),
            'Metric time': metric_time,
        })   
        
    return pd.DataFrame(results)

def compute_metric_over_dataset(
        dataset,
        metric,
        slope_size=100,
        step=1,
        slopes='precomputed',
        existence='optimized',
        conf_matrix='dynamic',
        testing=False,
):
    # Load dataset
    if dataset == 'tsb':
        filenames, labels, scores, _ = load_tsb(testing=testing)
    elif 'synthetic' in  dataset:
        filenames, labels, scores = load_synthetic(dataset=dataset, testing=testing)
    else:
        raise ValueError(f"Wrong argument for dataset: {dataset}")

    if metric == 'all':
        metrics = ['ff_vus_pr', 'ff_vus_pr_gpu', 'auc_pr', 'affiliation', 'range_auc_pr'] # , 'vus_pr', 'rf'
    else:
        metrics = [metric]

    for metric in metrics:
        df = compute_metric(filenames, labels, scores, metric, slope_size, step, slopes, existence, conf_matrix)

        # Generate saving path and results file name
        filename = f"{dataset}_{metric.replace('_', '-').upper()}"
        if metric in ['ff_vus_pr', 'ff_vus_pr_gpu', 'range_auc_pr', 'vus_pr']:
            filename += f"_{slope_size}"
        if metric in ['ff_vus_pr', 'ff_vus_pr_gpu']:
            filename += f"_{step}_{conf_matrix}"
        if metric == 'ff_vus_pr':
            filename += f"_{slopes}_{existence}"
        filename += ".csv"
        saving_path = os.path.join('experiments', 'vus_ffvus_auc_synthetic')

        # Save the results
        print(filename)
        print(df)
        print(f"Average computation time: {df['Metric time'].mean():.3f} seconds")
        
        if not testing:
            df.to_csv(os.path.join(saving_path, 'results', filename))

            info_df = pd.DataFrame([{
                "Experiment": f"Compute {metric}",
                "Number of results": len(df),
                "Slope size": slope_size,
                "Step": step,
                "Slopes": slopes,
                "Existence": existence,
                "Confusion matrix": conf_matrix,
                "Time": df.iloc[:, -1].sum(),
            }])
            info_df.to_csv(os.path.join(saving_path, 'info', filename), index=False)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Compute a specific metric for a whole dataset'
    )

    parser.add_argument('--dataset', type=str, required=True, help='Path or name of the dataset')
    parser.add_argument('--metric', type=str, required=True, choices=['ff_vus_pr', 'ff_vus_pr_gpu', 'vus_pr', 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'all'], 
                        help='Metric to compute (e.g., VUS, AUC-PR, etc.)')
    parser.add_argument('--slope_size', type=int, default=100, help='Number of slopes used for computation')
    parser.add_argument('--step', type=int, default=1, help='Step size between slopes')
    parser.add_argument('--slopes', type=str, choices=['precomputed', 'function'], default='precomputed',
                        help='Slope generation method')
    parser.add_argument('--existence', type=str, choices=['None', 'trivial', 'optimized', 'matrix'], default='optimized',
                        help='Existence computation method')
    parser.add_argument('--conf_matrix', type=str, choices=['trivial', 'dynamic', 'dynamic_plus'], default='dynamic',
                        help='Type of confusion matrix computation')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode (limits the data for fast testing)')

    args = parser.parse_args()

    if args.dataset == 'all_synthetic':
        synthetic_dir = os.path.join('data', 'synthetic')
        datasets = [x for x in os.listdir(synthetic_dir) if not '1000000000' in x and not '10000000000' in x]
    else:
        datasets = [args.dataset]
    datasets.sort(key=natural_keys)

    for dataset in datasets:
        compute_metric_over_dataset(
            dataset=dataset,
            metric=args.metric,
            slope_size=args.slope_size,
            step=args.step,
            slopes=args.slopes,
            existence=args.existence,
            conf_matrix=args.conf_matrix,
            testing=args.testing
        )