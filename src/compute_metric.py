"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

import argparse
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch

from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from legacy.sylli_metrics import sylli_get_metrics
from utils.utils import analyze_label, natural_keys, load_synthetic, load_tsb
import time


def compute_metric(
        filenames,
        labels, 
        scores, 
        metric,
        global_mask=True, 
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
            global_mask=global_mask,
            slope_size=slope_size, 
            step=step,  
            slopes=slopes,
            existence=existence,
            conf_matrix=conf_matrix,
        )
    elif metric == 'ff_vus_pr_gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ff_vus = VUSTorch(
            global_mask=global_mask,
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
                'Global mask': global_mask,
                'Slopes': slopes,
                'Existence': existence,
                'Confusion matrix': conf_matrix,
            })

            with torch.no_grad():
                if metric == 'ff_vus_pr_gpu':
                    label, score = torch.tensor(label, device=device, dtype=torch.uint8), torch.tensor(score, device=device, dtype=torch.float16)
                    results[-1]['Slopes'], results[-1]['Existence'] = 'function', 'matrix'

                (metric_value, ff_vus_time_analysis), metric_time = ff_vus.compute(label, score)
            results[-1].update(ff_vus_time_analysis)
        else:
            if metric == 'range_auc_pr' or metric == 'vus_pr':
                results[-1].update({
                    'Slope size': slope_size
                })
            if metric == 'vus_pr':
                results[-1].update({
                    'Existence': True if existence != 'None' else False
                })
            metric_value, metric_time = sylli_get_metrics(label, score, metric, slope_size, existence=True if existence != 'None' else False)

        results[-1].update({
            'Metric': metric_name,
            'Metric value': float(metric_value),
            'Metric time': metric_time,
        })   
        
    return pd.DataFrame(results)

def compute_metric_over_dataset(
        dataset,
        metric,
        global_mask=True,
        slope_size=100,
        step=1,
        slopes='precomputed',
        existence='optimized',
        conf_matrix='dynamic',
        testing=False,
):
    # Load dataset
    if dataset == 'tsb':
        filenames, labels, scores, _ = load_tsb(testing=testing, dataset='YAHOO', n_timeseries=10)
    elif 'synthetic' in  dataset:
        filenames, labels, scores = load_synthetic(dataset=dataset, testing=testing)
    else:
        raise ValueError(f"Wrong argument for dataset: {dataset}")

    if metric == 'all':
        metrics = ['ff_vus_pr', 'vus_pr']
        # metrics = ['ff_vus_pr_gpu', 'auc_pr', 'ff_vus_pr', 'affiliation', 'range_auc_pr', 'vus_pr', 'rf']
    else:
        metrics = [metric]

    slope_sizes = [16, 32, 64, 128, 256] if 'vus_pr' in metrics or 'ff_vus_pr' in metrics else [0]

    for slope_size in slope_sizes:
        for metric in metrics:
            start_time = time.time()
            df = compute_metric(filenames, labels, scores, metric, global_mask, slope_size, step, slopes, existence, conf_matrix)
            exec_time = time.time() - start_time

            # Generate saving path and results file name
            filename = f"{dataset}_{metric.replace('_', '-').upper()}"
            if metric in ['ff_vus_pr', 'ff_vus_pr_gpu', 'range_auc_pr', 'vus_pr']:
                filename += f"_{slope_size}"
            if metric in ['ff_vus_pr', 'ff_vus_pr_gpu']:
                filename += f"_{step}_{conf_matrix}_{'globalmask' if global_mask else 'noglobalmask'}"
            if metric == 'ff_vus_pr':
                filename += f"_{slopes}_{existence}"
            filename += ".csv"
            saving_path = os.path.join('experiments', 'vus_ffvus_auc_0')

            # Save the results
            print(f"Saving results in: {filename}")
            print(df)
            print(f"Average computation time: {df['Metric time'].mean():.3f} seconds")
            
            if not testing:
                os.makedirs(os.path.join(saving_path, 'results'), exist_ok=True)
                os.makedirs(os.path.join(saving_path, 'info'), exist_ok=True)

                df.to_csv(os.path.join(saving_path, 'results', filename))

                info_df = pd.DataFrame([{
                    "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "GPU Memory (GB)": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else None,
                    "CPU": os.popen("lscpu | grep 'Model name' | awk -F ':' '{print $2}'").read().strip(),
                    "Cores": os.cpu_count(),
                    "RAM (GB)": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2) if hasattr(os, 'sysconf') else None,
                    "Platform": os.uname().sysname if hasattr(os, 'uname') else None,
                    "Platform-release": os.uname().release if hasattr(os, 'uname') else None,
                    "Python version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                    "Experiment": f"Compute {metric}",
                    "Number of results": len(df),
                    "Slope size": slope_size,
                    "Step": step,
                    "Global mask": global_mask,
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
    parser.add_argument('--global_mask', action='store_true', help='Use global mask for metric computation')
    parser.add_argument('--slope_size', type=int, default=100, help='Number of slopes used for computation')
    parser.add_argument('--step', type=int, default=1, help='Step size between slopes')
    parser.add_argument('--slopes', type=str, choices=['precomputed', 'function'], default='precomputed',
                        help='Slope generation method')
    parser.add_argument('--existence', type=str, choices=['None', 'trivial', 'optimized', 'matrix'], default='optimized',
                        help='Existence computation method')
    parser.add_argument('--conf_matrix', type=str, choices=['trivial', 'dynamic', 'dynamic_plus'], default='dynamic_plus',
                        help='Type of confusion matrix computation')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode (limits the data for fast testing)')

    args = parser.parse_args()

    if args.dataset == 'all_synthetic':
        synthetic_dir = os.path.join('data', 'synthetic')
        datasets = [x for x in os.listdir(synthetic_dir) if not '10000000000' in x]
    else:
        datasets = [args.dataset]
    datasets.sort(key=natural_keys)

    for dataset in datasets:
        compute_metric_over_dataset(
            dataset=dataset,
            metric=args.metric,
            global_mask=True, # args.global_mask,
            slope_size=args.slope_size,
            step=args.step,
            slopes=args.slopes,
            existence=args.existence,
            conf_matrix=args.conf_matrix,
            testing=args.testing
        )