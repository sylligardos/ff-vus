
import argparse

import pandas as pd
import torch
from tqdm import tqdm

from legacy.sylli_metrics import sylli_get_metrics
from utils.dataloader import Dataloader
from utils.scoreloader import Scoreloader
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
import os


def compare_rank(
        testing,
        experiment_dir,
):
    
    # Load the TSB dataset for all detectors
    dataloader = Dataloader(raw_data_path='data/raw')
    datasets = ['YAHOO'] if testing else dataloader.get_dataset_names()
    _, labels, filenames = dataloader.load_raw_datasets(datasets)
    
    if testing:
        n_timeseries = 10
        labels = labels[:n_timeseries]
        filenames = filenames[:n_timeseries]

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()
    all_scores, idx_failed = scoreloader.load_parallel(filenames, detectors=detectors)
    labels, filenames = scoreloader.clean_failed_idx(labels, idx_failed), scoreloader.clean_failed_idx(filenames, idx_failed)
    if len(all_scores) != len(labels) or len(all_scores) != len(filenames):
        raise ValueError(f'Size of scores and labels is not the same, scores: {len(all_scores)}, labels: {len(labels)}, filenames: {len(filenames)}')
    
    # Call all 3 metrics, for every step
    step_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ffvus = {}
    ffvusgpu = {}
    for step in step_sizes:
        ffvus[step] = VUSNumpy(
            slope_size=512, 
            step=step,  
        )
        
        ffvusgpu[step] = VUSTorch(
            slope_size=512, 
            step=step,
            device=device,
        )

    # Compute the score for each instance
    results = []
    for i, (filename, label, scores) in tqdm(enumerate(zip(filenames, labels, all_scores)), desc=f'Comparing rank'):
        for detector, score in zip(detectors, scores.T):
            for step in step_sizes:
                curr_ffvus, curr_ffvus_time = ffvus[step].compute(label, score)
                curr_ffvusgpu, curr_ffvusgpu_time = ffvusgpu[step].compute(torch.tensor(label, device=device, dtype=torch.uint8), torch.tensor(score, device=device, dtype=torch.float16))
                results.append({'Time series': filename, 'Metric': 'FF-VUS', 'Step': step, 'Detector': detector, 'Score': curr_ffvus, 'Runtime': curr_ffvus_time})
                results.append({'Time series': filename, 'Metric': 'FF-VUS-GPU', 'Step': step, 'Detector': detector, 'Score': curr_ffvusgpu, 'Runtime': curr_ffvusgpu_time})

            # curr_vus, curr_vus_time = sylli_get_metrics(label, score, 'vus', 512, existence=True)
            # results.append({'Time series': filename, 'Metric': 'FF-VUS-GPU', 'Step': 1, 'Detector': detector, 'Metric value': curr_vus, 'Runtime': curr_vus_time})

    df = pd.DataFrame(results, index='Time series')
    print(df)

    # Save results
    if not experiment_dir:
        experiment_dir = os.getcwd()

    os.makedirs(experiment_dir, exist_ok=True)
    out_path = os.path.join(experiment_dir, 'compare_rank_results.csv')
    df.to_csv(out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Compute a specific metric for tsb and all detectors'
    )

    parser.add_argument('--test', action='store_true', help='Run in testing mode (limits the data for fast testing)')
    parser.add_argument('--experiment', type=str, default=None, help='Directory to save experiment results and info')

    args = parser.parse_args()

    compare_rank(
        testing=args.test,
        experiment_dir=args.experiment,
    )