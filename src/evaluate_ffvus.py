"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


from vus.vus_numpy import VUSNumpy

from utils.scoreloader import Scoreloader
from utils.dataloader import Dataloader
from legacy.sylli_metrics import sylli_get_metrics
from utils.utils import auc_pr_wrapper, time_it
from legacy.utils.metrics import metricor

import argparse
import math
import pandas as pd
import os
import time
import numpy as np

def evaluate_ffvus_random(testing):
    tic = time.time()
    dataloader = Dataloader(raw_data_path='data/raw')
    datasets = ['YAHOO'] if testing else dataloader.get_dataset_names()
    _, labels, filenames = dataloader.load_raw_datasets(datasets)
    
    if testing:
        labels = labels[:2]
        filenames = filenames[:2]
    else:
        zipped = list(zip(labels, filenames))
        sampled = np.random.choice(len(zipped), size=50, replace=False)
        labels, filenames = zip(*[zipped[i] for i in sampled])

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()
    scores, idx_failed = scoreloader.load_parallel(filenames)
    labels, filenames = scoreloader.clean_failed_idx(labels, idx_failed), scoreloader.clean_failed_idx(filenames, idx_failed)
    if len(scores) != len(labels):
        raise ValueError(f'Size of scores and labels is not the same, scores: {len(scores)} labels: {len(labels)}')

    results = []
    slope_size = 100 if testing else np.random.randint(low=1, high=256)
    step = 1
    zita = (1 / math.sqrt(2))
    slopes = 'precomputed' if testing else np.random.choice(['precomputed', 'function'])
    existence = 'optimized' if testing else np.random.choice(['optimized'])   # 'trivial', 'matrix' are also options
    conf_matrix = 'dynamic' if testing else np.random.choice(['trivial', 'dynamic', 'dynamic_plus'])          
    interpolation = 'stepwise'  # ['linear', 'stepwise']
    
    ff_vus_numpy = VUSNumpy(
        slope_size=slope_size, 
        step=step, 
        zita=zita, 
        slopes=slopes,
        existence=existence,
        conf_matrix=conf_matrix, 
        interpolation=interpolation,
    )

    ff_auc_numpy = VUSNumpy(
        slope_size=0, 
        step=1,
        zita=zita, 
        slopes=slopes,
        existence='None',
        conf_matrix=conf_matrix,
        interpolation=interpolation,
    )

    print(f">> Current settings: Slope size {slope_size}, step {step}, slopes {slopes}, existence {existence}, conf. matrix {conf_matrix}, interpolation {interpolation}")
    for i, (filename, label, curr_scores) in enumerate(zip(filenames, labels, scores)):
        detector_idx = np.random.randint(0, 11)
        detector = detectors[detector_idx]
        score = curr_scores[:, detector_idx]

        curr_result = {
            'Time series': filename,
            'Detector': detector,
            'Length': label.shape[0],
        }

        auc_pr, auc_pr_time = time_it(auc_pr_wrapper)(label, score, interpolation)
        curr_result.update({
            'AUC-PR': auc_pr,
            'AUC-PR time': auc_pr_time
        })

        ff_auc_pr, ff_auc_time_analysis = ff_auc_numpy.compute(label, score)
        ff_auc_time = sum([ff_auc_time_analysis[key] for key in ff_auc_time_analysis.keys()])
        curr_result.update({
            'FF-AUC-PR': ff_auc_pr,
            'FF-AUC-PR time': ff_auc_time,
        })
        curr_result.update(ff_auc_time_analysis)

        vus_pr, vus_time = time_it(sylli_get_metrics)(label, score, 'vus_pr', slope_size)
        curr_result.update({
            'VUS-PR': vus_pr,
            'VUS-PR time': vus_time,
        })

        ff_vus_pr, ff_vus_time_analysis = ff_vus_numpy.compute(label, score)
        ff_vus_time = sum([ff_vus_time_analysis[key] for key in ff_vus_time_analysis.keys()])
        curr_result.update({
            'FF-VUS-PR': ff_vus_pr,
            'FF-VUS-PR time': ff_vus_time,
        })
        curr_result.update(ff_vus_time_analysis)

        print(f"({i}) AUC-PR - FF-AUC-PR: {abs(auc_pr - ff_auc_pr):.5f}, VUS-PR - FF-VUS-PR: {abs(vus_pr - ff_vus_pr):.5f}, AUC Slow down: {(ff_auc_time / auc_pr_time):.2f}, VUS Speed up: {(vus_time / ff_vus_time):.2f}, Length: {label.shape[0]}")
        # print(f">> AUC-PR: {auc_pr:.3f}, FF AUC-PR: {ff_auc_pr:.3f}, VUS-PR: {vus_pr:.3f}, FF VUS-PR: {ff_vus_pr:.3f}")
        
        curr_result.update({
            "AUC-PR - FF-AUC-PR": abs(auc_pr - ff_auc_pr), 
            "VUS-PR - FF-VUS-PR": abs(vus_pr - ff_vus_pr), 
            "AUC-PR Slow down": ff_auc_time / auc_pr_time, 
            "VUS-PR Speed up": vus_time / ff_vus_time,
            "AUC-PR equal": abs(auc_pr - ff_auc_pr) < 1e-14,
            "VUS-PR equal": abs(vus_pr - ff_vus_pr) < 1e-14,
        })
        
        results.append(curr_result)

    df = pd.DataFrame(results)
    
    curr_experiment_name = f"evaluate_{slope_size}_{step}_{slopes}_{existence}_{conf_matrix}.csv"
    saving_path = os.path.join("experiments", "14_04_2025", "results", curr_experiment_name)
    df.to_csv(saving_path)

    info_df = pd.DataFrame([{
        "Experiment": "Evaluate FF-VUS-PR",
        "Slope size": slope_size,
        "Step": step,
        "Slopes": slopes,
        "Existence": existence,
        "Confusion matrix": conf_matrix,
        "Time": time.time() - tic,
    }])
    info_df.to_csv(os.path.join("experiments", "14_04_2025", "info", curr_experiment_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Run experiments on the current implementation of the VUS metric'
    )
    parser.add_argument('--testing', action='store_true', help='Run in testing mode (limits the data for fast testing)')
    
    args = parser.parse_args()

    evaluate_ffvus_random(testing=args.testing)