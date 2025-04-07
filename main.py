"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""


from vus.vus_numpy import VUSNumpy

from src.utils.scoreloader import Scoreloader
from src.utils.dataloader import Dataloader
from src.old_vus.metrics import get_metrics
from src.utils.utils import auc_pr_wrapper

import argparse
import time
import math
import torch

def main(testing):
    dataloader = Dataloader(raw_data_path='data/raw')
    datasets = ['Occupancy'] if testing else dataloader.get_dataset_names()
    _, labels, filenames = dataloader.load_raw_datasets(datasets)
    
    if testing:
        labels = labels[:10]
        filenames = filenames[:10]

    scoreloader = Scoreloader('data/scores')
    detectors = scoreloader.get_detector_names()
    scores, idx_failed = scoreloader.load_parallel(filenames)
    if len(scores) != len(labels):
        raise ValueError(f'Size of scores and labels is not the same, scores: {len(scores)} labels: {len(labels)}')

    results = []
    slope_size = 100
    step = 1
    zita = (1 / math.sqrt(2))
    slopes='precomputed'           # ['precomputed', 'function']
    existence = 'optimized'        # [None, 'trivial', 'optimized', 'matrix']
    conf_matrix = 'dynamic'     # ['trivial', 'dynamic', 'dynamic_plus']          
    interpolation = 'stepwise'  # ['linear', 'stepwise']
    metric = 'vus_pr'           # ['vus_pr', 'vus_roc', 'all']
    compare = True

    vus_numpy = VUSNumpy(
        slope_size=slope_size, 
        step=step, 
        zita=zita, 
        slopes=slopes,
        existence=existence,
        conf_matrix=conf_matrix, 
        interpolation=interpolation,
        metric=metric, 
    )

    for filename, label, curr_scores in zip(filenames, labels, scores):
        for detector, score in zip(detectors, curr_scores.T):
            curr_result = {
                'Time series': filename,
                'Detector': detector,
                'Length': label.shape[0],
                'Slope size': slope_size,
                'Step': step,
            }

            auc_pr, auc_pr_time = auc_pr_wrapper(label, score, interpolation)
            curr_result.update({
                'AUC-PR': auc_pr,
                'AUC-PR time': auc_pr_time
            })

            # TODO: Fix small difference between VUS and FF-VUS
            ff_vus_pr, ff_vus_time = vus_numpy.compute(label, score)
            # print(ff_vus_pr, ff_vus_time)
            curr_result.update({
                'FF-VUS-PR': ff_vus_pr,
                # 'FF-AUC-PR': ff_auc_pr,
                'FF-VUS time': ff_vus_time,
            })

            if compare:
                tic = time.time()
                repo_metrics, repo_existence = get_metrics(score, label, metric='vus', version='opt', slidingWindow=slope_size, thre=-1)
                vus_time = time.time() - tic
                vus_pr = repo_metrics['VUS_PR']
                curr_result.update({
                    'VUS-PR': vus_pr,
                    'VUS time': vus_time,
                })
                print(f"{(ff_vus_pr - vus_pr)}, {(vus_time/ff_vus_time):.2f}")
                # print(f"({i}, {j}) AUC-PR: {auc_pr:.5f}, FF AUC-PR: {ff_auc_pr:.5f}, Diff.: {abs(auc_pr - ff_auc_pr)}, VUS-PR: {vus_pr:.5f}, FF VUS-PR: {ff_vus_pr:.5f}, Diff.: {abs(vus_pr - ff_vus_pr)}, AUC Slow down: {(ff_vus_time / auc_pr_time):.2f}, VUS Speed up: {(vus_time / ff_vus_time):.2f}, Length: {y[i].shape[0]}")
                # print(f"({i}, {j}) AUC-PR: {auc_pr:.5f}, FF AUC-PR: {ff_auc_pr:.5f}, Diff.: {abs(auc_pr - ff_auc_pr)}, VUS-PR: {vus_pr:.5f}, FF VUS-PR: {ff_vus_pr:.5f}, Diff.: {abs(vus_pr - ff_vus_pr)}, AUC Slow down: {(ff_vus_time / auc_pr_time):.2f}, VUS Speed up: {(vus_time / ff_vus_time):.2f}, Length: {y[i].shape[0]}")
            else:
                pass
                # print(f"({i}, {j}) AUC-PR: {auc_pr:.5f}, FF AUC-PR: {ff_auc_pr:.5f}, Diff.: {abs(auc_pr - ff_auc_pr)}, AUC Slow down: {(ff_vus_time / auc_pr_time):.5f}, Length: {y[i].shape[0]}")
            results.append(curr_result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Run experiments on the current implementation of the VUS metric'
    )
    parser.add_argument('-t', '--testing', type=bool, help='run in testing mode (limits the data for fast testing)', default=False)
    
    args = parser.parse_args()

    main(testing=args.testing)