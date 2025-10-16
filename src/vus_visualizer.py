"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from utils.dataloader import Dataloader
from utils.scoreloader import Scoreloader
from utils.utils import load_tsb


def main(plot):
    if plot == 'existence_seq':
        visualize_existence_examples()
    elif plot == 'conf_matrix':
        confusion_matrix_visualization()
    else:
        raise ValueError('Unknown plot type')
    
def confusion_matrix_visualization():
    sns.set_style("whitegrid")
    ffvus = VUSNumpy(50)

    label = np.array([0]*100 + [1]*200 + [0]*400).astype(np.float64)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    T = label.shape[0]
    
    score = np.zeros_like(label, dtype=float)
    score[200:400] = 0.9
    # score += np.random.uniform(0, .2, score.shape[0])
    thresholds = np.linspace(1, 0, 11).round(2)
    sm, _ = ffvus.get_score_mask(score, thresholds)
    t = 0.6

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 3.5))

    ax[0].plot(labels[-1], color='k', linewidth=2.5)
    ax[0].fill_between(range(T), 0, labels[-1], where=label == 1, color="blue", alpha=0.4, label="Original anomaly")
    ax[0].fill_between(range(T), 0, labels[-1], where=np.logical_and(labels[-1] < 1, labels[-1] > 0), color="red", alpha=0.3, label="Buffer regions")
    ax[0].set_title('Label with buffer')
    ax[0].legend()

    ax[1].plot(score, color='k', linewidth=2.5)
    ax[1].fill_between(range(T), t, score, where=np.logical_and(np.logical_and(score > t, labels[-1]), label == 1), color="green", alpha=0.4, label="TP -> dot product")
    ax[1].fill_between(range(T), t, score, where=np.logical_and(np.logical_and(score > t, labels[-1]), labels[-1] < 1), color="orange", alpha=0.4, label="TP -> matrix mul.")
    ax[1].set_ylim(0, 1)
    ax[1].hlines(y=0.6, xmin=0, xmax=T, linestyle='--', color='gray', label='threshold')
    ax[1].set_xlabel('Time')
    ax[1].set_title('Score')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("experiments/figures/tp_buffer_masking.svg", bbox_inches="tight")
    plt.savefig("experiments/figures/tp_buffer_masking.pdf", bbox_inches="tight")

    plt.show()

    # ffvus.conf_matrix_mode = 'dynamic_plus'
    # (fp, fn, tp, p, n, fpr), _ = ffvus.compute_confusion_matrix(labels, sm)
    # print(fn.shape, tp.shape, p.shape)


def visualize_existence_examples():
    thresholds = np.linspace(1, 0, 11).round(2) 
    ffvus = VUSNumpy(10)
    
    label = np.array([0]*20 + [1]*20 + [0]*20)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    
    score = np.zeros_like(label, dtype=float)
    score[30] = 0.6
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_1", "Detected inside anomaly")

    score = np.zeros_like(label, dtype=float) # np.random.uniform(0, 0.05, label.shape[0])
    score[15] = 0.6
    score[30] = 0.4
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_2", "Left buffer + middle")

    score = np.zeros_like(label, dtype=float)
    score[13] = 0.6
    score[17] = 0.4
    score[30] = 0.2
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_3", "Two in left buffer + inside anomaly")


def plot_existence_toy(labels, score, existence, thresholds, filename, title_suffix):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(6, 2.5))
    gs = GridSpec(2, 5, figure=fig, hspace=.6, wspace=.9)

    # Buffered label
    ax0 = fig.add_subplot(gs[0, :3])
    sns.lineplot(labels.T, ax=ax0, palette='flare_r', legend=False, linewidth=1.5)
    ax0.set_title(f"Label with buffers") #  ({title_suffix})

    # Score
    ax1 = fig.add_subplot(gs[1, :3], sharex=ax0)
    ax1.plot(score, markersize=3, color='black', linewidth=2.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.linspace(0, 1, 6))
    ax1.set_title("Score")
    ax1.set_xlabel("Time")

    # Existence heatmap
    ax2 = fig.add_subplot(gs[:, 3:])
    sns.heatmap(existence, cmap="flare_r", rasterized=True, ax=ax2)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Buffer")
    # 
    # ax2.set_xticklabels(thresholds)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    ax2.set_xticks(np.arange(0, 11, 2))
    ax2.set_xticklabels(['1', '.8', '.6', '.4', '.2', '0'])
    ax2.invert_yaxis()

    # plt.tight_layout()
    plt.savefig(f"experiments/figures/{filename}.svg", bbox_inches='tight')
    plt.savefig(f"experiments/figures/{filename}.pdf", bbox_inches='tight')
    # plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Visualize different useful concepts of the ffvus metric'
    )
    
    parser.add_argument('--plot', type=str, choices=['existence_seq', 'conf_matrix'], default='existence_seq', help='Type of plot to generate')
    args = parser.parse_args()

    main(plot=args.plot)