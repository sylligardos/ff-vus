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
import torch

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
    elif plot == 'gpu_buffers':
        gpu_buffers_visualization()
    elif plot == 'gpu_existence':
        gpu_existence_visualization()
    else:
        raise ValueError('Unknown plot type')
    
def gpu_existence_visualization():
    sns.set_style("whitegrid")
    device = 'cpu'
    buffer_size = 10
    ffvus = VUSTorch(buffer_size)

    label = torch.tensor([0]*15 + [1]*20 + [0]*25 + [1]*10 + [0]*5 + [1]*1 + [0]*20, dtype=torch.float32)
    
    score = torch.tensor(np.zeros_like(label, dtype=float))
    # First anomaly (15–35): one detection inside (true positive) and one in left buffer
    score[10] = 0.5   # left buffer
    score[28] = 0.8   # inside anomaly
    # Second anomaly (60–70): one detection in right buffer (missed in anomaly)
    score[74] = 0.6   # right buffer
    # Third anomaly (75–80): perfect detection inside
    score[77] = 0.9
    # Fourth anomaly (85): one detection close but outside
    score[83] = 0.4   # left buffer (missed anomaly)
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)
    thresholds, _ = ffvus.get_unique_thresholds(score)
    score_mask, _ = ffvus.get_score_mask(score, thresholds)

    # Compute existence
    norm_labels = (labels > 0).int()

    diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
    diff = torch.clamp(diff, min=0, max=1)
    stairs = torch.cumsum(diff, dim=1)
    labels_stairs = norm_labels * stairs

    score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]

    cm = torch.cummax(score_hat, dim=2).values  # shape: [B, S, T']

    cm_diff = torch.diff(cm, dim=2)
    cm_diff_norm = torch.clamp(cm_diff - 1, min=0)

    total_anomalies = stairs[:, -1][:, None]
    final_anomalies_missed = total_anomalies - cm[:, :, -1]
    n_anomalies_not_found = torch.sum(cm_diff_norm, dim=2) + final_anomalies_missed
    n_anomalies_found = total_anomalies - n_anomalies_not_found
    existence = n_anomalies_found / total_anomalies

    # Visualization
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    palette = 'flare_r'

    sns.lineplot(labels.T, palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("(a) Label with buffers normalized")

    sns.lineplot(score, palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("(b) Score")

    # TODO: Why is the staircase of the third anomaly weird ?
    sns.lineplot(labels_stairs.T, palette=palette, ax=axes[2], legend=False)
    axes[2].set_title("(c) Staircase encoding of anomalies")

    # sns.lineplot(score_mask.T, palette="Greens", ax=axes[2])
    # axes[2].set_title("(c) Score mask for different thresholds")

    # sns.lineplot(cm[0].T, palette="Oranges", ax=axes[3])
    # axes[3].set_title("(d) Cumulative detection propagation")

    # sns.lineplot(existence.T, palette="Reds", ax=axes[4])
    axes[4].set_title("(e) Final existence surface")
    
    plt.xlabel("Time")
    plt.tight_layout()
    # plt.savefig("figures/gpu_existence.pdf", bbox_inches="tight")
    plt.show()
    
def gpu_buffers_visualization():
    sns.set_style("whitegrid")
    buffer_size = 10
    ffvus = VUSTorch(buffer_size)

    label = torch.tensor([0]*15 + [1]*20 + [0]*25 + [1]*10 + [0]*5 + [1]*1 + [0]*20, dtype=torch.float32)
    T = label.shape[0]
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 3.5))

    ax[0].plot(label, color='k', linewidth=2.5)
    ax[0].set_title('Label')
    ax[1].plot(pos, color='k', linewidth=2.5)
    ax[1].set_title('Position sequence')
    ax[1].fill_between(range(T), 0, 20, where=pos > buffer_size, color="lightskyblue", alpha=0.8, label="Not affected regions")
    ax[1].fill_between(range(T), 0, 20, where=label, color="pink", alpha=0.8, label="Original anomaly")
    sns.lineplot(labels.T, ax=ax[2], palette='flare_r', legend=False, linewidth=1.5)
    ax[2].set_title('Label with 10 buffers')

    # tmp = 1 - (((1 - ffvus.zita) * pos) / buffer_size)
    # tmp[label == 1] = 1
    # tmp[pos > buffer_size] = 0
    # ax[3].plot(tmp)
    
    plt.tight_layout()
    plt.savefig("experiments/figures/gpu_buffers.svg", bbox_inches="tight")
    plt.savefig("experiments/figures/gpu_buffers.pdf", bbox_inches="tight")
    plt.show()

def confusion_matrix_visualization():
    sns.set_style("whitegrid")
    ffvus = VUSNumpy(50)

    label = np.array([0]*100 + [1]*200 + [0]*400).astype(np.float64)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    T = label.shape[0]
    
    score = np.zeros_like(label, dtype=float)
    score[200:400] = 1
    # score += np.random.uniform(0, .2, score.shape[0])
    thresholds = np.linspace(1, 0, 11).round(2)
    sm, _ = ffvus.get_score_mask(score, thresholds)
    t = 0.6

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 3.5))

    ax[0].plot(labels[-1], color='k', linewidth=2.5)
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 0, color="lightskyblue", alpha=0.5, label="Non-anomalous region")
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 1, color="pink", alpha=0.5, label="Original anomaly")
    ax[0].fill_between(range(T), 0, 1, where=np.logical_and(labels[-1] < 1, labels[-1] > 0), color="purple", alpha=0.5, label="Buffer region")
    ax[0].set_title('Label with buffer L\'')
    ax[0].legend()

    ax[1].plot(score, color='k', linewidth=2.5)
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), label == 1), color="lightgreen", alpha=0.5, label="TP; O(t, T)")
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), labels[-1] < 1), color="darkgreen", alpha=0.5, label="TP; O(t, L, T)")
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 1, labels[-1]  == 0), color="red", alpha=0.5, label="FP; O(t, T)")
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 0, label == 1), color="orange", alpha=0.5, label="FN; O(t, T)")
    ax[1].set_xlabel('Time')
    ax[1].set_title('Score of threshold t\'')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("experiments/figures/conf_matrix_mask.svg", bbox_inches="tight")
    plt.savefig("experiments/figures/conf_matrix_mask.pdf", bbox_inches="tight")
    plt.show()


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
    
    parser.add_argument(
        '--plot', 
        type=str, 
        choices=['existence_seq', 'conf_matrix', 'gpu_buffers', 'gpu_existence'], 
        default='existence_seq', 
        help='Type of plot to generate'
    )
    args = parser.parse_args()

    main(plot=args.plot)