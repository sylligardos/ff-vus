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
import os

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
    elif plot == 'steps':
        steps_visualization()
    elif plot == 'global_mask':
        global_mask_visualization()
    else:
        raise ValueError('Unknown plot type')


def global_mask_visualization():
    sns.set_style("whitegrid")
    buffer_size = 20
    ffvus = VUSNumpy(buffer_size)
    global_mask_choices = [False, True]

    dataloader = Dataloader(os.path.join('data', 'raw'))
    x, y, fn = dataloader.load_raw_datasets(datasets=['MGAB']) # dataloader.get_dataset_names()
    ts_index = -1
    max_len = -1
    for i, ts in enumerate(x):
        if len(ts) > max_len:
            ts_index = i
            max_len = len(ts)
    label = y[ts_index]
    print(f"Time series '{fn[ts_index]}', Length: {len(label)}")
    # label = torch.tensor([0]*150 + [1]*20 + [0]*200 + [1]*50 + [0]*250 + [1]*20 + [0]*200, dtype=torch.float32)
    
    fig, axes = plt.subplots(2, 1, figsize=(4, 2.5), sharex=False)
    for i, global_mask in enumerate(global_mask_choices):
        ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), _ = ffvus.get_anomalies_coordinates(label)
        if global_mask:
            safe_mask, _ = ffvus.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=global_mask)
            label = label[safe_mask]
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), _ = ffvus.get_anomalies_coordinates(label)
        else:
            pass
        labels = ffvus.add_slopes_precomputed(label, start_no_edges, end_no_edges)
        
        sns.lineplot(labels.T, ax=axes[i], palette='flare_r', legend=False, linewidth=1.5)
        axes[i].set_title(f'Global mask: {global_mask}, Length: {len(label)}')
        axes[i].set_yticks([])
        print(f'Size reduction {(max_len / len(label)):.0f}x')

    plt.tight_layout()
    plt.savefig("experiments/figures/global_mask.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/global_mask.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def steps_visualization():
    sns.set_style("whitegrid")
    
    step_sizes = [1, 5, 10]

    label = np.array([0]*40 + [1]*20 + [0]*40).astype(np.float64)    

    fig, axes = plt.subplots(len(step_sizes), 1, figsize=(4, 2.5), sharex=True)
    for i, step in enumerate(step_sizes):
        buffer_size = 20
        ffvus = VUSNumpy(
            slope_size=buffer_size,
            step=step
        )
        ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
        labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    
        sns.lineplot(labels.T, ax=axes[i], palette='flare_r', legend=False, linewidth=1.5)
        axes[i].annotate(f'Step {step}', xy=(5, .5), xytext=(.5, .5))
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig("experiments/figures/buffer_steps.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/buffer_steps.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def gpu_existence_visualization():
    sns.set_style("whitegrid")
    device = 'cpu'
    buffer_size = 10
    ffvus = VUSTorch(buffer_size)

    label = torch.tensor([0]*15 + [1]*20 + [0]*40 + [1]*10 + [0]*25 + [1]*1 + [0]*20, dtype=torch.float32)
    score = torch.tensor(np.zeros_like(label, dtype=float))
    score[10] = 0.5
    score[28] = 0.8
    score[55] = 0.6
    # score[63] = 0.6
    # score[71] = 0.9
    score[90] = 0.4
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)
    thresholds, _ = ffvus.get_unique_thresholds(score)
    score_mask, _ = ffvus.get_score_mask(score, thresholds)

    # Compute existence
    # We myltiply every label in labels with every score of all thresholds (both normalized; since we are only counting anomalies found we don't care about the value of the buffer)
    # This means that only the regions where label and score are both 1 will survive in the output
    # Since the label is encoded with the staircase, the output also contains the info of which anomaly was found (e.g. value is 4 for the 4th anonaly found)
    # The cumulative max carries this info accross all time steps after every anomaly found and not only at the point where it was found
    norm_labels = (labels > 0).int()
    diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
    diff = torch.clamp(diff, min=0, max=1)
    stairs = torch.cumsum(diff, dim=1)
    labels_stairs = norm_labels * stairs
    score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]  # Multiply every score mask with every staircased buffered label, shape (L, t, T)
    cm = torch.cummax(score_hat, dim=2).values                      # Now every anomaly that is found, encodes the number of the anomaly, shape (L, t, T)
    cm_diff = torch.diff(cm, dim=2)                                 
    cm_diff_norm = torch.clamp(cm_diff - 1, min=0)                  

    total_anomalies = stairs[:, -1][:, None]                                            # Tells the total number of anomalies
    final_anomalies_missed = total_anomalies - cm[:, :, -1]                             # CM dif can't show the last anomaly(-ies) that were not found, since it's a dif operation
    n_anomalies_not_found = torch.sum(cm_diff_norm, dim=2) + final_anomalies_missed     # The normalized cm dif shows the number of anomalies not found, if every anomaly is found then cm_diff is always 1, so cm_diff_norm is always 0
    n_anomalies_found = total_anomalies - n_anomalies_not_found                         
    
    # --- Visualization ---
    fig, axes = plt.subplots(5, 1, figsize=(5, 6), sharex=True)
    
    # (a) Label and buffered label
    axes[0].plot(labels[-1], color="black", lw=2.5, label=f"Buffered label (L={buffer_size})")
    axes[0].plot(label, '--', color="royalblue", lw=2.5, label="Original label")
    axes[0].annotate('Single buffer\n     shown', xy=(50, 1), xytext=(40.5, .72))
    axes[0].set_title("(a) Original (dashed) and buffered label")
    axes[0].legend(loc="upper right").remove()

    # (b) Staircase encoding
    axes[1].plot(labels_stairs[-1], color="purple", lw=2.5)
    axes[1].set_title("(b) Staircase encoding of buffered labels")
    
    # (c) Scores and score mask
    axes[2].plot(score_mask[-2], color="green", lw=2.5, label=f"Score mask (thr={thresholds[-2]:.2f})")
    axes[2].plot(score, '--', color="orange", lw=2.5, label="Score")
    axes[2].annotate('Single threshold\n        shown', xy=(50, 1), xytext=(98, .55))
    axes[2].set_title("(c) Score (dashed) and score mask")
    axes[2].legend(loc="upper right").remove()

    # (d) Score hat
    axes[3].plot(score_hat[-1, -2, :], color="seagreen", lw=2.5)
    axes[3].set_title("(d) $\\overline{Score}$ (score mask$\\bullet$staircase label)")
    
    # (e) Cumulative max
    axes[4].plot(cm[-1, -2, :], color="dodgerblue", lw=2.5)
    axes[4].set_title("(e) Cumulative Max of $\\overline{Score}$")
    axes[4].hlines(y=[1, 3], xmin=0, xmax=label.shape[0], linestyle='--', color='k', lw=1.5)
    axes[4].annotate('Step difference is 2,\n 1 anomaly missed', xy=(40, 1), xytext=(42, 1.6))
    axes[4].annotate('', xy=(40, 1), xytext=(40, 3), arrowprops=dict(arrowstyle='<->', linestyle='--', color='k', lw=1.5))
    axes[4].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig("experiments/figures/gpu_existence.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/gpu_existence.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    
def gpu_buffers_visualization():
    sns.set_style("whitegrid")
    buffer_size = 100
    ffvus = VUSTorch(buffer_size, step=10)

    label = torch.tensor([0]*150 + [1]*200 + [0]*250 + [1]*100 + [0]*100 + [1]*10 + [0]*200, dtype=torch.float32)
    T = label.shape[0]
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 2.5))

    ax[0].plot(label, color='k', linewidth=2.5)
    # ax[0].set_title('Label')
    ax[1].plot(pos, color='k', linewidth=2.5)
    # ax[1].set_title('Position sequence')
    ax[1].fill_between(range(T), 0, 200, where=pos >= buffer_size, color="pink", alpha=0.8, label="Not affected regions")
    ax[1].fill_between(range(T), 0, 200, where=label, color="pink", alpha=0.8, label="Original anomaly")
    ax[1].fill_between(range(T), 0, 200, where=np.logical_and(pos < buffer_size, np.logical_not(label)), color="lightgreen", alpha=0.8, label="Buffers")
    sns.lineplot(labels.T, ax=ax[2], palette='flare_r', legend=False, linewidth=1.5)
    # ax[2].set_title('Label with 10 buffers')

    # tmp = 1 - (((1 - ffvus.zita) * pos) / buffer_size)
    # tmp[label == 1] = 1
    # tmp[pos > buffer_size] = 0
    # ax[3].plot(tmp)
    
    plt.tight_layout()
    plt.savefig("experiments/figures/gpu_buffers.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/gpu_buffers.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

def confusion_matrix_visualization():
    sns.set_style("whitegrid")
    buffer_length = 50
    ffvus = VUSNumpy(buffer_length)

    label = np.array([0]*100 + [1]*200 + [0]*400).astype(np.float64)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    T = label.shape[0]
    
    score = np.zeros_like(label, dtype=float)
    score += np.random.uniform(0, .1, score.shape[0])
    score[200:400] += .7
    thresholds = np.linspace(1, 0, 11).round(2)
    sm, _ = ffvus.get_score_mask(score, thresholds)
    t = 0.6

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4.5, 2.5))

    ax[0].plot(labels[-1], color='k', linewidth=2.5)
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 0, color="lightskyblue", alpha=0.5, label="Non-anomalous")
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 1, color="pink", alpha=0.5, label="Original anomaly")
    ax[0].fill_between(range(T), 0, 1, where=np.logical_and(labels[-1] < 1, labels[-1] > 0), color="purple", alpha=0.5, label="Buffer region")
    # ax[0].set_title(f'Label with buffer {buffer_length}')
    ax[0].set_ylabel('Label')
    ax[0].set_yticks([])
    ax[0].legend()

    ax[1].plot(score, color='k', linewidth=2.5)
    ax[1].hlines(y=t, xmin=0, xmax=len(label), lw=2.5, color='red', linestyle='--', label='threshold')
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), label == 1), color="lightgreen", alpha=0.5, label="$TP_{initial}$")
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), labels[-1] < 1), color="darkgreen", alpha=0.5, label="$TP_{buffer}$")
    # ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 1, labels[-1]  == 0), color="red", alpha=0.5, label="FP; O(t, T)")
    # ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 0, label == 1), color="orange", alpha=0.5, label="FN; O(t, T)")
    ax[1].set_xlabel('Time')
    # ax[1].set_xticks([])
    # ax[1].set_title('Score of threshold t\'')
    ax[1].set_ylabel('Score')
    ax[1].set_yticks([])
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("experiments/figures/conf_matrix_mask.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/conf_matrix_mask.pdf", bbox_inches="tight", pad_inches=0)
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

    # plt.tight_layout()global_mask
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
        choices=['existence_seq', 'conf_matrix', 'gpu_buffers', 'gpu_existence', 'steps', 'global_mask'], 
        default='existence_seq', 
        help='Type of plot to generate'
    )
    args = parser.parse_args()

    main(plot=args.plot)