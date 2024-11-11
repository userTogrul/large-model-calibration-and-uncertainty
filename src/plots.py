import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional,  Dict, List
import pandas as pd
sns.set_theme()
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['font.family'] = 'sans-serif'

"""
    Plots that are taken from Temperature scaling paper.
    TODO:
        Needs modifications.
"""

def plot_reliability_diagram(
        all_confidences: List[float],
        all_correctness: List[int],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        display_percentages: bool = True,
        success_percentage: float = 1,
):
    bins = np.arange(0.0, 1.0, 1.0 / num_bins)
    bins_per_prediction = np.digitize(all_confidences, bins)
    df = pd.DataFrame(
        {
            "y_pred": all_confidences,
            "y": all_correctness,
            "pred_bins": bins_per_prediction,
        }
    )

    grouped_by_bins = df.groupby('pred_bins')
    grouped_bins = grouped_by_bins.mean()
    grouped_bins = grouped_bins["y"].reindex(range(1, num_bins + 1), fill_value=0)
    bin_values = grouped_bins.values
    # calculate the number of items per bi
    bin_sizes = grouped_by_bins["y"].count()
    bin_sizes = bin_sizes.reindex(range(1, num_bins + 1), fill_value=0)
    # sns.set_style("ticks") # change the style to white grid
    sns.set_theme()
    sns.set_context("paper")
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(5, 5), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle="--")
    step_size = 1.0/num_bins
    bar_colors=[]

    if display_percentages:
        total = sum(bin_sizes.values)
        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = bin_size / total * success_percentage
            cmap = matplotlib.cm.get_cmap('Oranges')
            bar_colors.append(cmap(min(0.9999, bin_percentage + 0.2)))

    plt.bar(
        bins + step_size / 2,
        bin_values,
        width=0.1,
        alpha=0.75,
        color=bar_colors,
        edgecolor='white',
    )
    plt.plot(
        np.arange(0, 1 + 0.05, 0.05),
        np.arange(0, 1 + 0.05, 0.05),
        color="black",
        alpha=0.2,
        linestyle='-.'
    )

    # Now add the percentage value of points per bin as text
    if display_percentages:
        total = sum(bin_sizes.values)
        eps = 0.01

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = round(bin_size / total * success_percentage, 2)
            
            # omit labeling for small bars
            if bin_size == 0 or bin_values[i] < 0.2:
                continue   

            plt.annotate(
                f"{bin_percentage} %",
                xy=(bin + step_size /2, bin_values[i] - eps),
                ha="center",
                va="top",
                rotation=90,
                color="white" if bin_percentage > 40 else "black",
                alpha=0.7 if bin_percentage > 40 else 0.8,
                fontsize=14,
            )

        if success_percentage < 1:
            plt.annotate(
                f"Success: {round(success_percentage * 100, 2)} %",
                xy=(-0.19, -0.135),
                color="pink",
                fontsize=14,
                alpha=0.7,
                annotation_clip=False,
                bbox=dict(facecolor='none', edgecolor="orange", pad=4.0, alpha=0.7),
            )
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence", fontsize=18, weight='bold', alpha=0.75)
    plt.ylabel("Accuracy", fontsize=18, weight='bold', alpha=0.75)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()

def plot_conf(acc, conf):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    ax.plot([0,1], [0,1], 'k--')
    ax.plot(conf.data.cpu().numpy(), acc.data.cpu().numpy(), marker='.')
    ax.set_xlabel(r'confidence')
    ax.set_ylabel(r'accuracy')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax

def plot_uncert(err, entr):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    
    ax.plot([0,1], [0,1], 'k--') 
    ax.plot(entr.data.cpu().numpy(), err.data.cpu().numpy(), marker='.')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_ylabel(r'error')
    ax.set_xlabel(r'uncertainty')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax

