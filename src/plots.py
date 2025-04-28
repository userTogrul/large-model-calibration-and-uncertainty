import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional,  Dict, List
import pandas as pd
sns.set_theme()
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'sans-serif'

def plof_reliability_diagram_cdf(
        all_confidences: List[float],
        all_correctness: List[int],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        success_percentage: float = 1,
):
    # TODO: Create reldiag with cumulative distribution function (CDF)
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
    bin_sizes = grouped_by_bins["y"].count()
    bin_sizes = bin_sizes.reindex(range(1, num_bins + 1), fill_value=0)

    cumulative_accuracy = np.cumsum(bin_sizes) / sum(bin_sizes) 

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle="--", alpha=0.5)
    step_size = 1.0 / num_bins

    # plot CDF of predictions
    plt.plot(
        bins + step_size / 2, 
        cumulative_accuracy, 
        color='blue', 
        linewidth=2,
        label="Cumulative Accuracy"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        alpha=0.7,
        linestyle='-.',
        label="Perfect Calibration"
    )
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence", fontsize=18, weight='bold', alpha=0.8)
    plt.ylabel("Cumulative Accuracy", fontsize=18, weight='bold', alpha=0.8)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="lower left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()


def plot_reliability_diagram(
        all_confidences: List[float],
        all_correctness: List[int],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        display_percentages: bool = True,
        success_percentage: float = 1,
):
    """
        Loosly modified from:
            https://github.com/parameterlab/apricot/blob/main/src/plotting.py 
    """
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
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle="--", alpha=0.5)
    step_size = 1.0 / num_bins
    bar_colors=[]
    
    cmap = matplotlib.cm.get_cmap('cividis')

    if display_percentages:
        total = sum(bin_sizes.values)
        for i, bin_size in enumerate(bin_sizes.values):
            bin_percentage = bin_size / total * success_percentage
            bar_colors.append(cmap(min(0.9999, bin_percentage + 0.2)))
            # bar_colors.append(cmap(bin_percentage if bin_percentage < 1 else 0.99))

    plt.bar(
        bins + step_size / 2,
        bin_values,
        width=step_size * 0.8,
        alpha=0.85,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5,
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        alpha=0.7,
        linestyle='-.',
        label="Perfect Calibration"
    )

    # Now add the percentage value of points per bin as text
    if display_percentages:
        total = sum(bin_sizes.values)
        eps = 0.02

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = round(bin_size / total * success_percentage * 100, 1)
            
            # omit labeling for small bars
            if bin_size == 0 or bin_values[i] < 0.2:
                continue
            
            bar_height = bin_values[i] 
            y_position = min(bar_height + eps, 0.95)
            text_color = "black"
            
            plt.text(
                x = bin + step_size / 2,
                y = y_position,
                s = f"{bin_percentage}%",
                ha="center",
                va="bottom" if bar_height <= 0.9 else "top",
                color=text_color,
                alpha=0.6 if bin_percentage < 40 else 0.8,
                fontsize=16,
            )

        if success_percentage < 1:
            ax.text(
                x=-0.15, 
                y=-0.15,
                s=f"Success: {round(success_percentage * 100, 2)}%",
                color="gray",
                fontsize=16,
                weight="bold",
                # bbox=dict(facecolor='white', edgecolor="gray", boxstyle='round,pad=0.3'),
                transform=ax.transAxes,
            )
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence", fontsize=20, weight='bold', alpha=0.8)
    plt.ylabel("Accuracy", fontsize=20, weight='bold', alpha=0.8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc="upper left")
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
    """
        Loosly modified from:
            https://github.com/mlaves/bayesian-temperature-scaling 
    """
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    
    ax.plot([0,1], [0,1], 'k--') 
    ax.plot(entr.data.cpu().numpy(), err.data.cpu().numpy(), marker='.')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_ylabel(r'error')
    ax.set_xlabel(r'uncertainty')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax

