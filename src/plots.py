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
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['figure.dpi'] = 200

def plot_reliability_diagram_cdf(
        baseline_confidences: Dict[str, List[float]],
        baseline_correctness: Dict[str, List[int]],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        success_percentages: Optional[Dict[str, float]] = None,
):
    """
    Plot all baselines in a single CDF-style reliability figure.
        Args:
        baseline_confidences: Mapping from baseline name to confidence values.
        num_bins: Number of bins for confidence histogram.
        save_path: Optional path to save the plot.
        baseline_correctness: mapping baseline -> correctness array.
        success_percentages: mapping baseline -> success percentage.
    """
    if success_percentages is None:
        success_percentages = {baseline_name: 1.0 for baseline_name in baseline_confidences.keys()}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle="--", alpha=0.5)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fixed color per baseline (sorted names for consistent colors across plots)
    baseline_names_sorted = sorted(baseline_confidences.keys())
    cmap = plt.cm.get_cmap("tab10")
    baseline_colors = {name: cmap(i % cmap.N) for i, name in enumerate(baseline_names_sorted)}

    for baseline_name, confidences in baseline_confidences.items():
        conf_array = np.asarray(confidences, dtype=float)
        correctness = np.asarray(baseline_correctness.get(baseline_name, []), dtype=float)
        n = min(conf_array.size, correctness.size)
        conf_array, correctness = conf_array[:n], correctness[:n]
        # Keep confidence and correctness aligned when dropping NaNs
        mask = ~np.isnan(conf_array) & ~np.isnan(correctness)
        conf_array = np.clip(conf_array[mask], 0.0, 1.0)
        correctness = correctness[mask]

        if conf_array.size == 0:
            continue

        if "gpt-5.2" in save_path or "deepseek" in save_path:
            if baseline_name == "seq_likelihood" or baseline_name == "cot_seq_likelihood":
                continue

        # CDF-style: cumulative fraction of predictions by confidence
        bin_sizes, _ = np.histogram(conf_array, bins=bins)
        cumulative_density = np.cumsum(bin_sizes) / np.sum(bin_sizes)

        success_percentage = success_percentages[baseline_name]
        baseline_color = baseline_colors[baseline_name]

        if baseline_name == "seq_likelihood":
            baseline_name = "Few-shot"
        elif baseline_name == "cot_seq_likelihood":
            baseline_name = "Few-Shot CoT"
        elif baseline_name == "verbalized_cot_qual":
            baseline_name = "Verbalized CoT"
        elif baseline_name == "verbalized_qual":
            baseline_name = "Verbalized"
        elif baseline_name == "ps_seq_likelihood":
            baseline_name = "Platt Scaling"
        elif baseline_name == "ts_seq_likelihood":
            baseline_name = "Temp. Scaling"

        if correctness.size > 0:
            accuracy = np.mean(correctness)
            if "verbalized" in baseline_name:
                label = f"{baseline_name} (acc={accuracy * 100:.1f}%) (success={success_percentage * 100:.1f}%)"
            else:
                label = f"{baseline_name} (acc={accuracy * 100:.1f}%)"
        else:
            label = baseline_name

        ax.plot(
            bin_centers,
            cumulative_density,
            linewidth=2,
            label=label,
            color=baseline_color,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        alpha=0.7,
        linestyle="-.",
        label="Perfect Calibration",
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence", fontsize=18, weight='bold', alpha=0.8)
    ax.set_ylabel("Cumulative Fraction", fontsize=18, weight='bold', alpha=0.8)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15, loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()

def plot_reliability_diagram_cumulative_accuracy(
        baseline_confidences: Dict[str, List[float]],
        baseline_correctness: Dict[str, List[int]],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        success_percentages: Optional[Dict[str, float]] = None,
):
    """
    Plot all baselines in a single cumulative accuracy figure.

    Args:
        baseline_confidences: Mapping from baseline name to confidence values.
        num_bins: Number of bins for confidence histogram.
        save_path: Optional path to save the plot.
        baseline_correctness: mapping baseline -> correctness array.
        success_percentages: mapping baseline -> success percentage.
    """
    if success_percentages is None:
        success_percentages = {baseline_name: 1.0 for baseline_name in baseline_confidences.keys()}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle="--", alpha=0.5)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fixed color per baseline (sorted names for consistent colors across plots)
    baseline_names_sorted = sorted(baseline_confidences.keys())
    cmap = plt.cm.get_cmap("tab10")
    baseline_colors = {name: cmap(i % cmap.N) for i, name in enumerate(baseline_names_sorted)}

    for baseline_name, confidences in baseline_confidences.items():
        conf_array = np.asarray(confidences, dtype=float)
        correctness = np.asarray(baseline_correctness.get(baseline_name, []), dtype=float)
        n = min(conf_array.size, correctness.size)
        conf_array, correctness = conf_array[:n], correctness[:n]
        # Keep confidence and correctness aligned when dropping NaNs
        mask = ~np.isnan(conf_array) & ~np.isnan(correctness)
        conf_array = np.clip(conf_array[mask], 0.0, 1.0)
        correctness = correctness[mask]

        if conf_array.size == 0:
            continue

        if "gpt-5.2" in save_path or "deepseek" in save_path:
            if baseline_name == "seq_likelihood" or baseline_name == "cot_seq_likelihood":
                continue

        bin_sizes, _ = np.histogram(conf_array, bins=bins)
        bin_indices = np.digitize(conf_array, bins)  # 1-based bin index per prediction
        bin_correct = np.array(
            [np.sum(correctness[bin_indices == b]) for b in range(1, num_bins + 1)]
        )
        cumulative_correct = np.cumsum(bin_correct)
        cumulative_total = np.cumsum(bin_sizes)
        cumulative_accuracy = np.where(
            cumulative_total > 0, cumulative_correct / cumulative_total, np.nan
        )

        success_percentage = success_percentages[baseline_name]
        baseline_color = baseline_colors[baseline_name]

        if baseline_name == "seq_likelihood":
            baseline_name = "Few-shot"
        elif baseline_name == "cot_seq_likelihood":
            baseline_name = "Few-Shot CoT"
        elif baseline_name == "verbalized_cot_qual":
            baseline_name = "Verbalized CoT"
        elif baseline_name == "verbalized_qual":
            baseline_name = "Verbalized"
        elif baseline_name == "ps_seq_likelihood":
            baseline_name = "Platt Scaling"
        elif baseline_name == "ts_seq_likelihood":
            baseline_name = "Temp. Scaling"

        if correctness.size > 0:
            accuracy = np.mean(correctness)
            if "verbalized" in baseline_name:
                label = f"{baseline_name} (acc={accuracy * 100:.1f}%) (success={success_percentage * 100:.1f}%)"
            else:
                label = f"{baseline_name} (acc={accuracy * 100:.1f}%)"
        else:
            label = baseline_name

        ax.plot(
            bin_centers,
            cumulative_accuracy,
            linewidth=2,
            label=label,
            color=baseline_color,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        alpha=0.7,
        linestyle="-.",
        label="Perfect Calibration",
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence", fontsize=18, weight='bold', alpha=0.8)
    ax.set_ylabel("Cumulative Accuracy", fontsize=18, weight='bold', alpha=0.8)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16, loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()


def plot_reliability_diagram(
        all_confidences: List[float],
        all_correctness: List[int],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        display_percentages: bool = True,
        success_percentage: float = 1.0,
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

        # do not show success rate if percentage is 100
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

def plot_multiple_reliability_diagrams(
        all_confidences_list: List[List[float]],
        all_correctness_list: List[List[int]],
        labels: List[str],
        num_bins: int = 10,
        save_path: Optional[str] = None,
        display_percentages: bool = True,
        success_percentages: Optional[List[float]] = None,
):
    """
    Plot multiple reliability diagrams side by side in a single figure.
    
    Args:
        all_confidences_list: List of confidence scores for each diagram
        all_correctness_list: List of correctness values for each diagram
        labels: Labels for each diagram
        num_bins: Number of bins for the histograms
        save_path: Path to save the combined plot
        display_percentages: Whether to display percentage values on bars
        success_percentages: List of success percentages for each diagram
    """
    if success_percentages is None:
        success_percentages = [1.0] * len(all_confidences_list)
    
    n_plots = len(all_confidences_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), dpi=200)
    if n_plots == 1:
        axes = [axes]
    
    sns.set_theme(style="whitegrid")
    
    for idx, (confidences, correctness, label, success_percentage) in enumerate(zip(
        all_confidences_list, all_correctness_list, labels, success_percentages
    )):
        ax = axes[idx]
        bins = np.arange(0.0, 1.0, 1.0 / num_bins)
        bins_per_prediction = np.digitize(confidences, bins)
        df = pd.DataFrame({
            "y_pred": confidences,
            "y": correctness,
            "pred_bins": bins_per_prediction,
        })

        grouped_by_bins = df.groupby('pred_bins')
        grouped_bins = grouped_by_bins.mean()
        grouped_bins = grouped_bins["y"].reindex(range(1, num_bins + 1), fill_value=0)
        bin_values = grouped_bins.values
        bin_sizes = grouped_by_bins["y"].count()
        bin_sizes = bin_sizes.reindex(range(1, num_bins + 1), fill_value=0)
        
        ax.grid(visible=True, axis="both", which="major", linestyle="--", alpha=0.5)
        step_size = 1.0 / num_bins
        bar_colors = []
        
        cmap = matplotlib.cm.get_cmap('cividis')

        if display_percentages:
            total = sum(bin_sizes.values)
            for bin_size in bin_sizes.values:
                bin_percentage = bin_size / total * success_percentage
                bar_colors.append(cmap(min(0.9999, bin_percentage + 0.2)))

        ax.bar(
            bins + step_size / 2,
            bin_values,
            width=step_size * 0.8,
            alpha=0.85,
            color=bar_colors,
            edgecolor='black',
            linewidth=0.5,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="black",
            alpha=0.7,
            linestyle='-.',
            label="Perfect Calibration"
        )

        if display_percentages:
            total = sum(bin_sizes.values)
            eps = 0.02

            for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
                bin_percentage = round(bin_size / total * success_percentage * 100, 1)
                
                if bin_size == 0 or bin_values[i] < 0.2:
                    continue
                
                bar_height = bin_values[i] 
                y_position = min(bar_height + eps, 0.95)
                text_color = "black"
                
                ax.text(
                    x=bin + step_size / 2,
                    y=y_position,
                    s=f"{bin_percentage}%",
                    ha="center",
                    va="bottom" if bar_height <= 0.9 else "top",
                    color=text_color,
                    alpha=0.6 if bin_percentage < 40 else 0.8,
                    fontsize=12,
                )

            if success_percentage < 1:
                ax.text(
                    x=-0.15, 
                    y=-0.15,
                    s=f"Success: {round(success_percentage * 100, 2)}%",
                    color="gray",
                    fontsize=12,
                    weight="bold",
                    transform=ax.transAxes,
                )
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Confidence", fontsize=16, weight='bold', alpha=0.8)
        if idx == 0:  # Only show y-label for the first plot
            ax.set_ylabel("Accuracy", fontsize=16, weight='bold', alpha=0.8)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, loc="upper left")
        ax.set_title(label, fontsize=16, pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()

