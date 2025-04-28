import torch
import numpy as np
import pandas as pd

def eceloss(softmaxes, labels, n_bins=10):
    """
    Modified from:
        https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    # already sorted thanks to linspace fnc
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    print("Predictions:", predictions)
    accuracies = predictions.eq(labels)
    print("Accuracies:", accuracies)
    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    ece = torch.zeros(1, device=d)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # element-wise multiplication
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)

    return ece, acc_in_bin, avg_conf_in_bin


def eceloss_adapted(softmaxes, labels, n_bins=10):
    """
    Modified from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    # already sorted thanks to linspace fnc
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = softmaxes
    accuracies = labels
    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # element-wise multiplication
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    acc_in_bin = torch.tensor(accuracy_in_bin_list)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list)

    return ece

def ece(y_true: np.array, y_pred: np.array, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error: for each bin, the absolute difference between
    the mean fraction of positives and the average predicted probability is taken. The ECE is
    the weighed mean of these differences.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.
    Returns
    -------
    ece: float
        The expected calibration error.
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({"y_pred": y_pred, "y": y_true, "pred_bins": bins_per_prediction})

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins["y"].count()

    # calculate the proportion of data per bin
    binned["weight"] = binned_counts / n

    weighed_diff = abs(binned["y_pred"] - binned["y"]) * binned["weight"]
    return weighed_diff.sum()
