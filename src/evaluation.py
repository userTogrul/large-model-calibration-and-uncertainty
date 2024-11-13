#standard lib
import re
from typing import Dict, Optional, List, Union, Tuple, Callable
# external lib
import evaluate
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from methods.ece.ECE import eceloss_adapted, ece
from methods.uce import UCE
from sklearn.metrics import brier_score_loss, roc_auc_score # BS and AUROC
from torcheval.metrics.functional import bleu_score # BLEU score

from relplot.metrics import smECE_slow as SmECE # Smooth ECE metric from the paper
from sklearn.isotonic import IsotonicRegression # isotonic regression

def check_answer_correctness(
    correct_answers: List[str],
    model_answers: List[str],
    bleu_threshold: float = 0.3, # threshold for BLEU score
) -> List[bool]:
    global bleu_score

    results = []

    for correct_answer, model_answer in zip(correct_answers, model_answers):
        try:
            if (bleu_score( 
                model_answer,
                [correct_answer], 
                n_gram=1,
                # weights=torch.tensor([0.4, 0.4, 0.1, 0.1]) # more importance for 1 and 2-grams
            ) >= bleu_threshold ) or (correct_answer in model_answer): # use the BLEU score to check if the model answer is correct
                results.append(True)
            else:
                results.append(False)
        except ValueError:
            print("Model answer: ", model_answer)
            print("Correct answer: ", correct_answer)
            results.append(False)
            continue

    return results

def extract_verbalized_confidence(
        expressions: List[str],
        mode: str,
        expression_mapping: Optional[Dict[str, float]] = None,
) -> Tuple[List[bool], List[bool]]:
    assert mode in (
        "qualitative",
        "quantitative",
    ), f"Mode has to be either qualitative or quantitative, but {mode} found."

    if mode == "qualitative":
        assert (
            expression_mapping is not None
        ), "'expression_mapping' has to be specified for qualitative mode."
    
    confidences, successful = [], []

    for expression in expressions:
        if mode == "qualitative":
            template = rf"({'|'.join(expression_mapping.keys())})"
        
        try:
            res = re.search(template, expression).group(0)

            if mode == "qualitative":
                conf = expression_mapping[res]
            
            successful.append(True)
            confidences.append(conf)
        
        except AttributeError:
            successful.append(False)

    return confidences, successful

def evaluate_confidences(
        split_name: str,
        all_confidences: List[float],
        all_correctness: List[int],
        all_targets: Optional[List[float]] = None,
        num_bins: int = 10,
        add_name: Optional[str] = None,
) -> Dict[str, float]:
    if all_targets is None:
        target_func = get_target_function(all_confidences, all_correctness, num_bins)
        all_targets = target_func(all_confidences)

    infix =  ""
    if add_name is not None:
        infix = f"{add_name}_"
        
    metrics = {
        f"{split_name}_{infix}ECE": ece(y_true=all_correctness, y_pred=all_confidences),
        f"{split_name}_{infix}SmoothECE": SmECE(f=np.array(all_confidences), y=np.array(all_targets)),
        f"{split_name}_{infix}Brier_Score": brier_score_loss(
            y_true=all_correctness, y_prob=all_confidences
        ),
        f"{split_name}_{infix}AUROC": roc_auc_score(
            y_true=all_correctness, y_score=all_confidences,
        ),
        f"{split_name}_{infix}BLEU_accuracy": np.mean(all_correctness),
    }

    return metrics


def get_target_function(
        all_confidences: List[float],
        all_correctness: List[int], # all correctness of the model answers
        num_bins: int = 10, # number of bins to use for the target function
) -> Callable:
    """
        all_correctness: List[int]
            List of whether the target model was correct as ones and zeros.
        maps confidence to their target values as in ECE bining
    """
    bins = np.arange(0.0, 1.0, 1.0 / num_bins)
    bins_per_prediction = np.digitize(all_confidences, bins) # digitize the confidence scores into bins
    df = pd.DataFrame(
        {
            "y_pred": all_confidences,
            "y": all_correctness,
            "pred_bins": bins_per_prediction,
        }
    )

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean of y and predicted probabilities for each bin
    targets = grouped_by_bins.mean()["y"].values

    return np.vectorize(lambda conf: targets[np.abs(conf - targets).argmin()]) # return the target function as a vectorized function
