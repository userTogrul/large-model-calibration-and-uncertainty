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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from methods.ece.ECE import eceloss_adapted, ece
from methods.uce import UCE
from sklearn.metrics import brier_score_loss, roc_auc_score # BS and AUROC
from torcheval.metrics.functional import bleu_score # BLEU score
from relplot.metrics import smECE as SmECE # Smooth ECE metric from the paper
import logging

"""
    Loosely modified from:
        https://github.com/parameterlab/apricot/blob/main/src/evaluation.py
"""

logging.basicConfig(
    filename='calibration.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def calculate_aurc(
        confidences: List[float],
        correctness: List[int],
        num_points: int = 100
) -> float:
    """
    Calculate Area Under the Risk-Coverage curve (AURC).
    
    The Risk-Coverage curve shows the relationship between the error rate (risk)
    and the fraction of samples the model makes predictions for (coverage).
    Lower AURC values indicate better selective prediction performance.
    
    Args:
        confidences: List of confidence scores
        correctness: List of correctness indicators (0 or 1)
        num_points: Number of points to evaluate on the curve
        
    Returns:
        Area under the risk-coverage curve
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Sort by confidence in descending order
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_correctness = correctness[sorted_indices]
    
    coverages = np.linspace(0.01, 1.0, num_points)  # Avoid division by zero at 0
    risks = []
    
    for coverage in coverages:
        n_samples = int(len(correctness) * coverage)
        if n_samples == 0:
            risks.append(0.0)
            continue
            
        # Calculate error rate for current coverage
        errors = 1 - sorted_correctness[:n_samples]
        risk = np.mean(errors)
        risks.append(risk)
    
    # Calculate area under the curve using trapezoidal rule
    aurc = np.trapz(risks, coverages)
    return float(aurc)

def AUARC(labels, confidences):
    # An accuracy rejection curve (ARC) is a function representating the accuracy of a classifier \
    # as a function of its reject rate. 
    # An ARC is therefore produced by plotting the accuracy of a classifier against its reject rate, from [0,1].
    # All ARCs have an accuracy of 100% for a rejection rate of 100%
    # Source:
    #     https://github.com/shuoli90/Rank-Calibration/blob/main/metrics/calibration.py 
    rejection_rate = np.arange(0, 1, 0.01)
    if type(confidences) is not np.ndarray:
        confidences = np.array(confidences)
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    arcs = []
    for rate in rejection_rate:
        labels_tmp = labels[confidences > np.quantile(confidences, rate)]
        if len(labels_tmp) == 0:
            arc = 1.0
        else:
            # reject rate portion of the data
            arc = np.mean(labels_tmp)
        arcs.append(arc)
    return np.trapz(arcs, rejection_rate)

def check_answer_correctness(
    correct_answers: List[Union[str, List[str]]], 
    model_answers: List[str],
    bleu_threshold: float = 0.25,
    n_gram: int = 1,
) -> Tuple[List[bool], List[float]]:
    """
    Check correctness of model answers against correct answers using BLEU score.
    
    Args:
        correct_answers: List of correct answers or list of lists of correct answers
        model_answers: List of model generated answers
        bleu_threshold: Threshold for BLEU score to consider answer correct
        n_gram: N-gram level for BLEU score calculation
        
    Returns:
        Tuple of (correctness list, scores list)
    """

    results = []
    scores = []

    for correct_answer, model_answer in zip(correct_answers, model_answers):
        try:
            
            if isinstance(correct_answer, str):
                correct_answer = [correct_answer]
            
            score = bleu_score(
                [model_answer], 
                [correct_answer], # can be list of possible answers
                n_gram=n_gram,
            )
            
            # Check both BLEU score and substring match
            is_correct = (score >= bleu_threshold or 
                        any(ans in model_answer or model_answer in ans 
                            for ans in correct_answer))
            
            scores.append(float(score.item()))
            results.append(is_correct)

        except Exception as e:
            logging.error(f"Error in answer evaluation: {str(e)}")
            logging.error(f"Model answer: {model_answer}")
            logging.error(f"Correct answer: {correct_answer}")
            results.append(False)
            scores.append(0.0)

    return results, scores

def extract_verbalized_confidence(
        expressions: List[str],
        mode: str,
        expression_mapping: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[bool]]:
    """
    Extract confidence values from verbalized expressions.
    
    Args:
        expressions: List of verbalized confidence expressions
        mode: Either "qualitative" or "quantitative"
        expression_mapping: Mapping from expressions to confidence values
        
    Returns:
        Tuple of (confidence values, success flags)
    """
    if not expressions:
        return [], []
    assert mode in (
        "qualitative",
        "quantitative",
    ), f"Mode has to be either qualitative or quantitative, but {mode} found."

    if mode == "qualitative":
        assert (
            expression_mapping is not None
        ), "'expression_mapping' has to be specified for qualitative mode."
    
    confidences, successful = [], [] # whether successfully able to extract confidences [SUCCESS RATE]

    for expression in expressions:
        if mode == "qualitative":
            template = rf"({'|'.join(expression_mapping.keys())})"
        
        try:
            res = re.search(template, expression).group(0)

            if mode == "qualitative":
                conf = expression_mapping[res]
            
            successful.append(True)
            confidences.append(conf)
        
        except AttributeError: # interesting error catch
            successful.append(False)

    return confidences, successful

def evaluate_confidences(
        split_name: str,
        all_confidences: List[float],
        all_correctness: List[int],
        all_targets: Optional[List[float]] = None,
        all_scores: Optional[List[float]] = None,
        num_bins: int = 10,
        add_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model confidence predictions using multiple metrics.
    
    Args:
        split_name: Name of the data split (e.g., 'train', 'val', 'test')
        all_confidences: List of confidence scores predicted by the model
        all_correctness: List of binary correctness indicators (0 or 1)
        all_targets: Optional list of target values for calibration
        all_scores: Optional list of BLEU scores for answer evaluation
        num_bins: Number of bins for discretization in calibration
        add_name: Optional additional name to append to metric names
        
    Returns:
        Dictionary containing the following metrics:
        - ECE (Expected Calibration Error)
        - SmECE (Smooth Expected Calibration Error)
        - Brier Score
        - AUROC (Area Under ROC Curve)
        - AURC (Area Under Risk-Coverage Curve)
        - BLEU Score (if all_scores provided)
    """
    if all_targets is None:
        target_func = get_target_function(all_confidences, all_correctness, num_bins)
        all_targets = target_func(all_confidences)

    infix =  ""
    if add_name is not None:
        infix = f"{add_name}_"
        
    metrics = {
        f"{split_name}_{infix}ece": ece(y_true=all_targets, y_pred=all_confidences),
        f"{split_name}_{infix}smece": SmECE(f=np.array(all_confidences), y=np.array(all_targets)),
        f"{split_name}_{infix}brier_score": brier_score_loss(
            y_true=all_correctness, y_proba=all_confidences
        ),
        f"{split_name}_{infix}auroc": roc_auc_score(
            y_true=all_correctness, y_score=all_confidences,
        ),
        f"{split_name}_{infix}aurc": calculate_aurc(
            confidences=all_confidences,
            correctness=all_correctness,
        ),
        f"{split_name}_{infix}bleu": np.mean(all_scores) if all_scores is not None else 0.0,
    }

    return metrics

def get_target_function(
        all_confidences: List[float],
        all_correctness: List[int],
        num_bins: int = 10,
) -> Callable:
    """
    Get target function for calibration mapping confidence scores to target values.
    
    Args:
        all_confidences: List of confidence scores
        all_correctness: List of correctness indicators (0 or 1)
        num_bins: Number of bins for discretization
        
    Returns:
        Callable that maps confidence scores to target values
    """
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    
    all_confidences = np.array(all_confidences)
    if np.any(np.isnan(all_confidences)) or np.any(np.isinf(all_confidences)):
        raise ValueError("Input contains NaN or infinite values")
    
    bins = np.arange(0.0, 1.0, 1.0 / num_bins)
    bins_per_prediction = np.digitize(all_confidences, bins) # digitize the confidence scores into bins
    df = pd.DataFrame(
        {
            "y_pred": all_confidences,
            "y": all_correctness,
            "pred_bins": bins_per_prediction,
        }
    )

    # Calculate targets for each bin
    grouped_by_bins = df.groupby("pred_bins")
    targets = grouped_by_bins.mean()["y"].values

    return np.vectorize(lambda conf: targets[np.abs(conf - targets).argmin()]) # return the target function as a vectorized function that returns the correctness rate (accuracy) of bin that conf belongs.

