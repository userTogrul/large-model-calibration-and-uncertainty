"""
Diagnostic analysis: when does each confidence signal succeed and fail?

Two failure modes motivated by the rebuttal (Reviewer Gnde Q1, Reviewer 6rtY Q2):

  (A) Consistent hallucination
      The model agrees with itself across N sampled responses (freq is HIGH),
      but the claim is factually FALSE. This is the canonical TruthfulQA failure
      mode: a false premise is repeatedly affirmed. Show that verification
      against the gold answer assigns a LOWER score, i.e. ver catches what
      freq misses.

  (B) Verbalized miscalibration
      The model expresses HIGH verbalized / log-probability confidence in a
      claim that is FALSE, but cross-sample consistency is LOW (the model
      contradicts itself across samples). Here freq correctly reflects the
      underlying uncertainty that the verbalized signal hides.

Usage:
    python analyze_signal_complementarity.py \
        --model-name gpt-4o \
        --dataset-name truthful_qa \
        --max-examples 100 \
        --n-consistency-samples 5

Inputs are loaded from existing calibration_data dill files written by main.py.
Outputs are a JSON report + console summary; no model weights are written.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import dill
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from methods.ourmethod import ClaimUncertaintyMeasurer
from methods.ourmethod.claim_uncertainty import ClaimWithUncertainty
from src.constant_vals import DATA_DIR

try:
    from secret import OPENAI_API_KEY, BASE_URL
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["BASE_URL"] = BASE_URL
except (ImportError, ModuleNotFoundError):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    BASE_URL = os.environ.get("BASE_URL")
    if OPENAI_API_KEY is None:
        raise ImportError("Set OPENAI_API_KEY in secret.py or env vars.")


# ---------------------------------------------------------------------------
# Per-claim ground-truth label via gold-answer verifier
# ---------------------------------------------------------------------------

CLAIM_LABEL_PROMPT = (
    "You are a strict factual verifier. Decide whether the CLAIM is factually "
    "correct given the QUESTION and the GOLD ANSWER (treat the gold answer as "
    "ground truth). Answer with only 'True' or 'False'.\n\n"
    "QUESTION: {question}\n"
    "GOLD ANSWER: {gold_answer}\n"
    "CLAIM: {claim}\n\n"
    "Is the claim factually correct? Answer 'True' or 'False'."
)


def label_claim_against_gold(
    claim: str,
    question: str,
    gold_answer: str,
    openai_client: OpenAI,
    judge_model: str = "gpt-4o-mini",
) -> Optional[bool]:
    """Return True/False/None (None on judge failure)."""
    if not gold_answer or not claim.strip():
        return None
    prompt = CLAIM_LABEL_PROMPT.format(
        question=question or "",
        gold_answer=gold_answer,
        claim=claim,
    )
    try:
        resp = openai_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=4,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        if ans.startswith("true"):
            return True
        if ans.startswith("false"):
            return False
        return None
    except Exception as e:
        print(f"[judge] error: {e}")
        return None


# ---------------------------------------------------------------------------
# Records collected per claim
# ---------------------------------------------------------------------------

@dataclass
class ClaimRecord:
    question: str
    gold_answer: str
    claim: str
    ver: float
    freq: float
    label: Optional[bool]   # True = factually correct, False = hallucinated


@dataclass
class CategoryReport:
    name: str
    description: str
    threshold_high: float
    threshold_low: float
    n_matched: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    rescue_signal_mean: float = 0.0       # mean of the *other* signal on these claims
    target_signal_mean: float = 0.0       # mean of the *misleading* signal on these claims


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_test_split(
    data_dir: str,
    dataset_name: str,
    model_name: str,
    num_in_context_samples: int,
    split: str = "test",
) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(
        data_dir,
        dataset_name,
        model_name.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
        f"calibration_data_{split}.dill",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration data not found at {path}. Run main.py first to generate it."
        )
    with open(path, "rb") as f:
        data = dill.load(f)
    if "included_questions" in data:
        data.pop("included_questions", None)
    return data


def gold_answer_of(question_data: Dict[str, Any]) -> str:
    """Best-effort extraction of gold answer string from a calibration record."""
    g = question_data.get("gold_answer")
    if isinstance(g, list):
        return g[0] if g else ""
    if isinstance(g, str):
        return g
    # fallbacks
    for key in ("answer_text", "reference_answer", "target"):
        v = question_data.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def collect_claim_records(
    measurer: ClaimUncertaintyMeasurer,
    split_data: Dict[str, Dict[str, Any]],
    openai_client: OpenAI,
    n_consistency_samples: int,
    max_examples: int,
    judge_model: str,
) -> List[ClaimRecord]:
    items = list(split_data.values())[:max_examples]
    records: List[ClaimRecord] = []

    for question_data in tqdm(items, desc="Extracting claims + signals"):
        question = question_data.get("question", "") or ""
        answer = question_data.get("answer", "") or ""
        gold = gold_answer_of(question_data)
        if not answer.strip():
            continue

        try:
            claims: List[ClaimWithUncertainty] = measurer.measure_uncertainty_for_answer(
                answer=answer,
                question=question,
                openai_client=openai_client,
                alpha=0.0,                              # keep ver and freq separate
                n_consistency_samples=n_consistency_samples,
                verifier_model=judge_model,
            )
        except Exception as e:
            print(f"[measure] error on '{question[:60]}…': {e}")
            continue

        for c in claims:
            label = label_claim_against_gold(
                claim=c.text,
                question=question,
                gold_answer=gold,
                openai_client=openai_client,
                judge_model=judge_model,
            )
            records.append(
                ClaimRecord(
                    question=question,
                    gold_answer=gold,
                    claim=c.text,
                    ver=float(c.ver_confidence),
                    freq=float(c.freq_confidence),
                    label=label,
                )
            )
    return records


def categorize(
    records: List[ClaimRecord],
    high: float,
    low: float,
    top_k: int,
) -> Tuple[CategoryReport, CategoryReport]:
    consistent_halluc = CategoryReport(
        name="consistent_hallucination",
        description=(
            "Claim is FALSE, freq is HIGH (model agrees with itself), "
            "but ver is LOW — verification catches what consistency misses."
        ),
        threshold_high=high,
        threshold_low=low,
    )
    verbalized_miscal = CategoryReport(
        name="verbalized_miscalibration",
        description=(
            "Claim is FALSE, ver is HIGH (model states it confidently), "
            "but freq is LOW — consistency reflects the uncertainty that "
            "verbalized confidence hides."
        ),
        threshold_high=high,
        threshold_low=low,
    )

    false_only = [r for r in records if r.label is False]

    a_hits: List[ClaimRecord] = []
    b_hits: List[ClaimRecord] = []
    for r in false_only:
        if r.freq >= high and r.ver <= low:
            a_hits.append(r)
        if r.ver >= high and r.freq <= low:
            b_hits.append(r)

    # rank by margin so the most striking examples surface first
    a_hits.sort(key=lambda r: (r.freq - r.ver), reverse=True)
    b_hits.sort(key=lambda r: (r.ver - r.freq), reverse=True)

    consistent_halluc.n_matched = len(a_hits)
    verbalized_miscal.n_matched = len(b_hits)
    consistent_halluc.examples = [asdict(r) for r in a_hits[:top_k]]
    verbalized_miscal.examples = [asdict(r) for r in b_hits[:top_k]]

    if a_hits:
        consistent_halluc.target_signal_mean = float(np.mean([r.freq for r in a_hits]))
        consistent_halluc.rescue_signal_mean = float(np.mean([r.ver for r in a_hits]))
    if b_hits:
        verbalized_miscal.target_signal_mean = float(np.mean([r.ver for r in b_hits]))
        verbalized_miscal.rescue_signal_mean = float(np.mean([r.freq for r in b_hits]))

    return consistent_halluc, verbalized_miscal


def aggregate_signal_quality(records: List[ClaimRecord]) -> Dict[str, Any]:
    """Compare how well each signal separates true vs false claims (AUROC + means)."""
    from sklearn.metrics import roc_auc_score

    labeled = [r for r in records if r.label is not None]
    n_total = len(labeled)
    n_false = sum(1 for r in labeled if r.label is False)
    n_true = n_total - n_false
    summary: Dict[str, Any] = {
        "n_claims_labeled": n_total,
        "n_true_claims": n_true,
        "n_false_claims": n_false,
    }
    if n_total == 0 or n_false == 0 or n_true == 0:
        summary["auroc_ver"] = None
        summary["auroc_freq"] = None
        return summary

    y = np.array([1 if r.label else 0 for r in labeled])
    ver = np.array([r.ver for r in labeled])
    freq = np.array([r.freq for r in labeled])

    summary["auroc_ver"] = round(float(roc_auc_score(y, ver)), 4)
    summary["auroc_freq"] = round(float(roc_auc_score(y, freq)), 4)
    summary["mean_ver_on_true"] = round(float(ver[y == 1].mean()), 4)
    summary["mean_ver_on_false"] = round(float(ver[y == 0].mean()), 4)
    summary["mean_freq_on_true"] = round(float(freq[y == 1].mean()), 4)
    summary["mean_freq_on_false"] = round(float(freq[y == 0].mean()), 4)
    return summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_category(report: CategoryReport, top_k: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{report.name}]  matched: {report.n_matched}")
    print(report.description)
    print(
        f"thresholds: HIGH ≥ {report.threshold_high}  LOW ≤ {report.threshold_low}"
    )
    if report.n_matched:
        print(
            f"  mean misleading signal = {report.target_signal_mean:.3f}   "
            f"mean rescue signal     = {report.rescue_signal_mean:.3f}   "
            f"gap = {report.target_signal_mean - report.rescue_signal_mean:+.3f}"
        )
    print("-" * 78)
    for i, ex in enumerate(report.examples[:top_k], 1):
        print(f"\nExample {i}:")
        print(f"  Q:    {ex['question'][:140]}")
        print(f"  Gold: {ex['gold_answer'][:140]}")
        print(f"  Claim:{ex['claim'][:140]}")
        print(f"  ver={ex['ver']:.3f}  freq={ex['freq']:.3f}  label={ex['label']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True,
                        help="Same model identifier used by main.py to write calibration_data.")
    parser.add_argument("--dataset-name", required=True,
                        choices=["trivia_qa", "truthful_qa", "natural_questions"])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--num-in-context-samples", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=100,
                        help="Cap on number of QA records analyzed (each yields ~3-6 claims).")
    parser.add_argument("--n-consistency-samples", type=int, default=5,
                        help="N for freq(c_i) sampling; 0 disables freq.")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Closed-box model used both as cross-sample verifier and gold-answer judge.")
    parser.add_argument("--high-threshold", type=float, default=0.7,
                        help="Confidence considered HIGH for either signal.")
    parser.add_argument("--low-threshold", type=float, default=0.4,
                        help="Confidence considered LOW for either signal.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top examples to display per category.")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the full JSON report.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.high_threshold <= args.low_threshold:
        raise ValueError("--high-threshold must be strictly greater than --low-threshold")
    if args.n_consistency_samples <= 0:
        warnings.warn("n_consistency_samples=0 disables freq; analysis becomes degenerate.")

    print(f"Loading {args.dataset_name}/{args.model_name} {args.split} split…")
    split_data = load_test_split(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        num_in_context_samples=args.num_in_context_samples,
        split=args.split,
    )
    print(f"  loaded {len(split_data)} QA records (capped to {args.max_examples}).")

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    measurer = ClaimUncertaintyMeasurer(
        model_name=args.model_name,
        device=args.device,
        claim_extractor_model="gpt-4o-mini",
    )

    records = collect_claim_records(
        measurer=measurer,
        split_data=split_data,
        openai_client=openai_client,
        n_consistency_samples=args.n_consistency_samples,
        max_examples=args.max_examples,
        judge_model=args.judge_model,
    )
    print(f"\nCollected {len(records)} per-claim records "
          f"({sum(1 for r in records if r.label is False)} labeled false, "
          f"{sum(1 for r in records if r.label is True)} labeled true, "
          f"{sum(1 for r in records if r.label is None)} unlabeled).")

    consistent_halluc, verbalized_miscal = categorize(
        records,
        high=args.high_threshold,
        low=args.low_threshold,
        top_k=args.top_k,
    )
    print_category(consistent_halluc, args.top_k)
    print_category(verbalized_miscal, args.top_k)

    overall = aggregate_signal_quality(records)
    print("\n" + "=" * 78)
    print("[Aggregate signal quality on labeled claims]")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    if args.output:
        report = {
            "config": vars(args),
            "n_records": len(records),
            "aggregate": overall,
            "categories": [asdict(consistent_halluc), asdict(verbalized_miscal)],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote full report to {args.output}")


if __name__ == "__main__":
    main()
