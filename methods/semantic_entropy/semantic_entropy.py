"""
Semantic Entropy for free-form generative QA.

Adapted from the reference implementation by Kuhn et al. (2023) and
Farquhar et al. (2024):
    https://github.com/jlko/semantic_uncertainty/tree/master

The estimator follows three steps:
    1. Sample K generations s_1, ..., s_K from the model for the same prompt
       at a non-zero temperature, together with their length-normalized
       sequence log-likelihoods log p(s_k | x).
    2. Cluster the K samples into semantic equivalence classes C_1, ..., C_M
       using bidirectional NLI entailment: s_i and s_j are placed in the same
       cluster iff NLI(s_i -> s_j) == entailment AND NLI(s_j -> s_i) == entailment.
    3. Compute the Monte-Carlo estimate of cluster-marginal entropy.

We return SE both as a raw entropy and as a confidence score in [0, 1]
suitable for plugging into the existing calibration / AUROC / AURC pipeline.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_NLI_MODEL = "microsoft/deberta-large-mnli"


@dataclass
class SemanticEntropyResult:
    """Container for SE outputs for a single (question, answer) pair."""

    samples: List[str]
    sample_log_likelihoods: List[float]
    cluster_assignments: List[int]
    cluster_log_probs: List[float]
    semantic_entropy: float
    confidence: float
    metadata: dict = field(default_factory=dict)


class SemanticEntropy:
    """Compute semantic entropy for generative QA.

    The class supports two evaluation backends:
        * ``open_box``: an open-weight LLM loaded via Transformers. The
          generation step uses sampling with ``temperature`` and we score each
          sampled sequence with the model's own log-probabilities.
        * ``closed_box``: an OpenAI-compatible chat endpoint. The generation
          step issues K independent sampled completions and we treat each
          completion as equally likely under the model (uniform Monte-Carlo
          estimate), since per-token log-probabilities are not always
          available for chat endpoints.

    Entailment is computed with an NLI classifier (DeBERTa-MNLI by default),
    matching the original ``semantic_uncertainty`` reference repo.
    """

    def __init__(
        self,
        nli_model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_samples: int = 10,
        generation_temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.device = device
        self.num_samples = int(num_samples)
        self.generation_temperature = float(generation_temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = seed

        self._nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name
        ).to(device)
        self._nli_model.eval()

        # Resolve the entailment label index once (MNLI labels differ slightly
        # between checkpoints).
        id2label = {int(k): v.lower() for k, v in self._nli_model.config.id2label.items()}
        try:
            self._entail_id = next(k for k, v in id2label.items() if "entail" in v)
        except StopIteration as exc:  # pragma: no cover -- defensive
            raise ValueError(
                f"NLI checkpoint {nli_model_name} has no entailment label: {id2label}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Generation back-ends                                               #
    # ------------------------------------------------------------------ #

    def sample_open_box(
        self,
        prompt: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> Tuple[List[str], List[float]]:
        """Sample K generations with length-normalized log-likelihoods."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=self.generation_temperature,
                num_return_sequences=self.num_samples,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        sequences = outputs.sequences[:, input_len:]
        # outputs.scores is a list of length T_new, each [K, vocab]
        log_likelihoods: List[float] = []
        samples: List[str] = []
        for k in range(sequences.shape[0]):
            tok_ids = sequences[k]
            log_p = 0.0
            n_tok = 0
            for t, score in enumerate(outputs.scores):
                if t >= tok_ids.shape[0]:
                    break
                tok_id = tok_ids[t].item()
                if tok_id == tokenizer.eos_token_id:
                    break
                log_probs = torch.log_softmax(score[k], dim=-1)
                log_p += float(log_probs[tok_id].item())
                n_tok += 1
            # length-normalized
            log_likelihoods.append(log_p / max(1, n_tok))
            samples.append(
                tokenizer.decode(tok_ids, skip_special_tokens=True).strip()
            )

        return samples, log_likelihoods

    def sample_closed_box(
        self,
        prompt: str,
        openai_client: OpenAI,
        model_name: str,
    ) -> Tuple[List[str], List[float]]:
        """Sample K generations from an OpenAI-compatible chat endpoint.

        We use a uniform Monte-Carlo estimate of cluster probability since
        token-level log-likelihoods are not consistently exposed by chat APIs.
        """
        samples: List[str] = []
        for _ in range(self.num_samples):
            try:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.generation_temperature,
                    max_completion_tokens=self.max_new_tokens,
                    n=1,
                )
                samples.append(response.choices[0].message.content.strip())
            except Exception as exc:  # pragma: no cover -- network errors
                warnings.warn(f"Closed-box sample failed: {exc}")
                samples.append("")
        # Uniform log-likelihoods (Monte-Carlo).
        log_likelihoods = [-math.log(max(1, len(samples)))] * len(samples)
        return samples, log_likelihoods

    # ------------------------------------------------------------------ #
    # Clustering and entropy                                             #
    # ------------------------------------------------------------------ #

    def _entails(self, premise: str, hypothesis: str, context: str = "") -> bool:
        """Bidirectional helper: ``True`` if premise entails hypothesis."""
        text = (
            f"{context} {premise}".strip() if context else premise
        )
        inputs = self._nli_tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits[0]
        return int(torch.argmax(logits).item()) == self._entail_id

    def cluster_by_entailment(
        self,
        samples: Sequence[str],
        question: str = "",
    ) -> List[int]:
        """Bidirectional-entailment clustering (Kuhn et al. 2023, Alg. 1)."""
        cluster_ids: List[int] = [-1] * len(samples)
        next_id = 0
        for i, s_i in enumerate(samples):
            if cluster_ids[i] != -1:
                continue
            cluster_ids[i] = next_id
            for j in range(i + 1, len(samples)):
                if cluster_ids[j] != -1:
                    continue
                if not s_i or not samples[j]:
                    continue
                if self._entails(s_i, samples[j], question) and self._entails(
                    samples[j], s_i, question
                ):
                    cluster_ids[j] = next_id
            next_id += 1
        return cluster_ids

    @staticmethod
    def _entropy_from_cluster_log_probs(cluster_log_probs: Sequence[float]) -> float:
        """Compute -sum_c p(c) log p(c) given (unnormalized) log probabilities."""
        if not cluster_log_probs:
            return 0.0
        m = max(cluster_log_probs)
        norm = m + math.log(sum(math.exp(lp - m) for lp in cluster_log_probs))
        probs = [math.exp(lp - norm) for lp in cluster_log_probs]
        return float(-sum(p * math.log(p + 1e-16) for p in probs))

    @staticmethod
    def _entropy_to_confidence(entropy: float, num_clusters: int) -> float:
        """Map raw entropy to a confidence score in [0, 1].

        We normalize by log(num_clusters) (the maximum possible entropy for
        the observed cluster count) and invert.
        """
        if num_clusters <= 1:
            return 1.0
        max_h = math.log(num_clusters)
        normalized = min(1.0, entropy / max_h)
        return float(1.0 - normalized)

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def measure(
        self,
        prompt: str,
        question: str = "",
        *,
        # open-box back-end
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        # closed-box back-end
        openai_client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
    ) -> SemanticEntropyResult:
        """Compute semantic entropy for a single prompt."""
        if model is not None and tokenizer is not None:
            samples, log_lls = self.sample_open_box(prompt, model, tokenizer)
        elif openai_client is not None and model_name is not None:
            samples, log_lls = self.sample_closed_box(prompt, openai_client, model_name)
        else:
            raise ValueError(
                "Provide either (model, tokenizer) for open-box scoring or "
                "(openai_client, model_name) for closed-box scoring."
            )

        cluster_ids = self.cluster_by_entailment(samples, question=question)
        n_clusters = max(cluster_ids) + 1 if cluster_ids else 0

        # Aggregate sample log-likelihoods per cluster via log-sum-exp.
        cluster_log_probs: List[float] = []
        for c in range(n_clusters):
            members = [lp for lp, cid in zip(log_lls, cluster_ids) if cid == c]
            if not members:
                continue
            m = max(members)
            cluster_log_probs.append(
                m + math.log(sum(math.exp(lp - m) for lp in members))
            )

        entropy = self._entropy_from_cluster_log_probs(cluster_log_probs)
        confidence = self._entropy_to_confidence(entropy, len(cluster_log_probs))

        return SemanticEntropyResult(
            samples=samples,
            sample_log_likelihoods=log_lls,
            cluster_assignments=cluster_ids,
            cluster_log_probs=cluster_log_probs,
            semantic_entropy=entropy,
            confidence=confidence,
            metadata={
                "num_samples": self.num_samples,
                "num_clusters": len(cluster_log_probs),
                "temperature": self.generation_temperature,
            },
        )
