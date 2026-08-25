"""
Semantic Entropy (SE).

Reference implementation adapted from the Kuhn et al. (2023) and Farquhar
et al. (2024) work on semantic uncertainty:
    https://github.com/jlko/semantic_uncertainty

SE captures *meaning-level* uncertainty by sampling K generations, grouping
them into semantic-equivalence clusters using a bidirectional NLI entailment
check, and computing the entropy over the resulting cluster distribution.
"""
from .semantic_entropy import SemanticEntropy, SemanticEntropyResult

__all__ = ["SemanticEntropy", "SemanticEntropyResult"]
