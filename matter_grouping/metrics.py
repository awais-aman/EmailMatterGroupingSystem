"""Evaluation helpers for groupings and summaries."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def grouping_scores(true_labels: Sequence[int], predicted: Sequence[int]) -> dict[str, float]:
    """ARI and NMI. Both ignore label names, so cluster 0 need not be matter 1."""
    if len(true_labels) != len(predicted):
        raise ValueError("label sequences differ in length")
    return {
        "ari": float(adjusted_rand_score(true_labels, predicted)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted)),
    }


def jaccard_similarity(text1: str, text2: str) -> float:
    """Word-set overlap: a cheap baseline for comparing a summary with a reference."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    union = words1 | words2
    return len(words1 & words2) / len(union) if union else 0.0
