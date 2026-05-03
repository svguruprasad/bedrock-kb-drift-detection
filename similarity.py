"""Similarity metrics for measuring response equivalence."""

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 means identical direction.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F1 score using longest common subsequence.

    Returns a value in [0, 1] where 1 means identical token sequences.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, cand_tokens)

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    beta = 1.0
    f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return f1


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def composite_score(
    semantic: float,
    structural: float,
    factual: float,
    w_sem: float = 0.40,
    w_struct: float = 0.10,
    w_fact: float = 0.50,
) -> float:
    """Compute weighted composite drift score.

    All inputs and output are in [0, 1].
    Weights must sum to 1.0.
    """
    assert abs(w_sem + w_struct + w_fact - 1.0) < 1e-6, "Weights must sum to 1.0"
    return w_sem * semantic + w_struct * structural + w_fact * factual
