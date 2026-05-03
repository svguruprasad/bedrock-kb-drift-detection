"""Tests for similarity metrics and statistical testing."""

import math
from similarity import cosine_similarity, rouge_l, composite_score, _lcs_length
from stats import (
    t_test_one_sample,
    bonferroni_correction,
    classify_tier,
    run_drift_tests,
    TestResult,
)


# --- Cosine similarity tests ---

def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similar_vectors():
    a = [1.0, 2.0, 3.0]
    b = [1.1, 2.1, 3.1]
    score = cosine_similarity(a, b)
    assert score > 0.99


def test_cosine_high_dimensional():
    a = [float(i) for i in range(1024)]
    b = [float(i) + 0.01 for i in range(1024)]
    score = cosine_similarity(a, b)
    assert score > 0.999


# --- ROUGE-L tests ---

def test_rouge_l_identical():
    text = "the cat sat on the mat"
    assert abs(rouge_l(text, text) - 1.0) < 1e-6


def test_rouge_l_no_overlap():
    a = "the cat sat"
    b = "dogs run fast"
    assert rouge_l(a, b) == 0.0


def test_rouge_l_partial_overlap():
    a = "the cat sat on the mat"
    b = "the cat on the mat"
    score = rouge_l(a, b)
    assert 0.5 < score < 1.0


def test_rouge_l_empty():
    assert rouge_l("", "hello") == 0.0
    assert rouge_l("hello", "") == 0.0


def test_rouge_l_single_word():
    assert abs(rouge_l("hello", "hello") - 1.0) < 1e-6


# --- LCS tests ---

def test_lcs_identical():
    tokens = ["a", "b", "c"]
    assert _lcs_length(tokens, tokens) == 3


def test_lcs_no_common():
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0


def test_lcs_partial():
    a = ["a", "b", "c", "d"]
    b = ["a", "c", "d"]
    assert _lcs_length(a, b) == 3


# --- Composite score tests ---

def test_composite_equal_weights():
    score = composite_score(0.9, 0.8, 0.7, 1 / 3, 1 / 3, 1 / 3)
    expected = (0.9 + 0.8 + 0.7) / 3
    assert abs(score - expected) < 1e-6


def test_composite_factual_heavy():
    score = composite_score(0.9, 0.8, 0.5, 0.4, 0.1, 0.5)
    expected = 0.4 * 0.9 + 0.1 * 0.8 + 0.5 * 0.5
    assert abs(score - expected) < 1e-6


def test_composite_perfect():
    score = composite_score(1.0, 1.0, 1.0, 0.4, 0.1, 0.5)
    assert abs(score - 1.0) < 1e-6


def test_composite_zero():
    score = composite_score(0.0, 0.0, 0.0, 0.4, 0.1, 0.5)
    assert abs(score) < 1e-6


# --- Statistical tests ---

def test_t_test_clearly_above_threshold():
    scores = [0.95, 0.96, 0.94, 0.95, 0.96, 0.95, 0.94, 0.95, 0.96, 0.95]
    t_stat, p_value, passed = t_test_one_sample(scores, 0.85)
    assert passed is True
    assert t_stat > 0


def test_t_test_clearly_below_threshold():
    scores = [0.70, 0.72, 0.71, 0.69, 0.70, 0.71, 0.72, 0.70, 0.69, 0.71]
    t_stat, p_value, passed = t_test_one_sample(scores, 0.85)
    assert passed is False
    assert t_stat < 0


def test_t_test_at_threshold():
    scores = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    t_stat, p_value, passed = t_test_one_sample(scores, 0.85)
    assert passed is True


def test_bonferroni():
    adjusted = bonferroni_correction(0.05, 10)
    assert abs(adjusted - 0.005) < 1e-6


def test_bonferroni_single():
    adjusted = bonferroni_correction(0.05, 1)
    assert abs(adjusted - 0.05) < 1e-6


# --- Tier classification tests ---

def test_tier_all_pass():
    results = [
        TestResult("q1", 0.95, 0.01, 5.0, 0.99, True, "Green", 10),
        TestResult("q2", 0.92, 0.02, 3.0, 0.95, True, "Green", 10),
    ]
    assert classify_tier(results) == "Green"


def test_tier_all_fail():
    results = [
        TestResult("q1", 0.70, 0.05, -3.0, 0.01, False, "Red", 10),
        TestResult("q2", 0.65, 0.04, -4.0, 0.005, False, "Red", 10),
    ]
    assert classify_tier(results) == "Red"


def test_tier_empty():
    assert classify_tier([]) == "Green"


# --- Integration test ---

def test_run_drift_tests_all_pass():
    query_scores = {
        "q1": [0.95, 0.96, 0.94, 0.95, 0.96],
        "q2": [0.92, 0.93, 0.91, 0.92, 0.93],
    }
    results = run_drift_tests(query_scores, threshold=0.85, alpha=0.05)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_run_drift_tests_one_fails():
    query_scores = {
        "q1": [0.95, 0.96, 0.94, 0.95, 0.96],
        "q2": [0.70, 0.72, 0.71, 0.69, 0.70],
    }
    results = run_drift_tests(query_scores, threshold=0.85, alpha=0.05)
    assert len(results) == 2
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)
    assert passed_count == 1
    assert failed_count == 1
