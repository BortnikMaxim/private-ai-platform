"""Retrieval metrics used by the evaluation harness.

The metrics decide whether a pipeline change is an improvement, so they are
unit tested like production code rather than trusted by eye.
"""

import pytest

from backend.scripts.eval_retrieval import (
    hit_rate_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def test_recall_counts_the_share_of_relevant_documents_found():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)


def test_recall_respects_the_cutoff():
    retrieved = ["x", "y", "a"]

    assert recall_at_k(retrieved, {"a"}, k=2) == pytest.approx(0.0)
    assert recall_at_k(retrieved, {"a"}, k=3) == pytest.approx(1.0)


def test_recall_does_not_double_count_a_repeated_document():
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == pytest.approx(0.5)


def test_recall_without_relevant_documents_is_zero():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_recall_of_an_empty_result_is_zero():
    assert recall_at_k([], {"a"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# HitRate@k
# ---------------------------------------------------------------------------


def test_hit_rate_is_binary():
    assert hit_rate_at_k(["x", "a"], {"a", "b"}, k=2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_hit_rate_ignores_how_many_were_found():
    """Unlike recall, finding one of two relevant documents is a full hit."""
    assert hit_rate_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0
    assert hit_rate_at_k(["a", "x"], {"a", "b"}, k=2) == 1.0


def test_hit_rate_respects_the_cutoff():
    assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved,expected",
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank_of_the_first_relevant_hit(retrieved, expected):
    assert reciprocal_rank(retrieved, {"a"}) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_earliest_relevant_document():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------


def test_percentile_uses_nearest_rank():
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 0.50) == pytest.approx(50.0)
    assert percentile(values, 0.95) == pytest.approx(95.0)


def test_percentile_is_order_independent():
    assert percentile([9.0, 1.0, 5.0], 0.50) == percentile([1.0, 5.0, 9.0], 0.50)


def test_percentile_handles_a_single_sample():
    assert percentile([7.0], 0.95) == pytest.approx(7.0)


def test_percentile_of_nothing_is_zero():
    assert percentile([], 0.95) == 0.0


def test_percentile_never_indexes_past_the_end():
    """Nearest rank must stay inside the list for any fraction."""
    values = [1.0, 2.0, 3.0]

    for fraction in (0.0, 0.5, 0.99, 1.0):
        assert percentile(values, fraction) in values
