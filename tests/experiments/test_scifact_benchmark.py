"""Guard the held-out split used for the real-data benchmark."""

import math
from pathlib import Path

import pytest
from datasets import Dataset


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "experiments"))
    import scifact_benchmark

    return scifact_benchmark


def test_sample_excludes_test_text_and_positive_documents_independently(benchmark):
    queries = {"1": "A copied QUERY", "2": "another claim", "3": "different claim", "4": "a  copied query"}
    train = {"1": {"doc-e"}, "2": {"doc-b", "doc-c"}, "3": {"doc-a", "doc-d"}}
    test = {"4": {"doc-a"}}

    pairs, excluded = benchmark.select_train_pairs(queries, train, test, count=3, seed=42)
    assert set(pairs) == {("2", "doc-b"), ("2", "doc-c"), ("3", "doc-d")}
    assert excluded == {
        "excluded_matching_query_text": 1,
        "excluded_test_positive_document": 1,
        "eligible_pairs": 3,
    }
    shuffled, _ = benchmark.select_train_pairs(queries, dict(reversed(train.items())), test, count=3, seed=42)
    assert shuffled == pairs


def test_missing_test_positive_is_rejected(benchmark):
    with pytest.raises(ValueError, match="Test judgments reference unknown IDs.*missing"):
        benchmark.validate_judgments({"q": "query"}, {"d": "document"}, {"q": {"d", "missing"}}, "Test")


def test_mnrl_schedule_rejects_an_epoch_that_needs_extra_batches(benchmark):
    size = benchmark.BATCH_SIZE * 2
    dataset = Dataset.from_dict(
        {
            "anchor": [f"query {index}" for index in range(size)],
            "positive": ["shared document"] * 3 + [f"document {index}" for index in range(3, size)],
        }
    )
    with pytest.raises(ValueError, match="NO_DUPLICATES cannot consume all rows"):
        benchmark.validate_mnrl_schedule(dataset, data_seed=42, seed=42)


def test_per_query_metrics_use_distinct_cutoffs_and_all_positives(benchmark):
    hits = [{"corpus_id": f"d{index}", "score": -index} for index in range(102)]
    scores = benchmark.query_metrics(hits, {"d2", "d10", "d100"})

    assert scores["mrr@10"] == pytest.approx(1 / 3)
    assert scores["ndcg@10"] == pytest.approx((1 / math.log2(4)) / (1 + 1 / math.log2(3) + 1 / math.log2(4)))
    assert scores["recall@10"] == pytest.approx(1 / 3)
    assert scores["recall@100"] == pytest.approx(2 / 3)
