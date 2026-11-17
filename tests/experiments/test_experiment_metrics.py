"""Independent ranking examples for the synthetic comparison's evaluation contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[2] / "experiments" / "toy_two_tower.py"
    spec = importlib.util.spec_from_file_location("cosmea_toy_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedScores:
    """A score table that makes the expected top-k set independent of training."""

    def __init__(self, scores):
        self.scores = torch.as_tensor(scores, dtype=torch.float64)[None, :]

    def all_scores(self):
        return self.scores


def test_tail_starts_immediately_after_top_ten_percent(experiment):
    scores = -np.arange(100, dtype=float)
    scores[9] = -1000  # Top ten contains nine head items and the first tail item.
    popularity = 1 / np.arange(1, 101, dtype=float)
    popularity /= popularity.sum()
    result = experiment.evaluate(FixedScores(scores), popularity, np.full((1, 100), 0.01), np.ones(100))

    assert result["tailR@10"] == pytest.approx(1 / 90)


def test_tied_popularity_has_a_deterministic_nonempty_tail(experiment):
    probability = np.full(40, 1 / 40)
    result = experiment.evaluate(FixedScores(-np.arange(40)), probability, probability[None, :], np.ones(40))

    assert result["tailR@10"] == pytest.approx(6 / 36)
    assert all(np.isfinite(value) for value in result.values())


def test_unseen_share_uses_actual_number_of_retrieved_slots(experiment):
    probability = np.full(5, 1 / 5)
    result = experiment.evaluate(FixedScores(-np.arange(5)), probability, probability[None, :], np.zeros(5))

    assert result["unseen@10"] == pytest.approx(1.0)
    assert result["R@10"] == pytest.approx(1.0)
