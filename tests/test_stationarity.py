"""Small iid stationarity oracles; these finite examples are not consistency proofs."""

import itertools

import pytest
import torch

from cosmea.functional import mixed_negative_logq_loss


@pytest.mark.parametrize("num_in_batch,num_uniform", [(2, 0), (1, 1)])
def test_yi_expected_gradient_vanishes_at_log_true_probability(num_in_batch, num_uniform):
    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    proposal = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    logits = probability.log().requires_grad_()
    expected_gradient = torch.zeros(3, dtype=torch.float64)
    batch_size = num_in_batch + 1
    for candidate_tuple in itertools.product(range(3), repeat=batch_size + num_uniform):
        positive, *negatives = candidate_tuple
        chance = probability[positive]
        for item in negatives[:num_in_batch]:
            chance = chance * proposal[item]
        chance = chance * (1 / 3) ** num_uniform
        ids = torch.tensor(candidate_tuple)
        label = {
            "query_id": torch.arange(batch_size),
            "cand_id": ids,
            "cand_count": (100 * proposal[ids]).long(),
            "cand_source": torch.tensor([0] * batch_size + [1] * num_uniform),
            "cand_in_corpus": torch.ones(len(ids), dtype=torch.bool),
            "num_rows": torch.tensor(100),
            "corpus_size": torch.tensor(3),
        }
        loss = mixed_negative_logq_loss(
            logits[ids][None, :],
            label,
            rows=torch.tensor([0]),
            mask_same_query=False,
        )
        expected_gradient += chance * torch.autograd.grad(loss.sum(), logits)[0]

    assert expected_gradient.abs().max() < 1e-12
