"""Exact (enumerated, not sampled) check of which variants are Fisher consistent at finite n.

World: one query, 5 items, true p(d | u) and proposal q(d) arbitrary. A training example is a
positive y ~ p(. | u) plus n negatives drawn i.i.d. from q. We enumerate every (y, d_1..d_n)
with its probability and sum the gradient of the loss with respect to the item logits, evaluated
at the full-softmax optimum f(d) = log p(d | u). A zero expected gradient is a necessary stationarity check, not a proof of
Fisher consistency. The theorem additionally needs its stated assumptions.

Result (asserted below):
* "yi" without accidental-hit masking: zero gradient for every n -> stationary in these enumerated iid cases.
  This is the ranking-NCE result of Ma & Collins (2018): the corrected softmax is the exact
  posterior of "which candidate is the positive".
* "yi" with accidental-hit masking: not zero (masking changes the sampling process).
* "plugin" (= "improved"): not zero at finite n; the bias shrinks as n
  grows, in line with its n -> infinity consistency claim.
"""

import itertools

import torch

from cosmea.functional import SOURCE_IN_BATCH, mixed_negative_logq_loss

P_TRUE = torch.tensor([0.40, 0.25, 0.20, 0.10, 0.05], dtype=torch.float64)
Q_PROP = torch.tensor([0.50, 0.10, 0.20, 0.05, 0.15], dtype=torch.float64)
NUM_ITEMS = 5
TOTAL = 1000.0  # pretend training set: count(d) = q(d) * (N - 1)


def expected_gradient(num_negatives: int, variant: str, mask_same_item: bool) -> torch.Tensor:
    logits = torch.log(P_TRUE).clone().requires_grad_()
    batch_size = num_negatives + 1
    expected_grad = torch.zeros_like(logits)
    for positive in range(NUM_ITEMS):
        for negatives in itertools.product(range(NUM_ITEMS), repeat=num_negatives):
            probability = P_TRUE[positive] * torch.prod(Q_PROP[list(negatives)])
            cand = torch.tensor([positive, *negatives])
            label = {
                "query_id": torch.arange(batch_size),  # every row has its own query
                "cand_id": cand,
                "cand_count": (Q_PROP[cand] * TOTAL).float(),
                "cand_in_corpus": torch.zeros(batch_size, dtype=torch.bool),
                "cand_source": torch.full((batch_size,), SOURCE_IN_BATCH),
                "num_rows": torch.tensor(int(TOTAL) + 1),
                "corpus_size": torch.tensor(0),
            }
            loss = mixed_negative_logq_loss(
                logits[cand][None, :].to(torch.float64),
                label,
                rows=torch.tensor([0]),
                variant=variant,
                mask_same_item=mask_same_item,
                mask_same_query=False,
            )
            expected_grad += probability * torch.autograd.grad(loss[0], logits)[0]
    return expected_grad


def test_yi_without_masking_is_fisher_consistent_at_any_n():
    for n in (1, 2, 4):
        gradient = expected_gradient(n, "yi", mask_same_item=False)

        assert gradient.abs().max() < 1e-6, (n, gradient)


def test_yi_with_accidental_hit_masking_is_not():
    gradient = expected_gradient(3, "yi", mask_same_item=True)

    assert gradient.abs().max() > 1e-3, gradient


def test_plugin_is_biased_at_finite_n_and_improves_with_n():
    bias = [expected_gradient(n, "plugin", mask_same_item=True).abs().max().item() for n in (1, 2, 4, 6)]

    assert bias[0] > 1e-3
    assert bias[0] > bias[1] > bias[2] > bias[3], bias


if __name__ == "__main__":
    print("max |E[grad]| at f = log p(d|u)   (0 = Fisher consistent)")
    for n in (1, 2, 4, 6):
        row = {
            "yi, hits kept": expected_gradient(n, "yi", False),
            "yi, hits masked": expected_gradient(n, "yi", True),
            "plugin, hits masked": expected_gradient(n, "plugin", True),
            "plugin, hits kept": expected_gradient(n, "plugin", False),
        }
        print(f"n={n}: " + "  ".join(f"{k}: {v.abs().max().item():.2e}" for k, v in row.items()))


def expected_gradient_two_sources(num_in_batch: int, num_uniform: int, variant: str, combine: str) -> torch.Tensor:
    """Same enumeration with two proposals: q (in-batch) and uniform over the corpus (MNS)."""
    from cosmea.functional import SOURCE_UNIFORM

    logits = torch.log(P_TRUE).clone().requires_grad_()
    batch_size = num_in_batch + 1
    expected_grad = torch.zeros_like(logits)
    uniform_probability = 1.0 / NUM_ITEMS
    for positive in range(NUM_ITEMS):
        for in_batch in itertools.product(range(NUM_ITEMS), repeat=num_in_batch):
            for uniform in itertools.product(range(NUM_ITEMS), repeat=num_uniform):
                probability = P_TRUE[positive] * torch.prod(Q_PROP[list(in_batch)]) * uniform_probability**num_uniform
                cand = torch.tensor([positive, *in_batch, *uniform])
                label = {
                    "query_id": torch.arange(batch_size),
                    "cand_id": cand,
                    "cand_count": (Q_PROP[cand] * TOTAL).float(),
                    "cand_in_corpus": torch.ones(len(cand), dtype=torch.bool),
                    "cand_source": torch.cat(
                        [torch.full((batch_size,), SOURCE_IN_BATCH), torch.full((num_uniform,), SOURCE_UNIFORM)]
                    ),
                    "num_rows": torch.tensor(int(TOTAL) + 1),
                    "corpus_size": torch.tensor(NUM_ITEMS),
                }
                loss = mixed_negative_logq_loss(
                    logits[cand][None, :].to(torch.float64),
                    label,
                    rows=torch.tensor([0]),
                    variant=variant,
                    combine=combine,
                    mask_same_query=False,
                )
                expected_grad += probability * torch.autograd.grad(loss[0], logits)[0]
    return expected_grad


def test_mixed_sources_yi_separate_is_consistent_and_pooled_is_close():
    separate = expected_gradient_two_sources(2, 2, "yi", "separate").abs().max().item()
    pooled = expected_gradient_two_sources(2, 2, "yi", "pooled").abs().max().item()
    plugin = expected_gradient_two_sources(2, 2, "plugin", "pooled").abs().max().item()

    assert separate < 1e-6, separate
    assert separate < pooled < plugin, (separate, pooled, plugin)
