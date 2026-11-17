"""Enumerate iid score gradients in one small world.

These stationarity examples illustrate masking and finite-sample effects;
they are not proofs of consistency or general bias orderings.
"""

import itertools

import torch

from cosmea.functional import SOURCE_IN_BATCH, mixed_negative_logq_loss

P_TRUE = torch.tensor([0.40, 0.25, 0.20, 0.10, 0.05], dtype=torch.float64)
Q_PROP = torch.tensor([0.50, 0.10, 0.20, 0.05, 0.15], dtype=torch.float64)
NUM_ITEMS = 5
TOTAL = 1000.0  # pretend training set: count(d) = q(d) * N


def expected_gradient(num_negatives: int, variant: str, mask_same_document: bool) -> torch.Tensor:
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
                "num_rows": torch.tensor(int(TOTAL)),
                "corpus_size": torch.tensor(0),
            }
            loss = mixed_negative_logq_loss(
                logits[cand][None, :].to(torch.float64),
                label,
                rows=torch.tensor([0]),
                variant=variant,
                mask_same_document=mask_same_document,
                mask_same_query=False,
            )
            expected_grad += probability * torch.autograd.grad(loss[0], logits)[0]
    return expected_grad


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stationarity illustration, not a consistency proof")
    parser.add_argument("--max-negatives", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_negatives <= 6:
        parser.error("--max-negatives must be in [1, 6]; enumeration grows exponentially")
    for n in range(1, args.max_negatives + 1):
        for variant, masked in (("yi", False), ("yi", True), ("plugin", True)):
            gradient = expected_gradient(n, variant, masked)
            print(f"n={n} {variant} mask_same_document={masked}: max |E[gradient]|={gradient.abs().max().item():.6g}")
