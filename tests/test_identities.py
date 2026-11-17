"""Exact identities of the loss, checked with autograd.

Run: python -m pytest tests -q
"""

import itertools

import torch

from cosmea.functional import (
    SOURCE_IN_BATCH,
    SOURCE_OTHER,
    SOURCE_UNIFORM,
    mixed_negative_logq_loss,
)


def make_label(batch_size=12, num_uniform=7, num_other=0, num_documents=9, num_queries=6, seed=0):
    """Random metadata with many duplicate documents and repeated queries (the awkward cases)."""
    g = torch.Generator().manual_seed(seed)
    num_rows = 500
    corpus_size = 40
    counts = torch.randint(1, 60, (num_documents + corpus_size,), generator=g).float()
    pos = torch.randint(0, num_documents, (batch_size,), generator=g)
    other = torch.randint(0, num_documents + corpus_size, (batch_size * num_other,), generator=g)
    uni = num_documents + torch.randperm(corpus_size, generator=g)[:num_uniform]
    cand_id = torch.cat([pos, other, uni])
    # Items >= num_documents are corpus-only (count 0 in training) for the uniform part.
    cand_count = counts[cand_id] * (cand_id < num_documents)
    return {
        "query_id": torch.randint(0, num_queries, (batch_size,), generator=g),
        "cand_id": cand_id,
        "cand_count": cand_count,
        "cand_in_corpus": torch.ones_like(cand_id, dtype=torch.bool),  # positives need uniform proposal support
        "cand_source": torch.cat(
            [
                torch.full((batch_size,), SOURCE_IN_BATCH),
                torch.full((batch_size * num_other,), SOURCE_OTHER),
                torch.full((num_uniform,), SOURCE_UNIFORM),
            ]
        ),
        "num_rows": torch.tensor(num_rows),
        "corpus_size": torch.tensor(corpus_size),
    }


def test_improved_gradient_equals_plugin_gradient():
    """The confidence-weighted loss and the plug-in sampled softmax share their gradient.

    d/dtheta of  -sg(1 - P_hat) * (f_p - log Z_hat)  equals  d/dtheta of  -log(e^{f_p} / (e^{f_p} + Z_hat))
    because 1 - P_hat = Z_hat / (e^{f_p} + Z_hat).
    """
    for seed, q_measure, combine, num_other, strength in itertools.product(
        range(1),
        ["expected_count", "inclusion"],
        ["pooled", "separate"],
        [1],
        [1.0, 0.4],
    ):
        label = make_label(seed=seed, num_other=num_other)
        g = torch.Generator().manual_seed(100 + seed)
        base = torch.randn(12, label["cand_id"].numel(), generator=g, dtype=torch.float64) * 3
        grads = {}
        for variant in ("plugin", "improved"):
            scores = base.clone().requires_grad_()
            loss = mixed_negative_logq_loss(
                scores,
                label,
                variant=variant,
                q_measure=q_measure,
                combine=combine,
                logq_strength=strength,
            ).sum()
            grads[variant] = torch.autograd.grad(loss, scores)[0]

        assert torch.allclose(grads["plugin"], grads["improved"], atol=1e-10), (seed, q_measure, combine)
        assert grads["plugin"].abs().sum() > 0


def test_yi_is_invariant_to_q_scale_but_plugin_is_not():
    """Rescaling every q by a constant (here: via the corpus/row counts) leaves 'yi' unchanged."""
    label = make_label(num_uniform=0)
    scaled = dict(label)
    # In-batch only: q = (B-1) * count / (N-1). Halving N-1 doubles every q.
    scaled["num_rows"] = torch.tensor((int(label["num_rows"]) - 1) // 2 + 1)
    scores = torch.randn(12, label["cand_id"].numel(), dtype=torch.float64)
    yi = [mixed_negative_logq_loss(scores, lab, variant="yi", combine="pooled") for lab in (label, scaled)]
    plugin = [mixed_negative_logq_loss(scores, lab, variant="plugin") for lab in (label, scaled)]

    assert torch.allclose(yi[0], yi[1], atol=1e-4)
    assert not torch.allclose(plugin[0], plugin[1], atol=1e-4)


def test_strength_zero_without_duplicates_is_plain_infonce():
    """lambda = 0 and no accidental hits reduces to MultipleNegativesRankingLoss."""
    from sentence_transformers import util

    batch_size, num_uniform, dim = 8, 5, 16
    g = torch.Generator().manual_seed(0)
    queries = torch.randn(batch_size, dim, generator=g)
    positives = torch.randn(batch_size, dim, generator=g)
    uniform = torch.randn(num_uniform, dim, generator=g)
    label = {
        "query_id": torch.arange(batch_size),
        "cand_id": torch.arange(batch_size + num_uniform) + 1000,
        "cand_count": torch.ones(batch_size + num_uniform),
        "cand_in_corpus": torch.ones(batch_size + num_uniform, dtype=torch.bool),
        "cand_source": torch.cat(
            [torch.full((batch_size,), SOURCE_IN_BATCH), torch.full((num_uniform,), SOURCE_UNIFORM)]
        ),
        "num_rows": torch.tensor(1000),
        "corpus_size": torch.tensor(5000),
    }
    scores = util.cos_sim(queries, torch.cat([positives, uniform])) * 20.0
    ours = mixed_negative_logq_loss(scores, label, variant="plugin", logq_strength=0.0).mean()

    manual = torch.nn.functional.cross_entropy(scores, torch.arange(batch_size))

    assert torch.allclose(ours, manual, atol=1e-6)


def test_known_positives_are_masked():
    """An accidental hit must not change the loss, whatever its score."""
    label = make_label(batch_size=6, num_uniform=4, num_documents=3, num_queries=6)
    scores = torch.randn(6, label["cand_id"].numel(), dtype=torch.float64)
    pos_id = label["cand_id"][:6]
    hits = pos_id[:, None] == label["cand_id"][None, :]
    hits[torch.arange(6), torch.arange(6)] = False  # keep the positive's own score

    assert hits.any(), "test set-up should contain accidental hits"
    for variant in ("plugin", "yi", "improved"):
        a = mixed_negative_logq_loss(scores, label, variant=variant, mask_same_document=True, mask_same_query=False)
        b = mixed_negative_logq_loss(
            scores + 50.0 * hits,
            label,
            variant=variant,
            mask_same_document=True,
            mask_same_query=False,
        )

        assert torch.allclose(a, b)
        c = mixed_negative_logq_loss(scores + 50.0 * hits, label, variant=variant, mask_same_document=False)

        assert not torch.allclose(a, c)


def test_chunked_rows_match_full_batch():
    label = make_label()
    scores = torch.randn(12, label["cand_id"].numel(), dtype=torch.float64)
    for combine in ("pooled", "separate"):
        full = mixed_negative_logq_loss(scores, label, variant="plugin", combine=combine, q_measure="inclusion")
        rows = torch.arange(5, 9)
        part = mixed_negative_logq_loss(
            scores[rows],
            label,
            rows=rows,
            variant="plugin",
            combine=combine,
            q_measure="inclusion",
        )

        assert torch.allclose(full[rows], part)
