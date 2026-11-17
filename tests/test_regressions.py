"""Independent finite-population oracles and API edge cases discovered in review."""

import itertools

import pytest
import torch

from cosmea import DocumentFrequency, MixedNegativeDataCollator, stable_id
from cosmea.functional import candidate_log_q, mixed_negative_logq_loss, negative_log_partition


def label(ids, counts, n, uniform=(), corpus_size=0, query_ids=None):
    candidates = [*ids, *uniform]
    return {
        "query_id": torch.tensor(query_ids if query_ids is not None else list(range(len(ids)))),
        "cand_id": torch.tensor(candidates),
        "cand_count": torch.tensor([counts.get(i, 0) for i in candidates]),
        "cand_source": torch.tensor([0] * len(ids) + [1] * len(uniform)),
        "cand_in_corpus": torch.tensor([0 <= i < corpus_size for i in candidates]),
        "num_rows": torch.tensor(n),
        "corpus_size": torch.tensor(corpus_size),
    }


def test_exact_finite_population_inclusion():
    lab = label([0, 1, 2], {0: 1, 1: 1, 2: 2}, n=4)
    q = candidate_log_q(lab, "inclusion", False, rows=torch.tensor([0])).exp()[0]

    assert q.tolist() == pytest.approx([0.0, 2 / 3, 1.0])


def test_iid_inclusion_helper_uses_binomial_probability():
    lab = label([0, 1, 2], {0: 1, 1: 1, 2: 2}, n=4)
    q = candidate_log_q(lab, "inclusion", False, iid_proposal=True).exp()

    assert q.tolist() == pytest.approx([7 / 16, 7 / 16, 3 / 4])


@pytest.mark.parametrize("measure", ["expected_count", "inclusion"])
def test_partition_matches_exhaustive_sampling_expectation(measure):
    # Fixed first row, then two of four remaining rows; independent 2-of-5 corpus draw.
    population = [0, 0, 1, 2, 3]
    counts = {0: 2, 1: 1, 2: 1, 3: 1}
    document_scores = torch.tensor([0.5, -0.2, 1.2, 0.8, -0.7], dtype=torch.float64)
    values = []
    for others in itertools.combinations(range(1, 5), 2):
        for uniform in itertools.combinations(range(5), 2):
            lab = label([0, *(population[i] for i in others)], counts, 5, uniform, 5)
            scores = document_scores[lab["cand_id"]][None, :]
            values.append(
                negative_log_partition(
                    scores,
                    lab,
                    rows=torch.tensor([0]),
                    q_measure=measure,
                    mask_same_query=False,
                ).exp()[0]
            )

    assert torch.stack(values).mean().item() == pytest.approx(document_scores[1:].exp().sum().item(), abs=1e-12)


@pytest.mark.parametrize(
    "variant,dtype",
    [("yi", torch.float16), ("plugin", torch.bfloat16), ("improved", torch.float32), ("yi", torch.float64)],
)
def test_empty_negative_pool_is_finite_zero_gradient(variant, dtype):
    lab = label([7], {7: 1}, 1)
    scores = torch.tensor([[900.0]], dtype=dtype, requires_grad=True)
    value = mixed_negative_logq_loss(scores, lab, variant=variant).sum()
    value.backward()

    assert value.item() == 0
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_singleton_uniform_source_is_not_halved():
    lab = label([0], {0: 1}, 1, uniform=[1], corpus_size=2)
    scores = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    actual = mixed_negative_logq_loss(scores, lab, variant="yi", mask_same_query=False)

    assert actual.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(1.0)).item())


def test_positive_outside_uniform_proposal_is_rejected():
    lab = label([3, 4], {3: 1, 4: 1}, 2, uniform=[0], corpus_size=2)
    with pytest.raises(ValueError, match="every positive ID"):
        mixed_negative_logq_loss(torch.zeros(2, 3), lab)


def test_unsupported_yi_inclusion_is_rejected():
    with pytest.raises(ValueError, match="yi .* inclusion"):
        mixed_negative_logq_loss(torch.zeros(1, 1), label([0], {0: 1}, 1), q_measure="inclusion")


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1.0])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="finite and non-negative"):
        mixed_negative_logq_loss(torch.zeros(1, 1), label([0], {0: 1}, 1), logq_strength=strength)


def test_frequency_validation_and_id_bounds():
    with pytest.raises(ValueError):
        DocumentFrequency.from_ids([])
    with pytest.raises(ValueError):
        DocumentFrequency(counts={1: 2}, num_rows=3)
    with pytest.raises(ValueError):
        stable_id(2**63)
    with pytest.raises(TypeError):
        stable_id(None)

    assert stable_id("job") == stable_id("job")


def preprocess(texts, **kwargs):
    return {"input_ids": torch.arange(len(texts))[:, None]}


def test_hard_negative_ids_use_same_identity_namespace():
    freq = DocumentFrequency.from_ids([10, 20])
    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess,
        document_frequency=freq,
        document_id_column="pid",
        extra_document_id_columns={"negative": "nid"},
    )
    batch = collator([{"anchor": "q", "positive": "p", "negative": "another rendering", "pid": 10, "nid": 10}])

    assert batch["label"]["cand_id"].tolist() == [10, 10]
    assert batch["label"]["cand_count"].dtype == torch.long
    with pytest.raises(ValueError, match="extra_document_id_columns"):
        MixedNegativeDataCollator(preprocess_fn=preprocess, document_frequency=freq, document_id_column="pid")(
            [{"anchor": "q", "positive": "p", "negative": "n", "pid": 10}]
        )


def test_unknown_positive_frequency_is_rejected():
    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess, document_frequency=DocumentFrequency.from_ids(["other"])
    )
    with pytest.raises(ValueError, match="absent from document_frequency"):
        collator([{"anchor": "q", "positive": "not counted"}])


def test_duplicate_corpus_ids_sample_distinct_documents():
    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess,
        document_frequency=DocumentFrequency.from_ids([0]),
        document_id_column="pid",
        corpus=["a", "a", "b"],
        corpus_ids=[0, 0, 1],
        num_uniform=2,
    )
    batch = collator([{"anchor": "q", "positive": "a", "pid": 0}])

    assert collator.corpus_size == 2
    assert set(batch["label"]["cand_id"][1:].tolist()) == {0, 1}


@pytest.mark.parametrize("measure", ["expected_count", "inclusion"])
@pytest.mark.parametrize("mask_positive", [True, False])
@pytest.mark.parametrize("sources", [None, (0,), (1,)])
def test_exact_support_with_outside_positive_and_unclicked_document(measure, mask_positive, sources):
    population = [4, 4, 1, 2]  # positive 4 is outside the corpus; corpus document 0 is unclicked.
    counts = {4: 2, 1: 1, 2: 1}
    document_scores = torch.tensor([-0.5, 0.2, 0.7, 0.0, 1.1], dtype=torch.float64)
    estimates = []
    for other in range(1, 4):
        for uniform in range(3):
            lab = label([4, population[other]], counts, 4, [uniform], 3)
            estimates.append(
                negative_log_partition(
                    document_scores[lab["cand_id"]][None, :],
                    lab,
                    rows=torch.tensor([0]),
                    q_measure=measure,
                    mask_same_document=mask_positive,
                    mask_same_query=False,
                    sources=sources,
                ).exp()[0]
            )
    support = {0, 1, 2, 4} if sources is None else ({1, 2, 4} if sources == (0,) else {0, 1, 2})
    if mask_positive:
        support.discard(4)

    assert torch.stack(estimates).mean().item() == pytest.approx(
        document_scores[sorted(support)].exp().sum().item(), abs=1e-12
    )


def test_legacy_document_keywords_keep_existing_training_rows_working():
    from cosmea import ItemFrequency

    dataset = {"item_id": [7, 7]}
    frequency = ItemFrequency.from_dataset(dataset, item_id_column="item_id")
    assert frequency.count(item_id=7) == 2
    assert frequency == DocumentFrequency.from_ids(item_ids=[7, 7])

    def preprocess(texts, **kwargs):
        return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess,
        item_frequency=frequency,
        query_id_column="query_id",
        item_id_column="item_id",
        extra_item_id_columns={"negative": "negative_id"},
    )
    batch = collator([{"query": "q", "positive": "p", "negative": "n", "query_id": 1, "item_id": 7, "negative_id": 8}])

    assert batch["label"]["cand_id"].tolist() == [7, 8]
    scores = torch.tensor([[2.0, 1.0]])
    torch.testing.assert_close(
        mixed_negative_logq_loss(scores, batch["label"], mask_same_item=True),
        mixed_negative_logq_loss(scores, batch["label"], mask_same_document=True),
    )
    with pytest.raises(ValueError, match="must agree"):
        mixed_negative_logq_loss(scores, batch["label"], mask_same_document=True, mask_same_item=False)
