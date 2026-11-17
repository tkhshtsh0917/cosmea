"""Small independent oracles for source grouping, preprocessing and invalid scores."""

import math

import pytest
import torch

from cosmea import DocumentFrequency, MixedNegativeDataCollator, stable_id
from cosmea.functional import known_positive_mask, mixed_negative_logq_loss, negative_log_partition


def metadata(ids, counts, sources, queries, *, total=100, corpus_size=10):
    return {
        "query_id": torch.tensor(queries),
        "cand_id": torch.tensor(ids),
        "cand_count": torch.tensor(counts),
        "cand_source": torch.tensor(sources),
        "cand_in_corpus": torch.tensor([True] * len(ids)),
        "num_rows": torch.tensor(total),
        "corpus_size": torch.tensor(corpus_size),
    }


@pytest.mark.parametrize("combine", ["pooled", "separate"])
def test_singleton_with_only_mined_negatives_is_uncorrected_softmax(combine):
    lab = metadata([0, 5], [3, 0], [0, 2], [0], corpus_size=0)
    lab["cand_in_corpus"].fill_(False)
    scores = torch.tensor([[2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    actual = mixed_negative_logq_loss(scores, lab, combine=combine)
    actual.sum().backward()

    assert actual.item() == pytest.approx(math.log1p(math.exp(-1)))
    assert scores.grad.tolist()[0] == pytest.approx([-1 / (1 + math.e), 1 / (1 + math.e)])


def test_separate_extras_have_an_independent_loss_and_frequency_invariant_gradient():
    lab = metadata([0, 1, 7, 8], [1, 4, 0, 0], [0, 0, 2, 2], [0, 1])
    gradients = []
    for total in (100, 1000000):
        lab["num_rows"] = torch.tensor(total)
        scores = torch.tensor([[2.0, 0.0, 1.0, -1.0]], dtype=torch.float64, requires_grad=True)
        value = mixed_negative_logq_loss(scores, lab, rows=torch.tensor([0]))
        in_batch = math.log1p(math.exp(-2) / 4)
        mined = math.log1p(math.exp(-1) + math.exp(-3))

        assert value.item() == pytest.approx((in_batch + mined) / 2)
        gradients.append(torch.autograd.grad(value.sum(), scores)[0][0, 2:])

    assert torch.allclose(*gradients)


@pytest.mark.parametrize("variant,combine", [("yi", "separate"), ("yi", "pooled"), ("plugin", "pooled")])
def test_loss_value_matches_handwritten_two_source_formula(variant, combine):
    lab = metadata([0, 1, 2], [2, 4, 0], [0, 0, 1], [0, 1])
    scores = torch.tensor([[2.0, 0.5, -0.3]], dtype=torch.float64)
    if variant == "yi" and combine == "separate":
        expected = (math.log1p(math.exp(-1.5) * 2 / 4) + math.log1p(math.exp(-2.3))) / 2
    elif variant == "yi":
        expected = math.log1p((math.exp(0.5) / 0.14 + math.exp(-0.3) / 0.1) / (math.exp(2) / 0.12))
    else:
        expected = math.log1p((math.exp(0.5) / (4 / 99 + 0.1) + math.exp(-0.3) / 0.1) / math.exp(2))

    actual = mixed_negative_logq_loss(scores, lab, rows=torch.tensor([0]), variant=variant, combine=combine)

    assert actual.item() == pytest.approx(expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_fail_visibly_even_if_the_candidate_would_be_masked(bad):
    lab = metadata([0, 1], [1, 1], [0, 0], [0, 1])
    for index in (0, 1):
        scores = torch.zeros(1, 2)
        scores[0, index] = bad
        with pytest.raises(ValueError, match="scores must be finite"):
            mixed_negative_logq_loss(scores, lab, rows=torch.tensor([0]))
        with pytest.raises(ValueError, match="scores must be finite"):
            negative_log_partition(scores, lab, rows=torch.tensor([0]))


@pytest.mark.parametrize("same_document,same_query", [(True, True), (True, False), (False, True), (False, False)])
def test_chunk_mask_matches_set_membership_oracle(same_document, same_query):
    lab = metadata([1, 2, 3, 2, 3, 8], [1] * 6, [0, 0, 0, 2, 1, 1], [4, 4, 7])
    rows = torch.tensor([2, 0])
    expected = torch.zeros(2, 6, dtype=torch.bool)
    for local_row, row in enumerate(rows.tolist()):
        matching_rows = {i for i, q in enumerate([4, 4, 7]) if same_query and q == [4, 4, 7][row]} | {row}
        positive_ids = {lab["cand_id"][i].item() for i in matching_rows}
        for column, document in enumerate(lab["cand_id"].tolist()):
            expected[local_row, column] = (document in positive_ids) if same_document else column in matching_rows

    assert torch.equal(known_positive_mask(lab, rows, same_document, same_query), expected)


def test_uniform_preprocessing_matches_positive_prompt_task_and_length():
    calls = []

    def preprocess(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess,
        document_frequency=DocumentFrequency.from_ids(["document"]),
        corpus=["document", "unclicked"],
        num_uniform=2,
        prompts={"anchor": "query: ", "positive": "passage: "},
        router_mapping={"anchor": "query", "positive": "document"},
        max_length={"query": 12, "document": 24},
    )
    batch = collator([{"anchor": "question", "positive": "document"}])

    assert calls[0][1] == {"prompt": "query: ", "task": "query", "max_length": 12}
    assert calls[1][1] == calls[2][1] == {"prompt": "passage: ", "task": "document", "max_length": 24}
    assert batch["uniform_negative_task"] == "document"
    assert set(batch["label"]["cand_id"][1:].tolist()) == {stable_id("document"), stable_id("unclicked")}


@pytest.mark.parametrize("in_batch_strength,uniform_strength", [(0.0, None), (None, 0.0), (0.25, 0.75)])
def test_separate_strengths_match_independent_plugin_softmaxes(in_batch_strength, uniform_strength):
    lab = metadata([0, 1, 2], [2, 4, 0], [0, 0, 1], [0, 1])
    scores = torch.tensor([[2.0, 0.5, -0.3]], dtype=torch.float64, requires_grad=True)
    shared = 0.5
    alpha = shared if in_batch_strength is None else in_batch_strength
    beta = shared if uniform_strength is None else uniform_strength
    in_batch_margin = -1.5 - alpha * math.log(4 / 99)
    uniform_margin = -2.3 - beta * math.log(1 / 10)
    expected = (math.log1p(math.exp(in_batch_margin)) + math.log1p(math.exp(uniform_margin))) / 2

    actual = mixed_negative_logq_loss(
        scores,
        lab,
        rows=torch.tensor([0]),
        variant="plugin",
        combine="separate",
        logq_strength=shared,
        in_batch_logq_strength=in_batch_strength,
        uniform_logq_strength=uniform_strength,
    )

    assert actual.item() == pytest.approx(expected)
    actual.sum().backward()
    in_batch_grad = 0.5 / (1 + math.exp(-in_batch_margin))
    uniform_grad = 0.5 / (1 + math.exp(-uniform_margin))
    assert scores.grad.tolist()[0] == pytest.approx([-in_batch_grad - uniform_grad, in_batch_grad, uniform_grad])


def test_uniform_only_correction_uses_corpus_proposal_without_row_frequency():
    # A valid uniform proposal with no in-batch frequency supplied for candidate 1.
    lab = metadata([0, 1, 2], [2, 0, 0], [0, 0, 1], [0, 1])
    scores = torch.tensor([[2.0, 0.5, -0.3]], dtype=torch.float64)
    expected = (math.log1p(math.exp(-1.5)) + math.log1p(10 * math.exp(-2.3))) / 2

    for q_measure in ("expected_count", "inclusion"):
        actual = mixed_negative_logq_loss(
            scores,
            lab,
            rows=torch.tensor([0]),
            variant="plugin",
            combine="separate",
            q_measure=q_measure,
            in_batch_logq_strength=0,
            uniform_logq_strength=1,
        )
        assert actual.item() == pytest.approx(expected)


def test_yi_separate_uniform_correction_cancels_on_supported_positives():
    lab = metadata([0, 1, 2], [2, 4, 0], [0, 0, 1], [0, 1])
    scores = torch.tensor([[2.0, 0.5, -0.3]], dtype=torch.float64)
    expected = (math.log1p(math.exp(-1.5)) + math.log1p(math.exp(-2.3))) / 2

    loss = mixed_negative_logq_loss(
        scores, lab, rows=torch.tensor([0]), in_batch_logq_strength=0, uniform_logq_strength=1
    )
    assert loss.item() == pytest.approx(expected)

    lab["cand_in_corpus"][0] = False
    with pytest.raises(ValueError, match="every positive ID"):
        mixed_negative_logq_loss(scores, lab, rows=torch.tensor([0]), logq_strength=0, uniform_logq_strength=1)
    loss = mixed_negative_logq_loss(scores, lab, rows=torch.tensor([0]), uniform_logq_strength=0)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "options,pattern",
    [
        ({"combine": "pooled", "in_batch_logq_strength": 0}, 'require combine="separate"'),
        ({"variant": "plugin", "uniform_logq_strength": 1}, 'require combine="separate"'),
        ({"in_batch_logq_strength": -0.5}, "in_batch_logq_strength must be finite"),
        ({"uniform_logq_strength": float("nan")}, "uniform_logq_strength must be finite"),
    ],
)
def test_invalid_source_strengths_rejected_at_both_entry_points(options, pattern):
    from cosmea import LogQMultipleNegativesRankingLoss

    lab = metadata([0, 1], [2, 4], [0, 0], [0, 1])
    with pytest.raises(ValueError, match=pattern):
        mixed_negative_logq_loss(torch.zeros(2, 2), lab, **options)
    with pytest.raises(ValueError, match=pattern):
        LogQMultipleNegativesRankingLoss(torch.nn.Identity(), **options)
