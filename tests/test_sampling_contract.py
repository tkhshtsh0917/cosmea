"""Small independent oracles for source grouping, preprocessing and invalid scores."""

import math

import pytest
import torch

from cosmea import ItemFrequency, MixedNegativeDataCollator, stable_id
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


@pytest.mark.parametrize("same_item,same_query", [(True, True), (True, False), (False, True), (False, False)])
def test_chunk_mask_matches_set_membership_oracle(same_item, same_query):
    lab = metadata([1, 2, 3, 2, 3, 8], [1] * 6, [0, 0, 0, 2, 1, 1], [4, 4, 7])
    rows = torch.tensor([2, 0])
    expected = torch.zeros(2, 6, dtype=torch.bool)
    for local_row, row in enumerate(rows.tolist()):
        matching_rows = {i for i, q in enumerate([4, 4, 7]) if same_query and q == [4, 4, 7][row]} | {row}
        positive_ids = {lab["cand_id"][i].item() for i in matching_rows}
        for column, item in enumerate(lab["cand_id"].tolist()):
            expected[local_row, column] = (item in positive_ids) if same_item else column in matching_rows

    assert torch.equal(known_positive_mask(lab, rows, same_item, same_query), expected)


def test_uniform_preprocessing_matches_positive_prompt_task_and_length():
    calls = []

    def preprocess(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    collator = MixedNegativeDataCollator(
        preprocess_fn=preprocess,
        item_frequency=ItemFrequency.from_ids(["document"]),
        corpus=["document", "unclicked"],
        num_uniform=2,
        prompts={"query": "query: ", "positive": "passage: "},
        router_mapping={"query": "query", "positive": "document"},
        max_length={"query": 12, "document": 24},
    )
    batch = collator([{"query": "question", "positive": "document"}])

    assert calls[0][1] == {"prompt": "query: ", "task": "query", "max_length": 12}
    assert calls[1][1] == calls[2][1] == {"prompt": "passage: ", "task": "document", "max_length": 24}
    assert batch["uniform_negative_task"] == "document"
    assert set(batch["label"]["cand_id"][1:].tolist()) == {stable_id("document"), stable_id("unclicked")}
