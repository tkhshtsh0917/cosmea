"""Integration with Sentence Transformers: collator output, plain vs. cached loss, worker RNG."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import train_example  # noqa: E402

from cosmea import (  # noqa: E402
    CachedLogQMultipleNegativesRankingLoss,
    ItemFrequency,
    LogQMultipleNegativesRankingLoss,
    MixedNegativeDataCollator,
)
from cosmea.functional import SOURCE_IN_BATCH, SOURCE_UNIFORM  # noqa: E402

from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    workdir = Path(tempfile.mkdtemp())
    model = train_example.build_tiny_model(workdir)
    train_dataset, _, jobs = train_example.build_data()
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        item_frequency=ItemFrequency.from_dataset(train_dataset, item_id_column="job_id"),
        corpus=[job["text"] for job in jobs],
        corpus_ids=[job["job_id"] for job in jobs],
        num_uniform=10,
        query_id_column="query_id",
        item_id_column="job_id",
    )
    return model, train_dataset, jobs, collator


def split_features(batch):
    prefixes = ["query_", "job_", "uniform_negative_"]
    return [{k[len(p) :]: v for k, v in batch.items() if k.startswith(p)} for p in prefixes]


def test_collator_output(setup):
    model, train_dataset, jobs, collator = setup
    rows = [train_dataset[i] for i in range(24)]
    batch = collator(rows)
    label = batch["label"]

    assert batch["query_input_ids"].shape[0] == 24 and batch["job_input_ids"].shape[0] == 24
    assert batch["uniform_negative_input_ids"].shape[0] == 10
    assert not any(key.startswith(("query_id", "job_id")) for key in batch), "id columns must not be tokenized"
    assert label["cand_id"].tolist()[:24] == [row["job_id"] for row in rows]
    assert label["cand_source"].tolist() == [SOURCE_IN_BATCH] * 24 + [SOURCE_UNIFORM] * 10
    uniform_ids = label["cand_id"][24:].tolist()

    assert len(set(uniform_ids)) == 10, "uniform negatives are drawn without replacement"
    # The appended texts are the texts of the sampled ids.
    expected = model.preprocess([jobs[i]["text"] for i in uniform_ids])["input_ids"]

    assert torch.equal(batch["uniform_negative_input_ids"], expected)
    frequency = collator.item_frequency

    assert label["cand_count"].tolist() == [float(frequency.count(i)) for i in label["cand_id"].tolist()]
    assert int(label["num_rows"]) == len(train_dataset) and int(label["corpus_size"]) == len(jobs)
    assert label["cand_in_corpus"].all()


def test_cached_loss_matches_plain_loss(setup):
    model, train_dataset, _, collator = setup
    batch = collator([train_dataset[i] for i in range(24)])
    model.eval()  # dropout would make two forward passes differ
    for kwargs in (
        dict(),
        dict(variant="plugin"),
        dict(variant="yi", combine="pooled"),
        dict(variant="improved", q_measure="inclusion"),
    ):
        values, grads = [], []
        for loss_cls, extra in (
            (LogQMultipleNegativesRankingLoss, {}),
            (CachedLogQMultipleNegativesRankingLoss, {"mini_batch_size": 7}),
        ):
            model.zero_grad()
            loss = loss_cls(model, **kwargs, **extra)(split_features(batch), batch["label"])
            loss.backward()
            values.append(loss.item())
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))

        assert values[0] == pytest.approx(values[1], rel=1e-5), kwargs
        # float32 round-off: gradients are O(1), differences O(1e-6)
        assert torch.allclose(grads[0], grads[1], atol=5e-5, rtol=1e-4), kwargs


def test_improved_and_plugin_give_the_same_model_gradient(setup):
    model, train_dataset, _, collator = setup
    batch = collator([train_dataset[i] for i in range(24)])
    model.eval()
    grads = []
    for variant in ("plugin", "improved"):
        model.zero_grad()
        LogQMultipleNegativesRankingLoss(model, variant=variant)(split_features(batch), batch["label"]).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))

    assert torch.allclose(grads[0], grads[1], atol=5e-5, rtol=1e-4)


@pytest.mark.workers
def test_workers_draw_different_uniform_negatives(setup):
    _, train_dataset, _, collator = setup
    loader = DataLoader(train_dataset, batch_size=16, collate_fn=collator, num_workers=2, shuffle=False)
    iterator = iter(loader)
    first, second = next(iterator), next(iterator)  # one batch from each worker

    assert first["label"]["cand_id"][16:].tolist() != second["label"]["cand_id"][16:].tolist()


def test_loss_rejects_missing_metadata(setup):
    model, train_dataset, _, collator = setup
    batch = collator([train_dataset[i] for i in range(8)])
    with pytest.raises(ValueError, match="MixedNegativeDataCollator"):
        LogQMultipleNegativesRankingLoss(model)(split_features(batch), None)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "variant,measure",
    [
        ("yi", "expected_count"),
        ("plugin", "expected_count"),
        ("improved", "expected_count"),
        ("plugin", "inclusion"),
        ("improved", "inclusion"),
    ],
)
def test_matryoshka_matches_explicit_dimension_losses(setup, cached, variant, measure):
    """The wrapper must preserve dict metadata, dimension weights and model gradients."""
    model, dataset, _, collator = setup
    model.eval()
    batch = collator([dataset[i] for i in range(8)])
    dimensions = [32, 16, 8]
    weights = [1.0, 0.5, 0.25]
    reference = LogQMultipleNegativesRankingLoss(model, variant=variant, q_measure=measure)

    model.zero_grad()
    embeddings = [model(features)["sentence_embedding"] for features in split_features(batch)]
    expected = torch.stack(
        [
            weight
            * reference.compute_loss_from_embeddings(
                [torch.nn.functional.normalize(value[:, :dim], dim=-1) for value in embeddings],
                batch["label"],
            )
            for dim, weight in zip(dimensions, weights)
        ]
    ).sum()
    expected.backward()
    expected_gradient = torch.cat([p.grad.flatten().clone() for p in model.parameters() if p.grad is not None])

    loss_class = CachedLogQMultipleNegativesRankingLoss if cached else LogQMultipleNegativesRankingLoss
    base = loss_class(model, variant=variant, q_measure=measure, **({"mini_batch_size": 3} if cached else {}))
    loss = MatryoshkaLoss(
        model=model,
        loss=base,
        matryoshka_dims=dimensions,
        matryoshka_weights=weights,
    )

    model.zero_grad()
    actual = loss(split_features(batch), batch["label"])
    actual.backward()
    actual_gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    assert actual.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(actual_gradient, expected_gradient, atol=5e-5, rtol=1e-4)

    with torch.no_grad():
        evaluation_loss = loss(split_features(batch), batch["label"])

    assert evaluation_loss.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("matryoshka", [False, True])
def test_trainer_consumes_metadata_and_updates_model(tmp_path, cached, matryoshka):
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    model = train_example.build_tiny_model(tmp_path)
    dataset, _, jobs = train_example.build_data()
    dataset = dataset.select(range(16))
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        item_frequency=ItemFrequency.from_dataset(dataset, item_id_column="job_id"),
        corpus=[job["text"] for job in jobs],
        corpus_ids=[job["job_id"] for job in jobs],
        num_uniform=3,
        query_id_column="query_id",
        item_id_column="job_id",
    )
    loss_cls = CachedLogQMultipleNegativesRankingLoss if cached else LogQMultipleNegativesRankingLoss
    loss = loss_cls(model, variant="plugin", **({"mini_batch_size": 3} if cached else {}))
    if matryoshka:
        loss = MatryoshkaLoss(model=model, loss=loss, matryoshka_dims=[32, 16, 8])

    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        use_cpu=True,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
        loss=loss,
    )
    # Same dataset for interface smoke only; this is not held-out model evaluation.
    result = trainer.train()
    metrics = trainer.evaluate()
    after = torch.cat([p.detach().flatten() for p in model.parameters()])

    assert torch.isfinite(after).all() and not torch.equal(before, after)
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
