"""Integration with Sentence Transformers: collator output, plain vs. cached loss, worker RNG."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

spec = importlib.util.spec_from_file_location(
    "train_example", Path(__file__).resolve().parents[1] / "examples/train_example.py"
)
train_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_example)

from cosmea import (  # noqa: E402
    CachedLogQMultipleNegativesRankingLoss,
    DocumentFrequency,
    LogQMultipleNegativesRankingLoss,
    MixedNegativeDataCollator,
)
from cosmea.functional import SOURCE_IN_BATCH, SOURCE_UNIFORM  # noqa: E402

from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    workdir = tmp_path_factory.mktemp("model")
    model = train_example.build_tiny_model(workdir)
    train_dataset, _, jobs = train_example.build_data()
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=DocumentFrequency.from_dataset(train_dataset, document_id_column="document_id"),
        corpus=[job["text"] for job in jobs],
        corpus_ids=[job["document_id"] for job in jobs],
        num_uniform=10,
        query_id_column="query_id",
        document_id_column="document_id",
    )
    return model, train_dataset, jobs, collator


def split_features(batch):
    prefixes = ["anchor_", "positive_", "negative_", "uniform_negative_"]
    features = [{k[len(p) :]: v for k, v in batch.items() if k.startswith(p)} for p in prefixes]
    return [column for column in features if column]


def test_collator_output(setup):
    model, train_dataset, jobs, collator = setup
    rows = [train_dataset[i] for i in range(24)]
    batch = collator(rows)
    label = batch["label"]

    assert batch["anchor_input_ids"].shape[0] == 24 and batch["positive_input_ids"].shape[0] == 24
    assert batch["uniform_negative_input_ids"].shape[0] == 10
    assert not any(key.startswith(("query_id", "document_id")) for key in batch), "id columns must not be tokenized"
    assert label["cand_id"].tolist()[:24] == [row["document_id"] for row in rows]
    assert label["cand_source"].tolist() == [SOURCE_IN_BATCH] * 24 + [SOURCE_UNIFORM] * 10
    uniform_ids = label["cand_id"][24:].tolist()

    assert len(set(uniform_ids)) == 10, "uniform negatives are drawn without replacement"
    # The appended texts are the texts of the sampled ids.
    expected = model.preprocess([jobs[i]["text"] for i in uniform_ids])["input_ids"]

    assert torch.equal(batch["uniform_negative_input_ids"], expected)
    frequency = collator.document_frequency

    assert label["cand_count"].tolist() == [float(frequency.count(i)) for i in label["cand_id"].tolist()]
    assert int(label["num_rows"]) == len(train_dataset) and int(label["corpus_size"]) == len(jobs)
    assert label["cand_in_corpus"].all()


def test_cached_loss_matches_plain_loss(setup):
    model, train_dataset, jobs, collator = setup
    rows = []
    for index in range(24):
        row = dict(train_dataset[index])
        negative_id = (row["document_id"] + 1) % len(jobs)
        row.update(negative=jobs[negative_id]["text"], negative_id=negative_id)
        rows.append(row)
    collator = replace(collator, extra_document_id_columns={"negative": "negative_id"})
    batch = collator(rows)
    model.eval()  # dropout would make two forward passes differ
    for kwargs in (
        dict(),
        dict(variant="improved", q_measure="inclusion"),
        dict(variant="plugin", combine="separate", in_batch_logq_strength=0, uniform_logq_strength=1),
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


@pytest.mark.workers
def test_workers_draw_different_uniform_negatives(setup):
    _, train_dataset, _, collator = setup
    loader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=collator,
        num_workers=2,
        shuffle=False,
        multiprocessing_context="spawn",
    )
    iterator = iter(loader)
    first, second = next(iterator), next(iterator)  # one batch from each worker

    assert first["label"]["cand_id"][16:].tolist() != second["label"]["cand_id"][16:].tolist()


def test_loss_rejects_missing_metadata(setup):
    model, train_dataset, _, collator = setup
    batch = collator([train_dataset[i] for i in range(8)])
    with pytest.raises(ValueError, match="MixedNegativeDataCollator"):
        LogQMultipleNegativesRankingLoss(model)(split_features(batch), None)


@pytest.mark.parametrize("cached", [False, True])
def test_matryoshka_matches_explicit_dimension_losses(setup, cached):
    """The wrapper must preserve dict metadata, dimension weights and model gradients."""
    variant, measure = "plugin", "expected_count"
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
def test_trainer_consumes_metadata_and_updates_model(tmp_path, cached):
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    model = train_example.build_tiny_model(tmp_path)
    dataset, _, jobs = train_example.build_data()
    dataset = dataset.select(range(9))
    negative_ids = [(int(document) + 1) % len(jobs) for document in dataset["document_id"]]
    dataset = dataset.add_column("negative", [jobs[document]["text"] for document in negative_ids])
    dataset = dataset.add_column("negative_id", negative_ids)
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=DocumentFrequency.from_dataset(dataset, document_id_column="document_id"),
        corpus=[job["text"] for job in jobs],
        corpus_ids=[job["document_id"] for job in jobs],
        num_uniform=3,
        extra_document_id_columns={"negative": "negative_id"},
        query_id_column="query_id",
        document_id_column="document_id",
    )
    loss_cls = CachedLogQMultipleNegativesRankingLoss if cached else LogQMultipleNegativesRankingLoss
    loss = loss_cls(model, **({"mini_batch_size": 3} if cached else {}))
    if cached:
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


def test_retrieval_evaluator_accepts_documents_missing_from_training_counts(setup, tmp_path):
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    model, _, _, collator = setup
    evaluator = InformationRetrievalEvaluator(
        queries={"held_out_query": "query nurse"},
        corpus={"unseen_document": "job nurse", "other_document": "job driver"},
        relevant_docs={"held_out_query": {"unseen_document"}},
        name="held_out",
        show_progress_bar=False,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / "evaluation"),
            use_cpu=True,
            report_to="none",
            disable_tqdm=True,
        ),
        loss=LogQMultipleNegativesRankingLoss(model),
        data_collator=collator,
        evaluator=evaluator,
    )
    metrics = trainer.evaluate()

    assert "eval_loss" not in metrics
    assert torch.isfinite(torch.tensor(metrics[f"eval_{evaluator.primary_metric}"]))


@pytest.mark.parametrize(
    "headers",
    ["anchor,positive,query_id,document_id", "query,positive,query_id,item_id", "query,document,query_id,document_id"],
)
def test_training_csv_headers_map_to_the_same_training_roles(tmp_path, headers):
    spec = importlib.util.spec_from_file_location(
        "train_from_csv", Path(__file__).resolve().parents[1] / "examples/train_from_csv.py"
    )
    train_from_csv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_from_csv)
    path = tmp_path / "train.csv"
    path.write_text(f"{headers}\nquery text,document text,01,007\n", encoding="utf-8")

    assert train_from_csv.read_training_rows(path) == [
        {"anchor": "query text", "positive": "document text", "query_id": "01", "document_id": "007"}
    ]


@pytest.mark.parametrize("cached", [False, True])
def test_trainer_supports_no_duplicates_with_uniform_only_correction(tmp_path, cached):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.base.sampler import BatchSamplers

    model = train_example.build_tiny_model(tmp_path)
    rows = [
        {"anchor": "query nurse", "positive": "job nurse", "query_id": "q:1", "document_id": "d:1"},
        {"anchor": "query city", "positive": "job nurse", "query_id": "q:2", "document_id": "d:1"},
        {"anchor": "query driver", "positive": "job driver", "query_id": "q:3", "document_id": "d:2"},
    ]
    dataset = Dataset.from_list(rows)
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=DocumentFrequency.from_dataset(dataset, document_id_column="document_id"),
        query_id_column="query_id",
        document_id_column="document_id",
        corpus=["job nurse", "job driver", "job teacher"],
        corpus_ids=["d:1", "d:2", "d:3"],
        num_uniform=1,
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=True,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    loss_cls = CachedLogQMultipleNegativesRankingLoss if cached else LogQMultipleNegativesRankingLoss
    loss = loss_cls(
        model,
        variant="plugin",
        combine="separate",
        in_batch_logq_strength=0,
        uniform_logq_strength=1,
        mask_same_document=True,
        **({"mini_batch_size": 1} if cached else {}),
    )
    assert loss.get_config_dict()["in_batch_logq_strength"] == 0
    assert loss.get_config_dict()["uniform_logq_strength"] == 1
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        loss=loss,
    )

    label = next(iter(trainer.get_train_dataloader()))["label"]
    assert label["cand_id"][label["cand_source"] == SOURCE_IN_BATCH].unique().numel() == 2
    assert (label["cand_source"] == SOURCE_UNIFORM).sum() == 1
    assert torch.isfinite(torch.tensor(trainer.train().training_loss))
