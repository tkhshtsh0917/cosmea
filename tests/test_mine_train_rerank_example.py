"""Check ID-based evaluation and run the example with local tiny checkpoints."""

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertConfig, BertForSequenceClassification

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import mine_train_rerank

    return mine_train_rerank


def test_mining_and_reranking_use_document_ids(example):
    corpus = {"007": "positive", "7": "positive", "008": "relevant", "009": "unjudged"}
    document_vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query_vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    queries = {
        "q1": example.Query("first", {"007", "008"}),
        "q2": example.Query("second", {"7"}),
    }

    rows = example.mine(queries, [("q1", "007"), ("q1", "008")], corpus, query_vectors, document_vectors)
    assert [row["negative_document_id"] for row in rows] == ["009", "009"]

    class FixedReranker:
        def predict(self, pairs, **kwargs):
            return [1, 0, 2]

    metrics, rankings = example.evaluate(
        queries,
        corpus,
        query_vectors,
        document_vectors,
        FixedReranker(),
        top_n=3,
        cutoff=2,
        batch_size=4,
    )

    assert [(row["document_id"], row["score"]) for row in rankings[0]["before"]] == [
        ("007", 1.0),
        ("7", 1.0),
        ("008", 0.0),
    ]
    assert [(row["document_id"], row["score"]) for row in rankings[0]["after"]] == [
        ("008", 2.0),
        ("007", 1.0),
        ("7", 0.0),
    ]
    assert {row["document_id"] for row in rankings[1]["after"]} == {"009", "008", "007"}
    assert metrics["candidate_recall"] == 0.5
    assert metrics["before"]["recall@2"] == 0.25
    assert metrics["after"]["recall@2"] == metrics["after"]["ndcg@2"] == 0.5
    assert metrics["before"]["mrr@2"] == metrics["after"]["mrr@2"] == 0.5
    assert metrics["before"]["ndcg@2"] == pytest.approx(0.5 / (1 + 1 / math.log2(3)))


def test_mining_chooses_nearest_remaining_document(example):
    corpus = {"a": "match", "b": "farther", "c": "closer"}
    rows = example.mine(
        {"q": example.Query("query", {"a"})},
        [("q", "a")],
        corpus,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]]),
    )
    assert rows[0]["negative_document_id"] == "c"


def test_binary_metrics_honor_cutoff_and_all_known_positives(example):
    multiple = example.ranking_metrics(["a", "x", "b", "c"], {"a", "b", "c"}, 2)
    assert multiple["ndcg@2"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert multiple["recall@2"] == pytest.approx(1 / 3)
    assert example.ranking_metrics(["a", "x"], {"a"}, 2)["ndcg@2"] == 1.0


def test_ranking_ties_follow_input_order(example):
    corpus = {str(index): f"document {index}" for index in range(32)}
    document_vectors = torch.eye(2).repeat(16, 1)

    class TiedReranker:
        def predict(self, pairs, **kwargs):
            return [0.0] * len(pairs)

    _, rankings = example.evaluate(
        {"q": example.Query("query", {"1"})},
        corpus,
        torch.tensor([[0.0, 1.0]]),
        document_vectors,
        TiedReranker(),
        top_n=3,
        cutoff=2,
        batch_size=4,
    )
    for stage in ("before", "after"):
        assert [row["document_id"] for row in rankings[0][stage]] == ["1", "3", "5"]


@pytest.mark.parametrize(
    ("training", "evaluation", "message"),
    [
        ("01,query nurse,001\n02,query nurse,002\n", "03,other,002\n", "same query text"),
        ("01,query nurse,001\n", "02,query nurse,002\n", "must not share IDs or exact text"),
        ("01,query nurse,001\n", "01,another query,002\n", "must not share IDs or exact text"),
        ("01,query nurse,404\n", "02,other,002\n", "absent from the corpus"),
        ('01,"unterminated\n', "02,other,002\n", "unexpected end of data"),
        ("01,query nurse,001\n", None, "eval.csv"),
    ],
)
def test_cli_rejects_bad_csv_before_loading_models(
    example, tmp_path, monkeypatch, capsys, training, evaluation, message
):
    corpus, train, eval_path, output = [tmp_path / name for name in ("corpus.csv", "train.csv", "eval.csv", "run")]
    corpus.write_text("document_id,text\n001,job nurse\n002,job driver\n")
    header = "query_id,query,document_id\n"
    train.write_text(header + training)
    if evaluation is not None:
        eval_path.write_text(header + evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_train_rerank.py",
            "--model",
            "unused",
            "--reranker",
            "unused",
            "--corpus",
            str(corpus),
            "--train",
            str(train),
            "--eval",
            str(eval_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exit_error:
        example.main()
    assert exit_error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr and "Traceback" not in stderr
    assert not output.exists()


def test_cli_trains_and_scores_the_same_candidates(example, tmp_path, monkeypatch):
    from train_example import build_tiny_model

    retriever = build_tiny_model(tmp_path)
    base_path = tmp_path / "tiny-bert"
    encoder_path = tmp_path / "encoder"
    retriever.save_pretrained(encoder_path)
    cross_path = tmp_path / "reranker"
    config = BertConfig.from_pretrained(base_path, num_labels=1)
    BertForSequenceClassification(config).save_pretrained(cross_path)
    retriever.tokenizer.save_pretrained(cross_path)

    corpus = tmp_path / "corpus.csv"
    train = tmp_path / "train.csv"
    evaluation = tmp_path / "eval.csv"
    output = tmp_path / "run"
    corpus.write_text(
        "document_id,text\n001,job nurse\n1,job nurse\n002,job chef\n003,job driver\n", encoding="utf-8-sig"
    )
    train.write_text("query_id,query,document_id\n01,query nurse,001\n01,query nurse,002\n")
    evaluation.write_text("query_id,query,document_id\n02,query driver,003\n")
    preview = tmp_path / "preview"
    common_args = [
        str(EXAMPLES / "mine_train_rerank.py"),
        "--model",
        str(encoder_path),
        "--train",
        str(train),
        "--corpus",
        str(corpus),
        "--query-prompt",
        "query ",
        "--document-prompt",
        "job ",
        "--batch-size",
        "2",
    ]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    preview_run = subprocess.run(
        [sys.executable, *common_args, "--mine-only", "--output", str(preview)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert preview_run.returncode == 0, preview_run.stderr
    assert sorted(path.name for path in preview.iterdir()) == ["mined.csv"]

    trainer_class = example.SentenceTransformerTrainer

    def checked_trainer(**kwargs):
        assert (output / "mined.csv").read_bytes() == (preview / "mined.csv").read_bytes()
        assert kwargs["data_collator"].prompts == {"anchor": "query ", "positive": "job ", "negative": "job "}
        assert kwargs["loss"].mask_same_document and kwargs["loss"].mask_same_query
        return trainer_class(**kwargs)

    monkeypatch.setattr(example, "SentenceTransformerTrainer", checked_trainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common_args,
            "--reranker",
            str(cross_path),
            "--eval",
            str(evaluation),
            "--output",
            str(output),
            "--uniform",
            "1",
            "--candidates",
            "3",
            "--k",
            "2",
        ],
    )
    example.main()

    with (output / "mined.csv").open(newline="", encoding="utf-8") as handle:
        mined = list(csv.DictReader(handle))
    assert mined == [
        {"query_id": "01", "document_id": "001", "negative_document_id": "003"},
        {"query_id": "01", "document_id": "002", "negative_document_id": "003"},
    ]
    ranking = json.loads((output / "rankings.jsonl").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    before = {row["document_id"] for row in ranking["before"]}
    assert len(before) == 3
    assert before == {row["document_id"] for row in ranking["after"]}
    assert ranking["query_id"] == "02" and ranking["relevant_document_ids"] == ["003"]
    assert metrics["candidate_recall"] == float("003" in before)
    saved = SentenceTransformer(str(output / "model"), device="cpu")
    texts = dict(zip(["001", "1", "002", "003"], ["job nurse", "job nurse", "job chef", "job driver"], strict=True))
    ordered = ranking["before"]
    query = saved.encode_query(["query driver"], prompt="query ", convert_to_tensor=True)
    documents = saved.encode_document(
        [texts[row["document_id"]] for row in ordered],
        prompt="job ",
        convert_to_tensor=True,
    )
    expected = torch.nn.functional.cosine_similarity(query, documents)
    torch.testing.assert_close(torch.tensor([row["score"] for row in ordered]), expected)
