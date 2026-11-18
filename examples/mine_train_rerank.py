"""Mine ID-linked negatives, train a Cosmea retriever, and evaluate reranking on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from train_from_csv import read_rows
from transformers import set_seed

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator


@dataclass
class Query:
    text: str
    positives: set[str] = field(default_factory=set)


def load_queries(path: Path, corpus: dict[str, str]) -> tuple[dict[str, Query], list[tuple[str, str]]]:
    """Read distinct positive pairs and require one ID per exact query text."""
    rows = read_rows(path, ["query_id", "query", "document_id"], aliases={})
    queries: dict[str, Query] = {}
    ids_by_text: dict[str, str] = {}
    pairs = []
    for row in rows:
        query_id, query_text, document_id = row["query_id"], row["query"], row["document_id"]
        if document_id not in corpus:
            raise ValueError(f"{path.name}: document ID {document_id!r} is absent from the corpus.")
        prior_id = ids_by_text.setdefault(query_text, query_id)
        if prior_id != query_id:
            raise ValueError(
                f"{path.name}: the same query text has IDs {prior_id!r} and {query_id!r}; use one query_id."
            )
        query = queries.setdefault(query_id, Query(query_text))
        if query.text != query_text or document_id in query.positives:
            raise ValueError(f"{path.name}: inconsistent query text or duplicate pair for {query_id!r}.")
        query.positives.add(document_id)
        pairs.append((query_id, document_id))

    return queries, pairs


def encode(
    model: SentenceTransformer,
    texts: list[str],
    *,
    is_query: bool,
    prompt: str,
    batch_size: int,
) -> torch.Tensor:
    """Encode one side as a normalized (number of texts, embedding dimension) matrix."""
    encode_side = model.encode_query if is_query else model.encode_document
    embeddings = encode_side(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    ).float()
    if not torch.isfinite(embeddings).all():
        raise ValueError("The retriever produced nonfinite embeddings.")
    return embeddings


def mine(
    queries: dict[str, Query],
    pairs: list[tuple[str, str]],
    corpus: dict[str, str],
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
) -> list[dict[str, str]]:
    """Choose one cosine-nearest eligible document per query, preserving the original IDs."""
    document_ids = list(corpus)
    negatives = {}
    for query_vector, (query_id, query) in zip(query_vectors, queries.items(), strict=True):
        positive_texts = {corpus[document_id] for document_id in query.positives}
        eligible = [index for index, text in enumerate(corpus.values()) if text not in positive_texts]
        if not eligible:
            raise ValueError(f"No eligible mined negative for query {query_id!r}.")
        similarities = document_vectors @ query_vector
        best = eligible[int(similarities[eligible].argmax())]
        negatives[query_id] = document_ids[best]

    return [
        {
            "anchor": queries[query_id].text,
            "positive": corpus[document_id],
            "negative": corpus[negatives[query_id]],
            "query_id": query_id,
            "document_id": document_id,
            "negative_document_id": negatives[query_id],
        }
        for query_id, document_id in pairs
    ]


def ranking_metrics(ranked_ids: list[str], relevant_ids: set[str], cutoff: int) -> dict[str, float]:
    """Score binary judgments by ID, using *all* known positives in the ideal ranking."""
    hits = [rank for rank, document_id in enumerate(ranked_ids[:cutoff], start=1) if document_id in relevant_ids]
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(relevant_ids)) + 1))
    return {
        f"ndcg@{cutoff}": sum(1 / math.log2(rank + 1) for rank in hits) / ideal,
        f"recall@{cutoff}": len(hits) / len(relevant_ids),
        f"mrr@{cutoff}": 1 / hits[0] if hits else 0.0,
    }


def evaluate(
    queries: dict[str, Query],
    corpus: dict[str, str],
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
    reranker: CrossEncoder,
    *,
    top_n: int,
    cutoff: int,
    batch_size: int,
):
    """Return aggregate metrics and two aligned ranked lists per query, with no positive injection."""
    document_ids = list(corpus)
    scored_queries = []
    totals = {"before": {}, "after": {}}
    candidate_recall = 0.0
    for query_vector, (query_id, query) in zip(query_vectors, queries.items(), strict=True):
        similarities = document_vectors @ query_vector
        order = similarities.argsort(descending=True, stable=True)[:top_n].tolist()
        before_ids = [document_ids[index] for index in order]
        rerank_scores = list(
            reranker.predict(
                [(query.text, corpus[document_id]) for document_id in before_ids],
                batch_size=batch_size,
                activation_fn=torch.nn.Identity(),
                show_progress_bar=False,
            )
        )
        if len(rerank_scores) != len(order) or not all(math.isfinite(score) for score in rerank_scores):
            raise ValueError("The CrossEncoder must return one finite score per query-document pair.")
        rerank_order = sorted(range(len(order)), key=lambda index: -rerank_scores[index])
        after_ids = [before_ids[index] for index in rerank_order]

        candidate_recall += len(query.positives.intersection(before_ids)) / len(query.positives)
        for stage, ids in (("before", before_ids), ("after", after_ids)):
            for name, value in ranking_metrics(ids, query.positives, cutoff).items():
                totals[stage][name] = totals[stage].get(name, 0.0) + value
        scored_queries.append(
            {
                "query_id": query_id,
                "relevant_document_ids": sorted(query.positives),
                "before": [
                    {"document_id": document_id, "score": float(similarities[index])}
                    for index, document_id in zip(order, before_ids, strict=True)
                ],
                "after": [
                    {"document_id": before_ids[index], "score": float(rerank_scores[index])} for index in rerank_order
                ],
            }
        )

    metrics = {
        "num_queries": len(queries),
        "candidate_count": top_n,
        "candidate_recall": candidate_recall / len(queries),
        **{stage: {name: value / len(queries) for name, value in values.items()} for stage, values in totals.items()},
    }
    return metrics, scored_queries


def write_mined(output: Path, rows: list[dict[str, str]]) -> None:
    """Save the original ID triples before training or evaluating any models."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "mined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "document_id", "negative_document_id"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SentenceTransformer checkpoint")
    parser.add_argument("--reranker", help="Pretrained single-score CrossEncoder checkpoint")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mine-only", action="store_true", help="Write mined.csv and stop before training")
    parser.add_argument("--uniform", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=30, help="First-stage depth, capped at corpus size")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for ranking metrics")
    parser.add_argument("--query-prompt", help="Override the retriever's query prompt")
    parser.add_argument("--document-prompt", help="Override the retriever's document prompt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.mine_only and (args.eval is None or args.reranker is None):
        parser.error("--eval and --reranker are required unless --mine-only is set.")
    if args.batch_size < 1 or (not args.mine_only and args.epochs < 1):
        parser.error("Batch size and, when training, epoch count must be positive.")

    try:
        corpus_rows = read_rows(args.corpus, ["document_id", "text"], aliases={})
        corpus = {row["document_id"]: row["text"] for row in corpus_rows}
        if len(corpus) != len(corpus_rows):
            raise ValueError("Corpus document IDs must be unique.")
        train_queries, train_pairs = load_queries(args.train, corpus)
        if not args.mine_only:
            eval_queries, _ = load_queries(args.eval, corpus)
            if train_queries.keys() & eval_queries.keys() or {query.text for query in train_queries.values()} & {
                query.text for query in eval_queries.values()
            }:
                raise ValueError("Training and evaluation queries must not share IDs or exact text.")
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))

    top_n = min(args.candidates, len(corpus))
    if not args.mine_only and (not 1 <= args.k <= top_n or not 1 <= args.uniform <= len(corpus)):
        parser.error("Require 1 <= k <= candidate count and 1 <= uniform <= corpus size.")

    set_seed(args.seed)
    model = SentenceTransformer(args.model, device="cpu")
    query_prompt = model.prompts.get("query", "") if args.query_prompt is None else args.query_prompt
    document_prompt = model.prompts.get("document", "") if args.document_prompt is None else args.document_prompt

    def encode_queries(queries: dict[str, Query]) -> torch.Tensor:
        return encode(
            model,
            [query.text for query in queries.values()],
            is_query=True,
            prompt=query_prompt,
            batch_size=args.batch_size,
        )

    def encode_corpus() -> torch.Tensor:
        return encode(
            model,
            list(corpus.values()),
            is_query=False,
            prompt=document_prompt,
            batch_size=args.batch_size,
        )

    query_vectors, document_vectors = encode_queries(train_queries), encode_corpus()
    try:
        rows = mine(train_queries, train_pairs, corpus, query_vectors, document_vectors)
        write_mined(args.output, rows)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    del query_vectors, document_vectors
    if args.mine_only:
        print(f"Saved {len(rows)} mined rows to {args.output / 'mined.csv'}.")
        return

    reranker = CrossEncoder(args.reranker, device="cpu")
    if reranker.num_labels != 1:
        parser.error("The reranker must have one score per pair.")
    dataset = Dataset.from_list(rows)
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=DocumentFrequency.from_dataset(dataset, document_id_column="document_id"),
        corpus=list(corpus.values()),
        corpus_ids=list(corpus),
        num_uniform=args.uniform,
        query_id_column="query_id",
        document_id_column="document_id",
        extra_document_id_columns={"negative": "negative_document_id"},
        prompts={"anchor": query_prompt, "positive": document_prompt, "negative": document_prompt},
        router_mapping={"anchor": "query", "positive": "document", "negative": "document"},
    )
    loss = LogQMultipleNegativesRankingLoss(
        model,
        variant="plugin",
        combine="separate",
        mask_same_document=True,
        mask_same_query=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            batch_sampler=BatchSamplers.BATCH_SAMPLER,
            use_cpu=True,
            seed=args.seed,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=dataset,
        loss=loss,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output / "model")

    metrics, rankings = evaluate(
        eval_queries,
        corpus,
        encode_queries(eval_queries),
        encode_corpus(),
        reranker,
        top_n=top_n,
        cutoff=args.k,
        batch_size=args.batch_size,
    )
    with (args.output / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for ranking in rankings:
            handle.write(json.dumps(ranking, ensure_ascii=False) + "\n")
    report = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
