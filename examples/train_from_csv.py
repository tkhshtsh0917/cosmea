"""Train a retrieval model from your CSV files; see the examples README for formats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from cosmea import ItemFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator


def read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """Read required columns in model-input order, preserving all IDs as strings."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} must contain columns {columns}")
        rows = [{column: row[column] for column in columns} for row in reader]

    if not rows or any(not value for row in rows for value in row.values()):
        raise ValueError(f"{path.name} must contain nonempty rows and required values")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform", type=int, default=16, help="Distinct corpus negatives per batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")

    # 1. Load matching query/item pairs and the collection we want to search.
    dataset = Dataset.from_list(read_rows(args.train, ["query", "positive", "query_id", "item_id"]))
    corpus = read_rows(args.corpus, ["item_id", "text"])
    model = SentenceTransformer(args.model, device="cpu")

    # 2. Count positives and add uniform negatives to each batch.
    frequency = ItemFrequency.from_dataset(dataset, item_id_column="item_id")
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        item_frequency=frequency,
        corpus=[row["text"] for row in corpus],
        corpus_ids=[row["item_id"] for row in corpus],
        num_uniform=args.uniform,
        query_id_column="query_id",
        item_id_column="item_id",
    )

    # 3. Train with logQ correction, then save the model for retrieval.
    loss = LogQMultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output / "training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        use_cpu=True,
        save_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output / "model")


if __name__ == "__main__":
    main()
