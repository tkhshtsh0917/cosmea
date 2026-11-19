"""Train a retrieval model from your CSV files; see the examples README for formats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.base.sampler import BatchSamplers

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator


def read_rows(path: Path, columns: list[str], aliases: dict[str, str]) -> list[dict[str, str]]:
    """Read required columns in model-input order, preserving all IDs as strings."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path.name} has duplicate column names.")
        source_columns = {column: column if column in headers else aliases.get(column, column) for column in columns}
        if not set(source_columns.values()).issubset(headers):
            raise ValueError(f"{path.name} must contain columns {columns}")
        rows = [{column: row[source] for column, source in source_columns.items()} for row in reader]

    if not rows or any(not value for row in rows for value in row.values()):
        raise ValueError(f"{path.name} must contain nonempty rows and required values")
    return rows


def read_training_rows(path: Path) -> list[dict[str, str]]:
    """Load training pairs and normalize the text and ID headers."""
    return read_rows(
        path,
        ["anchor", "positive", "query_id", "document_id"],
        aliases={"anchor": "query", "positive": "document", "document_id": "item_id"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Training device (default: cpu)",
    )
    parser.add_argument("--uniform", type=int, default=16, help="Distinct corpus negatives per batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--query-prompt", help="Prefix required by the chosen model for queries")
    parser.add_argument("--document-prompt", help="Prefix required by the chosen model for documents")
    parser.add_argument("--max-length", type=int, help="Token limit passed to the collator")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if args.max_length is not None and args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires an available PyTorch MPS device")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requires an available CUDA GPU")
        if torch.cuda.device_count() != 1:
            parser.error("--device cuda requires one visible GPU; select a GPU with CUDA_VISIBLE_DEVICES")

    # 1. Load matching query and document pairs and the collection we want to search.
    try:
        rows = read_training_rows(args.train)
        corpus = read_rows(args.corpus, ["document_id", "text"], aliases={"document_id": "item_id"})
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
    dataset = Dataset.from_list(rows)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output / "training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        use_cpu=args.device == "cpu",
        dataloader_pin_memory=args.device != "mps",
        save_strategy="no",
        report_to="none",
    )
    if training_args.device.type != args.device:
        parser.error(f"--device {args.device} was requested, but Trainer selected {training_args.device}")

    model = SentenceTransformer(args.model, device=str(training_args.device))

    # 2. Count positives and add uniform negatives to each batch.
    frequency = DocumentFrequency.from_dataset(dataset, document_id_column="document_id")
    prompts = {
        column: prompt
        for column, prompt in (("anchor", args.query_prompt), ("positive", args.document_prompt))
        if prompt is not None
    }
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=frequency,
        corpus=[row["text"] for row in corpus],
        corpus_ids=[row["document_id"] for row in corpus],
        num_uniform=args.uniform,
        query_id_column="query_id",
        document_id_column="document_id",
        prompts=prompts,
        max_length=args.max_length,
    )

    # 3. Train with logQ correction, then save the model for retrieval.
    loss = LogQMultipleNegativesRankingLoss(
        model,
        variant="plugin",
        combine="pooled",
        mask_same_document=True,
        mask_same_query=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
        data_collator=collator,
    )

    device = trainer.accelerator.device
    if device.type != args.device:
        parser.error(f"--device {args.device} was requested, but Accelerate selected {device}")

    print(f"Training device: {device}", flush=True)
    trainer.train()

    model_dir = args.output / "model"
    model.save_pretrained(model_dir)
    print(f"Saved model: {model_dir}")


if __name__ == "__main__":
    main()
