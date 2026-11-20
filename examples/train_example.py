"""End-to-end example: SentenceTransformerTrainer + MNS collator + logQ loss.

Runs fully offline on CPU with a tiny randomly initialised BERT, so it doubles as an
integration test. Replace `build_tiny_model` with `SentenceTransformer("your/model")` and the
synthetic data with your click log + corpus.

    python examples/train_example.py            # plain loss
    python examples/train_example.py --cached   # GradCache loss
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from cosmea import (
    CachedLogQMultipleNegativesRankingLoss,
    DocumentFrequency,
    LogQMultipleNegativesRankingLoss,
    MixedNegativeDataCollator,
)

TOPICS = ["nurse", "driver", "engineer", "sales", "chef", "teacher", "designer", "clerk"]


def build_tiny_model(workdir: Path) -> SentenceTransformer:
    """Write a tiny local tokenizer/model and return a CPU sentence encoder."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "job", "query", "in", "city"]
    vocab += TOPICS + [f"n{i}" for i in range(400)]
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(vocab)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab.index("[CLS]")), ("[SEP]", vocab.index("[SEP]"))],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model_dir = workdir / "tiny-bert"
    BertModel(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    transformer = Transformer(str(model_dir), max_seq_length=16)
    return SentenceTransformer(
        modules=[transformer, Pooling(transformer.get_embedding_dimension(), "mean")],
        device="cpu",
    )


def build_data(seed: int = 0):
    """A click log with Zipf-popular jobs, repeated queries, and jobs that were never clicked."""
    rng = random.Random(seed)
    jobs = [{"document_id": j, "text": f"job {TOPICS[j % len(TOPICS)]} n{j} in city n{j % 7}"} for j in range(300)]
    clicked_jobs = jobs[:120]  # the other 180 jobs exist in the corpus but have no feedback
    weights = [1.0 / (rank + 1) for rank in range(len(clicked_jobs))]

    rows = []
    for _ in range(1200):
        job = rng.choices(clicked_jobs, weights=weights)[0]
        user = rng.randrange(60)
        topic = TOPICS[job["document_id"] % len(TOPICS)]
        rows.append(
            {
                "anchor": f"query {topic} n{user}",
                "positive": job["text"],
                "query_id": user * 100 + job["document_id"] % len(TOPICS),
                "document_id": job["document_id"],
            }
        )

    split = int(len(rows) * 0.9)
    return Dataset.from_list(rows[:split]), Dataset.from_list(rows[split:]), jobs


def recall_at_k(model: SentenceTransformer, eval_dataset: Dataset, jobs: list[dict], k: int = 10) -> float:
    """Recall@k of the clicked job, retrieving from the whole corpus (no sampled negatives)."""
    corpus_embeddings = model.encode([job["text"] for job in jobs], convert_to_tensor=True, normalize_embeddings=True)
    query_embeddings = model.encode(list(eval_dataset["anchor"]), convert_to_tensor=True, normalize_embeddings=True)

    top = (query_embeddings @ corpus_embeddings.T).topk(k, dim=1).indices.tolist()
    hits = sum(document_id in row for document_id, row in zip(eval_dataset["document_id"], top))
    return hits / len(eval_dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--variant", default="plugin")
    parser.add_argument("--combine", default="auto")
    parser.add_argument("--q-measure", default="expected_count")
    parser.add_argument("--logq-strength", type=float, default=1.0)
    parser.add_argument("--num-uniform", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=-1)
    args = parser.parse_args()

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        model = build_tiny_model(workdir)
        train_dataset, eval_dataset, jobs = build_data()

        # 1. Document frequencies of the *training* positives (exact counts).
        frequency = DocumentFrequency.from_dataset(train_dataset, document_id_column="document_id")

        # 2. Collator: adds uniformly sampled corpus jobs to every batch + the logQ metadata.
        collator = MixedNegativeDataCollator(
            preprocess_fn=model.preprocess,
            document_frequency=frequency,
            corpus=[job["text"] for job in jobs],
            corpus_ids=[job["document_id"] for job in jobs],
            num_uniform=args.num_uniform,
            query_id_column="query_id",
            document_id_column="document_id",
        )

        # 3. Loss: temperature-scaled cosine, then the sampling-bias correction.
        loss_kwargs = dict(
            variant=args.variant,
            combine=args.combine,
            q_measure=args.q_measure,
            logq_strength=args.logq_strength,
        )
        if args.cached:
            loss = CachedLogQMultipleNegativesRankingLoss(model, mini_batch_size=8, **loss_kwargs)
        else:
            loss = LogQMultipleNegativesRankingLoss(model, **loss_kwargs)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(workdir / "out"),
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=5e-3,
            logging_steps=1,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=args.workers,
            # Keep the default random batch sampler: the document counts assume it.
            use_cpu=True,
            seed=0,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            data_collator=collator,
        )

        before = recall_at_k(model, eval_dataset, jobs)
        trainer.train()
        after = recall_at_k(model, eval_dataset, jobs)
        print(f"Recall@10 over the full corpus: {before:.3f} -> {after:.3f}")

        history = [log for log in trainer.state.log_history if "loss" in log or "eval_loss" in log]
        for log in history:
            print({k: v for k, v in log.items() if k in ("step", "loss", "eval_loss")})
        train_losses = [log["loss"] for log in history if "loss" in log]
        assert train_losses and all(torch.isfinite(torch.tensor(train_losses))), "loss must be finite"
        print("OK")


if __name__ == "__main__":
    main()
