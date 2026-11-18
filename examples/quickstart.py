"""Fine-tune a small text retriever, reload it, and search a toy corpus on CPU."""

import argparse
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from transformers import set_seed

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator


def run(model_name: str, output_dir: Path) -> None:
    """Train on example pairs and evaluate both checkpoints on the same held-out queries."""
    corpus = {
        "lists": "In Python, append adds one element to the end of a list.",
        "joins": "A SQL LEFT JOIN keeps every row from the left table.",
        "plants": "Basil grows well in a sunny spot with regular watering.",
        "bike": "Use a floor pump to inflate a bicycle tire.",
        "rice": "Rinse rice before simmering it with water in a covered pan.",
        "weather": "A barometer measures atmospheric pressure.",
    }
    train = Dataset.from_dict(
        {
            "anchor": ["add an item to a list", "keep all rows from the first table", "how often to water basil"],
            "positive": [corpus["lists"], corpus["joins"], corpus["plants"]],
            "query_id": ["train-1", "train-2", "train-3"],
            "document_id": ["lists", "joins", "plants"],
        }
    )
    eval_queries = {"eval-1": "how do I put air in my bike tires?", "eval-2": "what does a barometer measure?"}
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus,
        relevant_docs={"eval-1": {"bike"}, "eval-2": {"weather"}},
        ndcg_at_k=[3],
        precision_recall_at_k=[3],
        score_functions={"cosine": cos_sim},
        show_progress_bar=False,
        write_csv=False,
    )

    set_seed(42)
    model = SentenceTransformer(model_name, device="cpu")
    before = evaluator(model)
    collator = MixedNegativeDataCollator(
        preprocess_fn=model.preprocess,
        document_frequency=DocumentFrequency.from_dataset(train, document_id_column="document_id"),
        corpus=list(corpus.values()),
        corpus_ids=list(corpus),
        num_uniform=2,
        query_id_column="query_id",
        document_id_column="document_id",
        prompts={"anchor": model.prompts.get("query", ""), "positive": model.prompts.get("document", "")},
    )
    loss = LogQMultipleNegativesRankingLoss(
        model,
        variant="plugin",
        combine="pooled",
        mask_same_document=True,
        mask_same_query=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=3,
            batch_sampler=BatchSamplers.BATCH_SAMPLER,
            use_cpu=True,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=train,
        data_collator=collator,
        loss=loss,
    )
    trainer.train()

    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    saved = SentenceTransformer(str(model_dir), device="cpu")
    after = evaluator(saved)
    print(f"Saved model: {model_dir}")
    print("Held-out queries (initial -> fine-tuned):")
    for metric in ("ndcg@3", "recall@3"):
        key = f"cosine_{metric}"
        print(f"  {metric}: {before[key]:.3f} -> {after[key]:.3f}")

    query = eval_queries["eval-1"]
    query_vector = saved.encode_query([query], show_progress_bar=False)
    document_vectors = saved.encode_document(list(corpus.values()), show_progress_bar=False)
    scores = cos_sim(query_vector, document_vectors)[0]
    best = int(scores.argmax())
    document_id, document = list(corpus.items())[best]
    print(f"Query: {query}")
    print(f"Top result: [{document_id}] {document}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
