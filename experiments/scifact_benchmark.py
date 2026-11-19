"""Compare base retrieval, Sentence Transformers MNRL and Cosmea on BEIR SciFact."""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median, stdev
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import torch
import cosmea
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers, NoDuplicatesBatchSampler
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim
from transformers import set_seed

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
BATCH_SIZE = 16
UNIFORM_COUNT = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
METRICS = ("ndcg@10", "mrr@10", "recall@10", "recall@100")
METHODS = ("mnrl", "in_batch", "uniform", "cosmea")
CONTRASTS = (("in_batch", "mnrl"), ("uniform", "in_batch"), ("cosmea", "uniform"), ("cosmea", "mnrl"))


def read_texts(archive: ZipFile, member: str) -> dict[str, str]:
    """Read BEIR JSONL and join each document's title and body."""
    result = {}
    with archive.open(member) as stream:
        for line in stream:
            row = json.loads(line)
            record_id = row["_id"]
            text = " ".join(part for part in (row.get("title", "").strip(), row["text"].strip()) if part)
            if record_id in result or not text:
                raise ValueError(f"{member} contains a repeated ID or empty text at {record_id!r}.")
            result[record_id] = text
    return result


def read_qrels(archive: ZipFile, split: str) -> dict[str, set[str]]:
    """Read the positive document IDs for each judged query."""
    result: dict[str, set[str]] = {}
    with io.TextIOWrapper(archive.open(f"scifact/qrels/{split}.tsv"), encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["score"]) > 0:
                result.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return result


def normalized_query(text: str) -> str:
    """Match copied queries after case folding and whitespace normalization."""
    return " ".join(text.casefold().split())


def validate_judgments(
    queries: dict[str, str], corpus: dict[str, str], judgments: dict[str, set[str]], split: str
) -> None:
    """Fail before training when a judged query or relevant document cannot be looked up."""
    if not judgments or any(not relevant for relevant in judgments.values()):
        raise ValueError(f"{split} has no positive judgments for at least one required query.")
    unknown_queries = judgments.keys() - queries.keys()
    unknown_documents = set().union(*judgments.values()) - corpus.keys()
    if unknown_queries or unknown_documents:
        raise ValueError(
            f"{split} judgments reference unknown IDs "
            f"(queries: {sorted(unknown_queries)[:3]}, documents: {sorted(unknown_documents)[:3]})."
        )


def select_train_pairs(
    queries: dict[str, str], train_qrels: dict[str, set[str]], test_qrels: dict[str, set[str]], count: int, seed: int
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Remove matching query texts and test-positive documents from supervised train pairs."""
    held_out = {normalized_query(queries[query_id]) for query_id in test_qrels}
    test_documents = set().union(*test_qrels.values())
    distinct_text = [
        (query_id, document_id)
        for query_id, document_ids in train_qrels.items()
        if normalized_query(queries[query_id]) not in held_out
        for document_id in document_ids
    ]
    eligible = sorted(pair for pair in distinct_text if pair[1] not in test_documents)
    if not 1 <= count <= len(eligible):
        raise ValueError(f"Train pair count must be between 1 and {len(eligible)} after test overlap filtering.")
    return random.Random(seed).sample(eligible, count), {
        "excluded_matching_query_text": sum(map(len, train_qrels.values())) - len(distinct_text),
        "excluded_test_positive_document": len(distinct_text) - len(eligible),
        "eligible_pairs": len(eligible),
    }


class MeasuredTrainer(SentenceTransformerTrainer):
    """Count optimizer inputs and retain sampled IDs for comparison after training."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rows_used = 0
        self.padded_tokens = 0
        self.sampled_ids = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.rows_used += inputs["anchor_input_ids"].size(0)
        self.padded_tokens += sum(value.numel() for name, value in inputs.items() if name.endswith("_input_ids"))
        if "label" in inputs and isinstance(inputs["label"], dict):
            label = inputs["label"]
            self.sampled_ids.append((label["query_id"], label["cand_id"]))
        return super().training_step(model, inputs, num_items_in_batch)


def ids_digest(batch_ids: list) -> str:
    """Hash in-order query and candidate IDs, preserving batch boundaries."""
    batches = [(queries.tolist(), documents.tolist()) for queries, documents in batch_ids]
    return hashlib.sha256(json.dumps(batches).encode()).hexdigest()


def validate_mnrl_schedule(dataset: Dataset, data_seed: int, seed: int) -> None:
    """Reject incompatible seed combinations before timing or evaluating any models."""
    sampler = NoDuplicatesBatchSampler(
        dataset.select_columns(["anchor", "positive"]),
        batch_size=BATCH_SIZE,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        seed=seed,
    )
    batches = list(sampler)
    if len(batches) != math.ceil(len(dataset) / BATCH_SIZE) or sorted(
        row for batch in batches for row in batch
    ) != list(range(len(dataset))):
        raise ValueError(
            f"Data seed {data_seed}, training seed {seed}: NO_DUPLICATES cannot consume "
            "all rows in the same number of steps as random row batches."
        )


def train(
    model: SentenceTransformer, dataset: Dataset, corpus: dict[str, str], method: str, seed: int, workdir: Path
) -> dict:
    """Time one epoch with the selected sampler and negative sources."""
    if method == "mnrl":
        sampler = BatchSamplers.NO_DUPLICATES
        loss = MultipleNegativesRankingLoss(model)
        collator = None
        dataset = dataset.select_columns(["anchor", "positive"])
    else:
        sampler = BatchSamplers.BATCH_SAMPLER
        loss = LogQMultipleNegativesRankingLoss(
            model,
            variant="plugin",
            combine="pooled",
            logq_strength=1.0 if method == "cosmea" else 0.0,
            mask_same_document=True,
            mask_same_query=True,
        )
        collator = MixedNegativeDataCollator(
            preprocess_fn=model.preprocess,
            document_frequency=DocumentFrequency.from_dataset(dataset, document_id_column="document_id"),
            corpus=list(corpus.values()),
            corpus_ids=list(corpus),
            num_uniform=0 if method == "in_batch" else UNIFORM_COUNT,
            query_id_column="query_id",
            document_id_column="document_id",
        )

    trainer = MeasuredTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(workdir / method),
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            batch_sampler=sampler,
            use_cpu=True,
            seed=seed,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        ),
        train_dataset=dataset,
        data_collator=collator,
        loss=loss,
    )
    start = perf_counter()
    trainer.train()
    elapsed = perf_counter() - start
    if trainer.rows_used != len(dataset):
        raise RuntimeError(f"{method} used {trainer.rows_used} of {len(dataset)} training rows.")
    positive_batches = [(queries, documents[: len(queries)]) for queries, documents in trainer.sampled_ids]
    return {
        "train_seconds": elapsed,
        "train_steps": trainer.state.global_step,
        "train_rows_used": trainer.rows_used,
        "padded_tokens": trainer.padded_tokens,
        "sampler": sampler.name,
        "positive_batches_sha256": ids_digest(positive_batches) if positive_batches else None,
        "candidate_batches_sha256": ids_digest(trainer.sampled_ids) if trainer.sampled_ids else None,
        "repeated_positive_rows": sum(
            len(documents) - len(set(documents.tolist())) for _, documents in positive_batches
        ),
        "repeated_query_rows": sum(len(queries) - len(set(queries.tolist())) for queries, _ in positive_batches),
    }


def query_metrics(hits: list, relevant: set[str]) -> dict[str, float]:
    """Compute binary metrics for one query, with corpus ID as the score tie-breaker."""
    ranked = sorted(hits, key=lambda hit: (-hit["score"], hit["corpus_id"]))[:100]
    ranks = [rank for rank, hit in enumerate(ranked, start=1) if hit["corpus_id"] in relevant]
    top10 = [rank for rank in ranks if rank <= 10]
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    return {
        "ndcg@10": sum(1 / math.log2(rank + 1) for rank in top10) / ideal,
        "mrr@10": 1 / top10[0] if top10 else 0.0,
        "recall@10": len(top10) / len(relevant),
        "recall@100": len(ranks) / len(relevant),
    }


class QueryMetricsEvaluator(InformationRetrievalEvaluator):
    """Reuse Sentence Transformers search and retain the scores for each judged query."""

    def compute_metrics(self, queries_result_list):
        self.per_query = {
            query_id: query_metrics(hits, self.relevant_docs[query_id])
            for query_id, hits in zip(self.queries_ids, queries_result_list, strict=True)
        }
        return super().compute_metrics(queries_result_list)


def evaluate(model: SentenceTransformer, evaluator: QueryMetricsEvaluator) -> dict:
    """Return macro averages and per-query results, checking that the two agree."""
    scores = evaluator(model)
    metrics = {metric: float(scores[f"cosine_{metric}"]) for metric in METRICS}
    for metric, value in metrics.items():
        if not math.isclose(value, mean(row[metric] for row in evaluator.per_query.values()), abs_tol=1e-12):
            raise RuntimeError(f"Per-query {metric} does not match the evaluator's aggregate.")
    return {"metrics": metrics, "per_query": evaluator.per_query}


def load_model(seed: int) -> SentenceTransformer:
    """Start from a fresh copy of the same checkpoint and RNG state."""
    set_seed(seed)
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    model.max_seq_length = MAX_LENGTH
    return model


def run_case(
    dataset: Dataset,
    corpus: dict[str, str],
    evaluator: QueryMetricsEvaluator,
    seed: int,
    position: int,
    repeats: int,
    workdir: Path,
) -> dict:
    """Train all methods before evaluating, reversing training order between trials."""
    timing = []
    models = {}
    for repeat in range(repeats):
        order = METHODS if (position + repeat) % 2 == 0 else METHODS[::-1]
        profiles = {}
        for method in order:
            print(f"Training {method} (repeat {repeat + 1}/{repeats})...", flush=True)
            model = load_model(seed)
            profiles[method] = train(model, dataset, corpus, method, seed, workdir)
            if repeat == 0:
                models[method] = model

        if len({profile["train_steps"] for profile in profiles.values()}) != 1:
            raise RuntimeError("Conditions consumed different numbers of optimizer steps.")
        random_batches = {profiles[method]["positive_batches_sha256"] for method in METHODS[1:]}
        if len(random_batches) != 1 or None in random_batches:
            raise RuntimeError("Conditions with random sampling used different positive batches.")
        if profiles["uniform"]["candidate_batches_sha256"] != profiles["cosmea"]["candidate_batches_sha256"]:
            raise RuntimeError("The logQ comparison used different sampled candidates.")
        timing.append({"order": order, "profiles": profiles})

    results = {}
    for method in METHODS:
        print(f"Evaluating {method}...", flush=True)
        results[method] = evaluate(models[method], evaluator)
        print(json.dumps(results[method]["metrics"]), flush=True)
    return {"timing": timing, "results": results}


def summarize(cases: list[dict]) -> dict:
    """Summarize methods and paired differences over one axis of the seed experiment."""

    def values(method: str, metric: str) -> list[float]:
        return [case["results"][method]["metrics"][metric] for case in cases]

    def spread(samples: list[float]) -> dict:
        return {
            "mean": mean(samples),
            "sd": stdev(samples) if len(samples) > 1 else None,
            "min": min(samples),
            "max": max(samples),
        }

    return {
        "runs": len(cases),
        "methods": {method: {metric: spread(values(method, metric)) for metric in METRICS} for method in METHODS},
        "paired_differences": {
            f"{left}-minus-{right}": {
                metric: spread([a - b for a, b in zip(values(left, metric), values(right, metric), strict=True)])
                for metric in METRICS
            }
            for left, right in CONTRASTS
        },
    }


def library_source() -> dict:
    """Identify the imported Cosmea source even when package versions are unchanged."""
    package = Path(cosmea.__file__).resolve().parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package.glob("*.py"))}
    try:
        tracked = subprocess.run(
            ["git", "-C", str(package), "ls-files", "--error-unmatch", "__init__.py"],
            capture_output=True,
            check=False,
        )
        commit = (
            subprocess.check_output(["git", "-C", str(package), "rev-parse", "HEAD"], text=True).strip()
            if tracked.returncode == 0
            else None
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "file_sha256": files}


def run(args: argparse.Namespace, workdir: Path) -> dict:
    """Sweep training and data seeds separately, sharing their first configuration."""
    with ZipFile(args.data) as archive:
        corpus = read_texts(archive, "scifact/corpus.jsonl")
        queries = read_texts(archive, "scifact/queries.jsonl")
        train_qrels = read_qrels(archive, "train")
        test_qrels = read_qrels(archive, "test")

    validate_judgments(queries, corpus, train_qrels, "Train")
    validate_judgments(queries, corpus, test_qrels, "Test")
    samples = {}
    datasets = {}
    for data_seed in args.data_seed:
        pairs, exclusions = select_train_pairs(queries, train_qrels, test_qrels, args.train_pairs, data_seed)
        samples[str(data_seed)] = pairs
        datasets[data_seed] = Dataset.from_list(
            [
                {
                    "anchor": queries[query_id],
                    "positive": corpus[document_id],
                    "query_id": query_id,
                    "document_id": document_id,
                }
                for query_id, document_id in pairs
            ]
        )

    first_data_seed = args.data_seed[0]
    first_seed = args.seed[0]
    settings = [(first_data_seed, seed) for seed in args.seed]
    settings.extend((data_seed, first_seed) for data_seed in args.data_seed[1:])
    for data_seed, seed in settings:
        validate_mnrl_schedule(datasets[data_seed], data_seed, seed)

    evaluator = QueryMetricsEvaluator(
        queries={query_id: queries[query_id] for query_id in test_qrels},
        corpus=corpus,
        relevant_docs=test_qrels,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        precision_recall_at_k=[10, 100],
        score_functions={"cosine": cos_sim},
        batch_size=64,
        show_progress_bar=False,
        write_csv=False,
    )
    torch.set_num_threads(args.threads)
    print("Evaluating base...", flush=True)
    baseline = evaluate(load_model(first_seed), evaluator)
    print(json.dumps(baseline["metrics"]), flush=True)

    cases = []
    for index, (data_seed, seed) in enumerate(settings):
        print(f"Case {index + 1}/{len(settings)}: data seed {data_seed}, training seed {seed}", flush=True)
        case = run_case(
            datasets[data_seed],
            corpus,
            evaluator,
            seed,
            position=index,
            repeats=args.timing_repeats if index == 0 else 1,
            workdir=workdir,
        )
        cases.append({"data_seed": data_seed, "training_seed": seed, **case})

    timing = cases[0]["timing"]
    return {
        "dataset": {
            "source": DATA_URL,
            "archive_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "corpus_documents": len(corpus),
            "test_queries": len(test_qrels),
            "test_positive_judgments": sum(map(len, test_qrels.values())),
            **exclusions,
            "selected_train_pairs": samples,
        },
        "model": {"name": MODEL, "revision": MODEL_REVISION},
        "settings": {
            "data_seeds": args.data_seed,
            "training_seeds": args.seed,
            "batch_size": BATCH_SIZE,
            "uniform_count": UNIFORM_COUNT,
            "cosmea_variant": "plugin",
            "cosmea_combine": "pooled",
            "cosmea_mask_same_query": True,
            "cosmea_mask_same_document": True,
            "learning_rate": LEARNING_RATE,
            "epochs": 1,
            "max_sequence_length": MAX_LENGTH,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("cosmea", "sentence-transformers", "torch", "transformers")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cosmea_source": library_source(),
        },
        "base": baseline,
        "cases": cases,
        "summary": {
            "vary_training_seed": summarize([case for case in cases if case["data_seed"] == first_data_seed]),
            "vary_data_seed": summarize([case for case in cases if case["training_seed"] == first_seed]),
            "reference_timing_medians": {
                method: median(trial["profiles"][method]["train_seconds"] for trial in timing) for method in METHODS
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the official BEIR scifact.zip")
    parser.add_argument("--output", type=Path, default=Path("outputs/scifact/results.json"))
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument(
        "--seed", nargs="+", type=int, default=[42], help="Training seeds; the first anchors the data sweep"
    )
    parser.add_argument(
        "--data-seed",
        nargs="+",
        type=int,
        default=[42],
        help="Pair sampling seeds; the first anchors the training sweep",
    )
    parser.add_argument("--timing-repeats", type=int, default=1, help="Train repeats for the first combination only")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    args = parser.parse_args()
    if args.threads < 1 or args.timing_repeats < 1:
        parser.error("Require positive --threads and --timing-repeats.")
    for seeds in (args.seed, args.data_seed):
        if len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
            parser.error("Seeds must be distinct within each list and between 0 and 2**32 - 1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scifact-trainers-", dir=args.output.parent) as directory:
        result = run(args, Path(directory))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
