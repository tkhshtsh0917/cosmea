# Cosmea — sampling-aware retrieval training

[日本語](README.ja.md)

Cosmea adds **logQ correction and Mixed Negative Sampling (MNS)** to Sentence Transformers. The standard random row sampler selects training examples; a collator adds uniform corpus negatives and sampling metadata; the loss corrects temperature-scaled logits.

The name *Cosmea* comes from cosmos flowers and their association with harmony.

Development version `0.1.0.dev0`. CPU mathematical and integration checks were run on 2026-11-17. Real-data effectiveness is unverified. Correcting negative sampling does not guarantee reduced popularity concentration or causal fairness.

## Installation

In your Python project, install Cosmea directly from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv add "cosmea[train] @ git+https://github.com/tkhshtsh0917/cosmea"
```

The `train` extra includes the dependencies needed by `SentenceTransformerTrainer`. If you only need the collator and loss functions, install `cosmea` without `[train]`. Python 3.12 is used in our checks; supported dependency ranges are listed in the [API guide](docs/api.md).

uv records the resolved Git commit in your project's `uv.lock`. To choose a specific revision, add `--rev` followed by a commit SHA to the command above.

For contributing, running the offline example, or building the package, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Use your own data

For a complete script that reads your training data, see the [CSV example](examples/README.md).

Prepare a model, a training dataset with ordered columns `anchor, positive, query_id, document_id`, and aligned corpus text/ID lists. The anchor contains the query text and the positive contains the matching document text.

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from cosmea import (
    DocumentFrequency,
    MixedNegativeDataCollator,
    LogQMultipleNegativesRankingLoss,
)

model = SentenceTransformer("path/to/local/model", device="cpu")
frequency = DocumentFrequency.from_dataset(train_dataset, document_id_column="document_id")

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,  # Must not exceed the distinct corpus size.
    query_id_column="query_id",
    document_id_column="document_id",
)

loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="yi",
    logq_strength=1.0,
    mask_same_document=True,
    mask_same_query=True,
)

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    per_device_train_batch_size=64,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    use_cpu=True,
    report_to="none",
    save_strategy="no",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    data_collator=collator,
)

trainer.train()
```

If the model needs prompts, Router tasks or length limits, pass `prompts`, `router_mapping` and `max_length` to the **collator**. TrainingArguments alone does not configure a custom collator. See the [preprocessing example](docs/api.md#set-prompts-router-tasks-and-length-limits).

For held-out evaluation, use a retrieval `evaluator` instead of reusing this collator with `eval_dataset`. Unknown positive IDs in `eval_dataset` cause an error; see the [evaluation guide](docs/api.md#evaluate-and-reproduce-a-run).

The standard workflow uses random row sampling with both known-positive masks enabled, as shown above. Build counts from the same training split. The masks exclude the anchor's positive and other positives attached to the same query in the batch, including matching IDs in uniform or mined candidates.

The example explicitly sets `mask_same_document=True`. The API default is `None`, which resolves to False for yi and True for plugin/improved; `mask_same_query` defaults to True. This masked yi objective differs from yi with its default settings. Masking can reduce the usable negative count; adjust `num_uniform` and evaluate retrieval quality. Alternative sampler configurations are described in the [API guide](docs/api.md#choose-a-batch-sampler).

Both loss classes can also be wrapped in [MatryoshkaLoss](docs/api.md#train-at-multiple-embedding-sizes) to train several embedding sizes together.

## Components and objectives

| API | Responsibility |
| --- | --- |
| `DocumentFrequency` | Static positive-document counts, with int64 IDs/counts |
| `MixedNegativeDataCollator` | K distinct corpus samples without replacement and a dict label of IDs/counts/sources |
| `LogQMultipleNegativesRankingLoss` | Ordinary forward/backward |
| `CachedLogQMultipleNegativesRankingLoss` | ST GradCache with query-row chunks |
| `mixed_negative_logq_loss` | Per-row loss from already scaled `(b,C)` scores |

| Variant | Behavior | Scope |
| --- | --- | --- |
| `yi` (default) | Correct positives too; auto averages separate source objectives | Empirical iid proposal approximation, not a finite-batch consistency guarantee for the actual collator |
| `plugin` | Positive weight one; correct negatives by expected counts or inclusion probabilities | An unbiased partition estimate does not imply unbiased finite-sample loss/gradients |
| `improved` (research) | Stop-gradient form with the same first gradient as plugin | Its non-monotonic scalar loss is unsuitable for early stopping or best-checkpoint selection |

Use `num_uniform=0` for logQ alone, `logq_strength=0` for no correction and `combine="pooled"` for one joint candidate softmax. A plain-MNRL comparison needs **both `logq_strength=0, combine="pooled"` and matching masks**; yi's auto/separate mode is a different objective.

For mined negatives, separate mode gives them an independent uncorrected objective; pooled mode can reduce their relative influence. Start with yi/separate or plugin/pooled and expected counts.

See the [API contract](docs/api.md), [design and compatibility](docs/design.md), and [development checks](CONTRIBUTING.md).

## Validation and limits

Automated checks cover the sampling calculations, loss gradients and Sentence Transformers integration. Both ordinary and cached CPU Trainer integration tests run in [CI](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml). These checks use synthetic data; real-data improvements need separate evaluation.

- Verified environment: macOS arm64, Python 3.12.13, ST 6.3.0, PyTorch 2.15.0, Transformers 5.23.0.
- CPU, single training process, finite map-style datasets. CUDA/MPS, DDP, cross-device gathering, streaming counts and multiple datasets are unsupported or unverified.
- `yi + inclusion` is rejected; use plugin or improved for inclusion weighting.
- `mask_same_query=True` is a batch-local known-positive heuristic. Unbiased-estimator checks use False.
- Mined negative columns are uncorrected heuristic terms, not samples from a known proposal.
- Exact inclusion calculations can be expensive for large batches; expected counts are the usual starting point.

## Research background

Cosmea builds on the following research. It adapts the sampling and loss calculations to the batches it creates; it does not claim to reproduce the papers’ experiments.

- [Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- [Yang et al., 2020 — Mixed Negative Sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- [Ma & Collins, 2018 — Conditional NCE](https://arxiv.org/abs/1809.01812)
- [Khrylchenko et al., 2025 — Correcting the LogQ Correction](https://arxiv.org/abs/2507.09331) / [author code at the inspected commit](https://github.com/NonameUntitled/logq/tree/f48221cad05612a2558966288e77020e61f3ff01)

## License

Cosmea is licensed under [Apache-2.0](LICENSE). Dependencies retain their own licenses; see [NOTICE](NOTICE.md) for third-party references.

No package release has been made. Hosted check results are available in [GitHub Actions](https://github.com/tkhshtsh0917/cosmea/actions/workflows/ci.yml).
