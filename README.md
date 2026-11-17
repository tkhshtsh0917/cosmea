# Cosmea — sampling-aware retrieval training

[日本語](README.ja.md)

Cosmea adds **logQ correction and Mixed Negative Sampling (MNS)** to Sentence Transformers. The standard random row sampler selects training examples; a collator adds uniform corpus negatives and sampling metadata; the loss corrects temperature-scaled logits.

The name comes from *Cosmea*, a synonym of Cosmos, and the red flower's Japanese meaning “harmony”: combining negative distributions with matching corrections. [Kew nomenclature](https://powo.science.kew.org/taxon/8407-1) / [Chiba Prefecture flower symbolism](https://www.pref.chiba.lg.jp/ryuhan/pbmgm/zukan/kaki/cosmos.html).

Development version `0.1.0.dev0`. CPU mathematical and integration checks were run on 2026-11-17. Real-data effectiveness is unverified. Correcting negative sampling does not guarantee reduced popularity concentration or causal fairness.

## Setup and offline example

From this checkout, with [mise](https://mise.jdx.dev/getting-started.html) installed:

```bash
mise trust
mise install
mise run setup
mise run smoke
mise run check
```

mise manages development tools; uv manages dependencies and Python. See [contributing](CONTRIBUTING.md) for version ownership and individual checks. For library use, `pip install -e '.[train]'` is also supported. The tiny-BERT example generates its model and synthetic data locally; after dependency installation it needs no model downloads. This is a local checkout, not an instruction to install a published package named cosmea.

## Use your own data

Prepare a model, a training dataset with ordered columns `query, positive, query_id, item_id`, and aligned corpus text/ID lists:

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from cosmea import (
    ItemFrequency,
    MixedNegativeDataCollator,
    LogQMultipleNegativesRankingLoss,
)

model = SentenceTransformer("path/to/local/model", device="cpu")
frequency = ItemFrequency.from_dataset(train_dataset, item_id_column="item_id")

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,  # Must not exceed the distinct corpus size.
    query_id_column="query_id",
    item_id_column="item_id",
)

loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="yi",
    logq_strength=1.0,
)

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    per_device_train_batch_size=64,
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

Compute frequencies from the **same split that the DataLoader samples uniformly**. NO_DUPLICATES, weighted or grouped samplers invalidate this proposal model. Reusing training frequencies for held-out in-batch loss has the same problem; the example evaluates held-out queries by full-corpus Recall separately.

## Components and objectives

| API | Responsibility |
| --- | --- |
| `ItemFrequency` | Static positive-item counts, with int64 IDs/counts |
| `MixedNegativeDataCollator` | K distinct corpus samples without replacement and a dict label of IDs/counts/sources |
| `LogQMultipleNegativesRankingLoss` | Ordinary forward/backward |
| `CachedLogQMultipleNegativesRankingLoss` | ST GradCache with query-row chunks |
| `mixed_negative_logq_loss` | Per-row loss from already scaled `(b,C)` scores |

| Variant | Behavior | Scope |
| --- | --- | --- |
| `yi` (default) | Correct positives too; auto averages separate source objectives | Empirical iid proposal approximation, not a finite-batch consistency guarantee for the actual collator |
| `plugin` | Positive weight one; correct negatives by expected counts or inclusion probabilities | An unbiased partition estimate does not imply unbiased finite-sample loss/gradients |
| `improved` | Stop-gradient confidence-weighted objective | Sampling-design adaptation of the 2025 formulation; same first gradient as plugin under identical options, not literal reproduction of the paper or author code |

Use `num_uniform=0` for logQ alone, `logq_strength=0` for no correction and `combine="pooled"` for one joint candidate softmax. A plain-MNRL comparison needs **both `logq_strength=0, combine="pooled"` and matching masks**; yi's auto/separate mode is a different objective.

See the [API contract](docs/api.md), [design and compatibility](docs/design.md), and [development checks](CONTRIBUTING.md).

## Validation and limits

Validation covers 53 tests: 37 ordinary/integration cases, 7 statistical checks, one two-worker test, 5 public-document contract checks and 3 experiment-metric regressions. Ordinary and cached CPU examples completed four steps. These synthetic smoke runs are not performance comparisons.

- Verified environment: macOS arm64, Python 3.12.13, ST 6.3.0, PyTorch 2.15.0, Transformers 5.23.0.
- CPU, single training process, finite map-style datasets. CUDA/MPS, DDP, cross-device gathering, streaming counts and multiple datasets are unsupported or unverified.
- `yi + inclusion` is rejected; use plugin or improved for inclusion weighting.
- `mask_same_query=True` is a batch-local known-positive heuristic. Unbiased-estimator checks use False.
- Mined negative columns are uncorrected heuristic terms, not samples from a known proposal.
- Exact inclusion calculations can be expensive for large batches; expected counts are the usual starting point.
- Performance numbers reported for the original prototype are not results for this version.

## Origins and references

Reviewed and reorganized from a user-supplied st-logq prototype. This implementation adapts the following ideas to its own sampling design; it does not claim to reproduce their experiments.

- [Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- [Yang et al., 2020 — Mixed Negative Sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- [Ma & Collins, 2018 — Conditional NCE](https://arxiv.org/abs/1809.01812)
- [Khrylchenko et al., 2025 — Correcting the LogQ Correction](https://arxiv.org/abs/2507.09331) / [author code at the inspected commit](https://github.com/NonameUntitled/logq/tree/f48221cad05612a2558966288e77020e61f3ff01)

See [NOTICE](NOTICE.md) for licensing/provenance status. No public release has been made. Hosted CI is configured but has not yet been run.
