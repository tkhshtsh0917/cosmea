# API guide

[日本語](ja/api.md) · [README](../README.md)

Start with the [README example](../README.md#use-your-own-data). This guide explains how to prepare your data and what each option changes.

## Prepare pairs and a searchable collection

A **positive** is an item known to match a query, such as a document that was clicked. A **negative** is a comparison item used during training; it is not necessarily irrelevant. The **corpus** is the complete collection of items you want to search.

Prepare the following inputs:

| Input | What to put in it |
| --- | --- |
| Training dataset | Columns in the order `query, positive, query_id, item_id` |
| `corpus` | A list of item texts, including items without clicks |
| `corpus_ids` | IDs in the same order as `corpus` |
| `ItemFrequency` | Counts built from the training dataset that the DataLoader actually samples |

The collator prepares a group of examples for the model. After removing ID columns, it reads the first two text columns as query and positive. Remove unrelated metadata columns. Dataset columns named `label`, `labels`, `score` or `scores` are reserved and cause an error.

Use the same ID for the same item everywhere. IDs may be strings, bytes, or signed 64-bit integers. Text IDs are converted to stable 64-bit hashes; `"12"` and `12` are different IDs. Hash collisions are possible, so a consistent integer ID mapping is preferable for large collections. Without explicit IDs, items are identified by exact text equality.

A positive ID missing from `ItemFrequency` causes an error. Repeated corpus IDs are merged, keeping the first text; sampling is uniform over the remaining distinct IDs.

## Add your own negative examples

Extra text columns, such as a hard negative found by another retriever, are also included as candidates. Their sampling probability is unknown, so Cosmea leaves these extra terms uncorrected.

If positives use explicit item IDs, supply an ID column for every extra negative column too:

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    item_id_column="item_id",
    extra_item_id_columns={"hard_negative": "hard_negative_id"},
)
```

This lets Cosmea recognize that two different texts refer to the same item. A missing mapping causes an error.

## Choose the main options

The defaults are a starting point for comparison, not a claim that one configuration always works best.

| Option | Meaning |
| --- | --- |
| `num_uniform` | Number of distinct corpus items to add to each batch. Use 0 for in-batch negatives alone; do not exceed the number of distinct corpus IDs |
| `variant` | `yi` is the default. See [how the losses differ](design.md#three-loss-options) before switching to `plugin` or `improved` |
| `logq_strength` | Strength of the correction: 0 disables it, 1 applies it fully. Must be finite and nonnegative. Intermediate values need evaluation |
| `scale` | Multiplies similarity before correction; default 20. Must be finite and positive. It is the inverse temperature |
| `combine` | `pooled` uses all candidate sources together. `separate` computes one loss per source and averages the active sources. `auto` chooses separate for yi and pooled otherwise |
| `q_measure` | `expected_count` keeps repeated items and uses their expected number of appearances. `inclusion` keeps one eligible occurrence per item and uses the chance of appearing at least once. Inclusion is unavailable with yi |
| `mini_batch_size` | Positive integer chunk size for the Cached loss. Smaller chunks reduce score-matrix memory. Gradient accumulation alone does not increase the candidate pool |

A source with no usable negatives for a query is left out of the separate average. For example, a batch with one training row does not halve the uniform-source loss by averaging it with an empty in-batch loss.

## Avoid treating known matches as negatives

**Masking** means leaving a candidate out of the loss calculation.

- `mask_same_item=None` keeps accidental occurrences of the positive item for yi and excludes them for plugin/improved. Setting True or False overrides this choice and changes the objective.
- `mask_same_query=True` uses the positives attached to matching queries in the current batch. It cannot find all relevant items in the corpus. This filtering is a practical heuristic; the strict unbiased-estimator result does not cover it.

For yi with a separate uniform loss, every positive ID must also occur in the corpus. Cosmea reports an error when this condition is not met.

## Evaluate and reproduce a run

Use the default uniform random row sampler. `NO_DUPLICATES`, weighted samplers and grouped samplers change which rows are drawn, so the same count-based correction no longer describes their sampling.

Build counts from the sampled training split. For held-out quality, search the full corpus and compute retrieval metrics. Reusing training counts for a different evaluation split generally gives the wrong sampling correction. The Trainer evaluation test checks that metadata reaches the loss; it is not a held-out quality result.

Each DataLoader worker gets its own random stream derived from PyTorch's RNG. The same launch conditions are needed for reproducibility. Restoring the exact collator stream from a checkpoint, or keeping identical draws after changing the worker count, is not guaranteed.

The supported ranges are Sentence Transformers `>=6.3,<6.4` and PyTorch `>=2.15,<2.16`. CPU with one training process is verified. GPU, distributed training and streaming datasets need additional validation; see [design](design.md#resource-and-compatibility-limits).

## Train at multiple embedding sizes

You can wrap either Cosmea loss in Sentence Transformers 6.3's `MatryoshkaLoss`. It trains the leading dimensions of an embedding at several sizes, so retrieval can later use a shorter embedding.

The following example assumes a 768-dimensional model. Choose positive dimensions no larger than your model's output, and include the full dimension if you want to use it too.

```python
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss
from cosmea import LogQMultipleNegativesRankingLoss

base_loss = LogQMultipleNegativesRankingLoss(model, variant="yi")
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 384, 192],
)
```

Pass `loss` to the Trainer and keep the same `MixedNegativeDataCollator`. To use gradient caching, create a `CachedLogQMultipleNegativesRankingLoss` as the base instead. Use a fresh base loss for each wrapper; the upstream cached wrapper decorates the base loss in place.

Cosmea's sampling metadata is independent of embedding size, so the same IDs and counts are used at each size. With weights $w_m$, the combined loss is $\sum_m w_m\ell_m$; the weights default to 1 and are not automatically averaged.

CPU tests compare plain and cached wrappers with explicit losses at 32, 16 and 8 dimensions, including unequal weights. They cover yi/expected-count, plugin and improved with expected-count or inclusion, plus Trainer training/evaluation for plugin. This verifies integration and gradients, not retrieval quality after truncation. Random dimension subsets (`n_dims_per_step`), GPU and Matryoshka2dLoss have not been tested here.

## Reference: metadata passed to the loss

Most users can leave this section to the collator. It matters when calling the score-level functions directly.

Let $B$ be the number of training rows, $H$ the number of extra negative columns and $K$ the number of uniform samples. There are $C=B(1+H)+K$ candidate entries, ordered as positives, each extra column, then uniform samples. Shapes below describe tensor dimensions.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `query_id` | $(B,)$ / int64 | Query IDs |
| `cand_id` | $(C,)$ / int64 | Candidate item IDs |
| `cand_count` | $(C,)$ / int64 | Number of positive training rows for each item |
| `cand_source` | $(C,)$ / int64 | 0=in-batch, 1=uniform, 2=extra |
| `cand_in_corpus` | $(C,)$ / bool | Whether the item can be drawn from the corpus |
| `num_rows` / `corpus_size` | scalar / int64 | Total training rows $N$ / distinct corpus items $M$ |

These tensors travel in `batch["label"]`; they are not model input features. Loss classes return a scalar mean. `mixed_negative_logq_loss` returns one value per query row, shape $(b,)$. For a chunk, `rows` identifies the original batch rows and their positive candidate columns.

`candidate_log_q(..., rows=...)` returns shape $(b,C)$ and accounts for the positive row already being selected. Without `rows`, it returns $(C,)$ and does not subtract that occurrence. `iid_proposal=True` uses the count/$N$ approximation for yi; it does not change how the collator samples.
