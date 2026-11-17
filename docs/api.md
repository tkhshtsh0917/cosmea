# API guide

[日本語](ja/api.md) · [README](../README.md)

Start with the [README example](../README.md#use-your-own-data). This guide explains how to prepare your data and what each option changes.

## Prepare pairs and a searchable collection

A **query** is the search input and a **document** is a searchable entry. In a training row, the query is the **anchor**. A **positive** is an document known to match a query, such as a document that was clicked. A **negative** is a comparison document used during training; it is not necessarily irrelevant. The **corpus** is the complete collection of documents you want to search.

Prepare the following inputs:

| Input | What to put in it |
| --- | --- |
| Training dataset | Columns in the order `anchor, positive, query_id, document_id` |
| `corpus` | A list of document texts, including documents without clicks |
| `corpus_ids` | IDs in the same order as `corpus` |
| `DocumentFrequency` | Counts built from the training dataset that the DataLoader actually samples |

The collator prepares a group of examples for the model. After removing ID columns, it reads the first two text columns as anchor and positive. Any additional text columns supply negatives. The column names are configurable through the dataset schema; their order determines their training role. Remove unrelated metadata columns. Dataset columns named `label`, `labels`, `score` or `scores` are reserved and cause an error.

Use the same ID for the same document everywhere. IDs may be strings, bytes, or signed 64-bit integers. Text IDs are converted to stable 64-bit hashes; `"12"` and `12` are different IDs. Hash collisions are possible, so a consistent integer ID mapping is preferable for large collections. Without explicit IDs, documents are identified by exact text equality.

A positive ID missing from `DocumentFrequency` causes an error. Repeated corpus IDs are merged, keeping the first text; sampling is uniform over the remaining distinct IDs.

Existing code can still use `ItemFrequency` and the keywords `item_frequency`, `item_id_column`, `extra_item_id_columns` and `mask_same_item`. Use `DocumentFrequency`, `document_frequency`, `document_id_column`, `extra_document_id_columns` and `mask_same_document` in new code.

## Set prompts, Router tasks and length limits

Pass `prompts`, `router_mapping` and `max_length` directly to `MixedNegativeDataCollator`. Sentence Transformers does not copy these TrainingArguments settings into a custom collator. Setting them only on the Trainer can silently omit the expected preprocessing.

The following illustrates a Router model with query/document tasks. Mapping keys are dataset columns, and mapping values are Router task names. Use the prompts and task names your model requires; for an ordinary model omit `router_mapping` and use an integer `max_length`.

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    corpus=corpus_texts,
    corpus_ids=corpus_ids,
    num_uniform=64,
    query_id_column="query_id",
    document_id_column="document_id",
    prompts={"anchor": "query: ", "positive": "passage: "},
    router_mapping={"anchor": "query", "positive": "document"},
    max_length={"query": 64, "document": 256},
)
```

Uniform candidates always use the positive column's prompt, task and length limit. Configure extra mined columns explicitly when they need document preprocessing too. Keep evaluation/inference prompts consistent with training. Router metadata transfer is tested; full training of a Router model remains unverified.

## Add your own negative examples

Extra text columns, such as a hard negative found by another retriever, are also included as candidates. Their sampling probability is unknown, so Cosmea leaves these extra terms uncorrected.

If positives use explicit document IDs, supply an ID column for every extra negative column too:

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=frequency,
    document_id_column="document_id",
    extra_document_id_columns={"hard_negative": "hard_negative_id"},
)
```

This lets Cosmea recognize that two different texts refer to the same document. A missing mapping causes an error.

Use `combine="separate"` to give mined negatives their own uncorrected objective (the default for yi). In pooled mode their influence can shrink relative to corrected candidates as the training population grows; see [mined negatives and monitoring](design.md#mined-negatives-and-loss-monitoring).

## Choose the main options

Follow the README with random row sampling, `mask_same_document=True` and `mask_same_query=True`. It uses yi/separate with expected counts; plugin/pooled is another objective to compare under the same sampling and masks. `improved` and inclusion are advanced research controls; `improved` must not be selected by its scalar evaluation loss.

| Option | Meaning |
| --- | --- |
| `num_uniform` | Number of distinct corpus documents to add to each batch. Use 0 for in-batch negatives alone; do not exceed the number of distinct corpus IDs |
| `variant` | `yi` is the default. See [how the losses differ](design.md#three-loss-options) before switching to `plugin` or `improved` |
| `logq_strength` | Strength of the correction: 0 disables it, 1 applies it fully. Must be finite and nonnegative. Intermediate values need evaluation |
| `in_batch_logq_strength` / `uniform_logq_strength` | Overrides for the corresponding source with `combine="separate"`. `None` inherits `logq_strength`. Each value must be finite and nonnegative. A zero leaves that source's loss active and skips its correction. Pooled mode uses one shared strength |
| `scale` | Multiplies similarity before correction; default 20. Must be finite and positive. It is the inverse temperature |
| `combine` | `pooled` uses all candidate sources together. `separate` computes one loss per source and averages the active sources. `auto` chooses separate for yi and pooled otherwise |
| `q_measure` | `expected_count` keeps repeated documents and uses their expected number of appearances. `inclusion` keeps one eligible occurrence per document and uses the chance of appearing at least once. Inclusion is unavailable with yi |
| `mini_batch_size` | Positive integer chunk size for the Cached loss. Smaller chunks reduce score-matrix memory. Gradient accumulation alone does not increase the candidate pool |

In-batch, uniform and extra mined candidates are separate sources. A source with no usable negatives for a query is left out of the separate average. For example, a batch with one training row does not halve the uniform-source loss by averaging it with an empty in-batch loss.

## Avoid treating known matches as negatives

**Masking** means leaving a candidate out of the loss calculation.

- `mask_same_document=None` keeps accidental occurrences of the positive document for yi and excludes them for plugin/improved. Setting True or False overrides this choice and changes the objective.
- `mask_same_query=True` uses the positives attached to matching queries in the current batch. It cannot find all relevant documents in the corpus. This filtering is a practical heuristic; the strict unbiased-estimator result does not cover it.

With yi defaults (`mask_same_document=False`), query matching excludes matching positive columns only; copies in uniform/mined columns remain eligible. Set `mask_same_document=True` to propagate known document matches to all candidate columns, which changes the yi objective.

For yi with an active separate uniform correction, every positive ID must also occur in the corpus. Cosmea reports an error when this condition is not met.

## Choose a batch sampler

The standard Cosmea workflow uses `BatchSamplers.BATCH_SAMPLER`, `mask_same_document=True` and `mask_same_query=True`. This keeps the row draw aligned with the count-based proposal and removes known positives from negative candidates.

With the default random row sampler, `mask_same_document=True` masks candidate IDs that match the anchor's known positives. Combine it with `mask_same_query=True` to use the other positives attached to the same query in the batch. This also catches known positives in uniform or mined negatives, which are outside the sampler's check. Duplicates of other negative documents can still remain. Masked candidates are not replaced with other rows, so duplicate-heavy data can leave fewer usable negatives. Tune the uniform count using held-out retrieval metrics.

### Keep an existing NO_DUPLICATES setup

Sentence Transformers recommends [`NO_DUPLICATES` for in-batch negative losses](https://sbert.net/docs/package_reference/base/sampler.html). It avoids repeated input values across rows even when the values occur in different columns. In Sentence Transformers 6.3 this comparison also includes explicit ID columns, so use distinct namespaces for query and document IDs to avoid accidental matches between the two.

`NO_DUPLICATES` changes which rows may join a batch. With a row already selected, the chance of selecting another document depends on the values in the rows selected so far. The current count-based in-batch Q assumes uniform row sampling, including with `q_measure="inclusion"`. Weighted and grouped samplers also require their own proposal. Uniform corpus sampling is independent of the row sampler, so its correction only needs the corpus size and the number of draws.

For MNS with `NO_DUPLICATES` and uniform-only logQ correction, keep `num_uniform > 0` on the collator and use the following settings with the Trainer:

```python
from sentence_transformers.base.sampler import BatchSamplers

args = SentenceTransformerTrainingArguments(
    output_dir="outputs/run",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)
loss = LogQMultipleNegativesRankingLoss(
    model,
    variant="plugin",
    combine="separate",
    in_batch_logq_strength=0,
    uniform_logq_strength=1,
    mask_same_document=True,
)
```

The plugin loss leaves the positive logit unchanged, so the uniform correction changes the loss. In a separate yi objective, the same uniform logQ offset applies to the positive and all uniform negatives and cancels inside their softmax. A positive with an active yi uniform correction must still belong to the corpus. Both the ordinary and cached Cosmea losses accept the same strengths.

For plugin, the uniform term is $\log(1+(M/K)^{\lambda_u}S_u/e^{F_p})$, where $S_u$ sums the exponentiated eligible uniform scores, $F_p$ is the positive score and $\lambda_u$ is the uniform correction strength. At strength 1, this estimates a full-corpus negative sum. It changes the margin against the positive by $\log(M/K)$ and the resulting gradient. It is not a document-specific popularity correction. An unbiased sum does not imply unbiased loss or gradients, or improved retrieval quality. The separate average with an uncorrected in-batch loss remains a composite objective. Only sources with eligible negatives enter that average.

## Evaluate and reproduce a run

Build counts from the sampled training split. For held-out quality, search the full corpus and compute retrieval metrics. Reusing training counts for a different evaluation split generally gives the wrong sampling correction. The Trainer evaluation test checks that metadata reaches the loss; it is not a held-out quality result.

For held-out quality, pass a retrieval `evaluator` to the Trainer and omit `eval_dataset`. `eval_queries` maps query IDs to text; `eval_relevant_docs` maps those IDs to sets of relevant corpus IDs. IDs must match the corpus dictionary. Supply the model's evaluation prompts if needed.

```python
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=dict(zip(corpus_ids, corpus_texts)),
    relevant_docs=eval_relevant_docs,
    name="held_out",
)
```

Pass this as `evaluator=evaluator` alongside the training dataset and collator. An `eval_dataset` containing positive IDs absent from training counts raises `Positive ID absent from document_frequency`; even if all IDs exist, training counts generally give the wrong evaluation proposal. The collator is not a full-corpus evaluator. For early stopping or best-checkpoint selection, use a held-out retrieval metric, not the `improved` scalar loss.

Each DataLoader worker gets its own random stream derived from PyTorch's RNG. The same launch conditions are needed for reproducibility. Restoring the exact collator stream from a checkpoint, or keeping identical draws after changing the worker count, is not guaranteed.

The supported ranges are Sentence Transformers `>=6.3,<6.4` and PyTorch `>=2.6,<3`, with NumPy `>=2,<3`. CPU with one training process is verified. GPU, distributed training and streaming datasets need additional validation; see [design](design.md#resource-and-compatibility-limits).

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

CPU tests compare plain and cached wrappers with explicit losses at 32, 16 and 8 dimensions, including unequal weights. The wrapper comparison uses plugin/expected-count; score-level tests cover variant identities separately. Trainer tests also exercise a singleton final batch with mined negatives, using plain yi and cached yi wrapped in Matryoshka. This verifies integration and gradients, not retrieval quality after truncation. Random dimension subsets (`n_dims_per_step`), GPU and Matryoshka2dLoss have not been tested here.

## Reference: metadata passed to the loss

Most users can leave this section to the collator. It matters when calling the score-level functions directly.

Let $B$ be the number of training rows, $H$ the number of extra negative columns and $K$ the number of uniform samples. There are $C=B(1+H)+K$ candidate entries, ordered as positives, each extra column, then uniform samples. Shapes below describe tensor dimensions.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `query_id` | $(B,)$ / int64 | Query IDs |
| `cand_id` | $(C,)$ / int64 | Document IDs in candidate order |
| `cand_count` | $(C,)$ / int64 | Number of positive training rows for each document |
| `cand_source` | $(C,)$ / int64 | 0=in-batch, 1=uniform, 2=extra |
| `cand_in_corpus` | $(C,)$ / bool | Whether the document can be drawn from the corpus |
| `num_rows` / `corpus_size` | scalar / int64 | Total training rows $N$ / distinct corpus documents $M$ |

These tensors travel in `batch["label"]`; they are not model input features. Loss classes return a scalar mean. `mixed_negative_logq_loss` returns one value per query row, shape $(b,)$. For a chunk, `rows` identifies the original batch rows and their positive candidate columns.

`candidate_log_q(..., rows=...)` returns shape $(b,C)$ and accounts for the positive row already being selected. Without `rows`, it returns $(C,)$ and does not subtract that occurrence. `iid_proposal=True` uses the count/$N$ approximation for yi; it does not change how the collator samples.

Low-level helpers (`candidate_log_q`, `negative_log_partition`, `stable_id`) remain available for existing callers, but the classes and `mixed_negative_logq_loss` are the main usage API. The helper-only `iid_proposal=True, q_measure="inclusion"` computes binomial inclusion; it does not enable yi/inclusion losses.
