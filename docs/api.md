# API and data contract

[日本語](ja/api.md)

## Columns and identity

After removing ID columns, `MixedNegativeDataCollator` treats the first two columns as query and positive. Set column order explicitly and remove unrelated numeric metadata. `label`, `labels`, `score` and `scores` are reserved for sampling metadata and rejected as dataset columns.

IDs accept strings, bytes and signed int64 integers. Integers remain unchanged; text uses a stable 64-bit blake2b digest. String `"12"` and integer `12` differ. Hash collisions cannot be ruled out; use a consistent int64 mapping for large deployments. Without IDs, identity is exact text equality, so alternate renderings of an item are separate items.

Counts and corpus IDs must share one identity convention. Unknown positive IDs raise an error. Duplicate corpus IDs retain the first text and are sampled uniformly over distinct IDs; ensure that representative text is suitable.

When using explicit item IDs with extra negative columns, provide the mapping:

```python
collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    item_frequency=frequency,
    item_id_column="item_id",
    extra_item_id_columns={"hard_negative": "hard_negative_id"},
)
```

Omitting this mapping would mix text hashes with item IDs and miss accidental hits, so every extra column must be mapped when positive IDs are explicit.

## Metadata shapes

B is the training batch size, H the number of extra text columns, K uniform samples, C=B(1+H)+K. Candidate order is B positives, B entries per extra column, then K uniform entries.

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `query_id` | `(B,)` / int64 | Query identity |
| `cand_id` | `(C,)` / int64 | Item identity |
| `cand_count` | `(C,)` / int64 | Positive counts in the sampled dataset |
| `cand_source` | `(C,)` / int64 | 0=in-batch, 1=uniform, 2=other |
| `cand_in_corpus` | `(C,)` / bool | Membership in uniform proposal support |
| `num_rows` / `corpus_size` | scalar / int64 | N / distinct M |

The dict travels in `batch["label"]` to the loss, not into model inputs. Module losses return a mean scalar; the functional API returns `(b,)` values. For chunks, `rows` indexes the original B rows and also identifies each positive candidate column.

## Options

- `scale`: positive finite inverse temperature, default 20. Apply `scale * similarity - lambda * log q` in that order.
- `logq_strength`: finite nonnegative value, normally 1. Intermediate values require evaluation; this is not a monotonic exposure control.
- `combine="auto"`: separate for yi, pooled for plugin/improved.
- `separate`: average only sources with a valid negative for that row. An empty in-batch source at B=1 does not halve uniform loss.
- `q_measure="expected_count"`: retain duplicates. Plugin/improved use conditional expected counts in the N−1 remaining-row population.
- `q_measure="inclusion"`: deduplicate after row-specific masking; use exact hypergeometric inclusion probabilities. Rejected with yi.
- `mask_same_item=None`: keep accidental hits for yi and mask them for plugin/improved. Explicit overrides change the objective and theoretical assumptions.
- `mask_same_query=True`: mask other known positives for a query visible in this batch. This is not a complete relevance set. Query-dependent filtering changes the proposal and falls outside strict unbiasedness claims.
- `mini_batch_size`: positive integer GradCache chunk size. Gradient accumulation alone does not enlarge the candidate pool.

`candidate_log_q(..., rows=...)` returns `(b,C)` conditional on the row being present; without rows it returns `(C,)`. The latter does not subtract the positive's own occurrence. `iid_proposal=True` uses the yi approximation count/N; it does not change the collator into an iid sampler.

## Reproducibility and evaluation

Workers initialize separate streams from PyTorch's RNG. Reproducibility assumes the same launch conditions. Exact collator-RNG restoration from checkpoints and identical draws after changing worker counts are not guaranteed.

Held-out in-batch proposals generally differ from training frequencies. Use full-corpus retrieval for held-out quality. Trainer dict-label propagation is tested with same-dataset evaluation as an interface smoke check, not a generalization result. See [design](design.md) for estimator assumptions.
