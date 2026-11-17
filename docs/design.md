# Design, objectives and compatibility

[日本語](ja/design.md)

## Responsibilities

`src/cosmea` contains runtime code, `examples` runnable usage, `tests` behavioral and mathematical verification, `experiments` optional comparisons, and `scripts` development tooling. Training integration belongs to the `train` dependency extra; tooling belongs to `dev`. Public documentation is sufficient to use, build and evaluate the package.

A collator sees the raw batch and adds K uniform corpus candidates plus metadata. The loss receives scaled scores and applies correction. Logit correction cannot be implemented by a sampler alone. The standard random row sampler remains responsible for selecting B training rows.

## Proposal measures

Let N be training rows, B batch rows, M distinct corpus items, K uniform items, p_i the row's positive, and c_d its dataset count. For a fixed row i, n=B−1 other rows are drawn from N−1, with c'_id=c_d−1[d=p_i]. Corpus draws are independent of row sampling.

Expected multiplicity and pooled inclusion probability are:

```
e_i(d) = (B-1) * c'_id/(N-1) + (K/M) * 1[d in corpus]
pi_in_i(d) = 1 - choose(N-1-c'_id, B-1)/choose(N-1, B-1)
pi_i(d) = 1 - (1-pi_in_i(d)) * (1-(K/M)*1[d in corpus])
```

Define the in-batch contribution as zero for B=N=1, and the uniform contribution as zero for K=0. Retain duplicate occurrences when dividing by expected multiplicity; keep one eligible occurrence per item when dividing by inclusion probability. For example, drawing two of `[a,b,b]` gives expected multiplicities 2/3 and 4/3 but inclusion probabilities 2/3 and 1. Deduplicating b and still dividing by 4/3 would undercount its contribution.

The pooled negative-partition estimator is unbiased only for its supported item set, with the specified sampling design and positive-item exclusion; query-dependent masks and mined negatives are outside this claim. Taking a logarithm or ratio generally reintroduces finite-sample bias.

## Objectives and defaults

Scores F have shape `(B,C)`, C=B(1+H)+K for H extra negative columns. Compute `F = scale * similarity` before subtracting `lambda * log q`. Correcting similarity before scaling changes the objective. Masks remove candidates from the normalization set.

The default yi + expected_count + separate uses the empirical iid proposal c_d/N. The actual collator samples without replacement, so iid ranking-NCE consistency results are not guarantees for this implementation. yi+inclusion is rejected. Separate uniform yi requires every positive ID in the corpus. Fixed-source mixtures, filtering and deduplication also change theoretical assumptions.

Plugin leaves the positive uncorrected. If z=log(Zhat_neg), its loss is `softplus(z-F_positive)`. Improved uses `sg(sigmoid(z-F_positive)) * (z-F_positive)`, where sg stops gradients through the weight. With the same z, both have identical first derivatives, but different values and higher-order derivatives. This is an adaptation of the cited confidence-weighted formulation, not a literal reproduction of the paper's Q-prime or the author's candidate construction.

Under ideal iid negative sampling and an expressive score model, subtracting logQ in the loss targets the observed conditional item distribution; uncorrected learning targets a probability ratio. That does not remove exposure bias already present in the positive log or guarantee lower popularity concentration. Defaults are starting points, not empirically established superiority.

## Bounds and numerical behavior

Require `0 <= K <= M`, `1 <= B <= N` and positive integer cached chunk size. Invalid options/IDs/counts raise errors. An unknown positive is not assigned an artificial count. An empty negative pool returns zero loss and gradient. Separate mode averages only active sources. Half/bfloat16 scores are promoted for loss arithmetic; mixed-precision model training remains unverified.

All corpus identity indexing is held in memory, O(M), in addition to text storage. Ordinary score memory is O(B*C); Cached mode uses b query rows at a time, giving O(b*C) score memory while candidate embeddings remain cached. Exact inclusion uses a product over B−1 draws for every queried candidate, O((B−1)*b*C) arithmetic per chunk. No automatic memory quota or corpus streaming is implemented. Use expected counts as the initial large-batch setting.

## Compatibility and evaluation

The package depends on ST 6.3 internal collator helpers and GradCache, so the supported ranges are ST `>=6.3,<6.4` and PyTorch `>=2.15,<2.16`; exact development versions are locked. ST v5 compatibility is not claimed. Updating these ranges requires Trainer metadata and cached-gradient integration checks.

NO_DUPLICATES, weighted/grouped samplers, DDP padding, cross-device gathering, streaming/time-window counts and multiple datasets require additional proposal definitions and verification. GPU execution is unverified. Use consistent ID spaces, including explicit IDs for mined negative columns.

Evaluate real-data quality with temporal splits, train-only counts, full-corpus Recall/NDCG, popularity buckets, coverage and concentration, with multiple seeds and uncertainty estimates. Match candidate pools, masks and softmax grouping across baselines. `lambda=0` alone does not reproduce ordinary pooled MNRL. The synthetic world tests finite-sample coverage, not a causal exposure policy.

## Optional synthetic experiment

After setup, run the comparison harness from the checkout:

```bash
mise exec -- uv run --frozen --no-sync python experiments/toy_two_tower.py --seeds 3 --out results.json
```

This optional long-running experiment is separate from `mise run check`. It creates synthetic click logs and reports KL, retrieval, tail and coverage metrics; its output file records arguments and seed-level results. Record the source version, environment and evaluation conditions alongside any retained output. The full default experiment has not been rerun for this revision, and no speedup or real-data advantage is claimed.

For experiment metrics, the head contains exactly `ceil(0.1 * M)` items sorted by descending popularity; ties use original item order. The tail is its complement, requiring at least two items. Top-k uses `min(k, M)` slots, including the denominator of unseen-item share. Invalid empty-training configurations and oversized uniform draws are rejected before training.
