# How Cosmea works

[日本語](ja/design.md) · [API guide](api.md)

## Why mix negative examples?

A retrieval model learns to give a query's positive document a higher score than comparison documents. Using the other positives in the same batch is convenient, but frequently clicked documents then appear more often as comparisons. Items without clicks may never appear at all.

Cosmea combines two sources: the other positives in the batch, and documents drawn uniformly from the searchable collection. This is **Mixed Negative Sampling (MNS)**. The uniform part can include documents without feedback.

**logQ correction** adjusts scores according to how candidates were sampled. It requires both sampling information and a loss calculation, so a sampler alone is not enough. The row sampler chooses training examples, the collator adds corpus candidates and counts, and the loss applies the correction.

This corrects how negative examples are drawn. It does not identify documents that users would have clicked if they had been shown something else. Exposure bias in the positive log therefore needs separate treatment.

## “How many times” and “at least once” are different

Suppose we draw two entries from `[a, b, b]` without replacement. On average, a appears $2/3$ times and b appears $4/3$ times. The chance of appearing at least once is $2/3$ for a and $1$ for b.

Cosmea supports both calculations:

- `expected_count` keeps repeated occurrences and divides their contributions by the expected count.
- `inclusion` keeps one eligible occurrence per document and divides by its chance of appearing at least once.

Mixing these rules gives the wrong weight. For example, keeping only one b and still dividing by $4/3$ makes its contribution too small. Candidates excluded by a mask are removed before deduplication.

The standard usage examples combine uniform random row sampling with document and query masks. This uses the declared row proposal while excluding batch-known positives. Query masking is a heuristic and the masked yi objective differs from its unmasked default. API defaults and objective comparisons are described in the [API guide](api.md#avoid-treating-known-matches-as-negatives).

## Three loss options

| Variant | What changes | How to interpret it |
| --- | --- | --- |
| `yi` (default) | Corrects positive and negative scores. By default, computes and averages losses for the two sources separately | Uses an independent-draw approximation based on training counts |
| `plugin` | Keeps the positive score uncorrected and estimates the total negative contribution | A useful way to match the correction to the actual draw procedure |
| `improved` | Multiplies a score difference by a confidence weight whose gradient is stopped | Has the same first gradient as plugin with identical settings; loss values and higher derivatives differ |

The row sampler draws training rows without replacement; the collator independently draws distinct corpus documents. Thus, a consistency result proved for independent draws does not automatically apply to the default yi implementation. yi does not support `inclusion`. With a separate uniform yi loss, every positive must be present in the corpus.

Under suitable sampling assumptions, the average estimate of the total negative contribution is correct. This property is called **unbiasedness**. Taking a logarithm or a ratio can introduce bias again, so it does not imply unbiased loss gradients. Query-based filtering and manually supplied negatives also change the assumptions.

Under ideal independent sampling and a sufficiently expressive model, logQ correction targets the document distribution in the observed positive log; uncorrected learning targets a probability ratio. Neither statement establishes lower popularity concentration in a real application.

## Mined negatives and loss monitoring

In `separate` mode, in-batch, uniform and extra mined candidates form independent objectives. Extra candidates are shared across queries, receive no correction, and use an uncorrected positive. Only objectives with eligible negatives enter the average. A singleton batch with mined negatives therefore remains trainable. This changes the initial development behavior, which grouped mined candidates with in-batch candidates.

In `pooled` mode, mined terms keep weight 1 while sampled terms receive $1/q$ weighting. With rare documents or a larger training population, mined terms can have much less relative influence. Choose separate mode when you want an independent mined-negative objective; its real-data quality still needs evaluation. If there are no in-batch or uniform draws at all, pooled mined candidates use an uncorrected positive as well. A sampled yi objective with zero positive support remains an error.

`improved` is a research option, not an extra source of first-order gradient improvement over plugin. Its scalar value is non-monotonic in the margin and can be negative. Do not use its `eval_loss` for early stopping or best-checkpoint selection. Use held-out retrieval metrics, or train with plugin when a comparable loss value is needed. `scale=20` is a starting value, not a calibrated choice for the catalogue-sized partition estimate; tune it against retrieval metrics.

`inclusion` remains available for comparisons that require deduplication. It has higher compute cost and no established quality or variance advantage over expected counts in general. Start with `expected_count`.

## Compare configurations fairly

To compare against ordinary pooled MNRL, use `logq_strength=0` and `combine="pooled"`, and match the candidate pool and masks. Setting the correction strength to zero alone does not change separate losses into one pooled loss.

For quality evaluation, use training-only counts, held-out queries and the full corpus. With time-dependent logs, split by time. Report retrieval quality such as Recall/NDCG, results for popular and less popular documents, coverage and concentration. Repeat across seeds and include uncertainty estimates. Synthetic checks verify calculations and interfaces; real-data improvement is unverified.

The optional synthetic experiment is described in [CONTRIBUTING.md](../CONTRIBUTING.md#offline-example-and-optional-experiment). Its head is the top $\lceil 0.1M\rceil$ documents by popularity, with ties resolved by original document order; the tail is the remainder. At least two documents are required. Top-k uses $\min(k,M)$ slots, including the denominator of unseen-document share. Invalid settings that give no training steps or too many uniform samples are rejected.

## Resource and compatibility limits

The collection and its ID index are held in memory. There is no streaming corpus or automatic memory quota. The batch must contain between 1 and $N$ rows; the uniform sample count must be between 0 and $M$. Invalid IDs, counts and options raise errors. NaN/Inf scores also raise an error, even in masked columns, so invalid model outputs cannot look like a zero loss. Empty negative sets give zero loss and gradient. Separate mode averages only sources with usable negatives.

With $B$ query rows and $C$ candidates, ordinary score memory is $O(BC)$. Cached mode computes $b$ query rows at a time, so score memory is $O(bC)$; candidate embeddings still stay cached. Masking also uses $O(bC)$ working memory: repeated queries collect their known document IDs without constructing a full $B\times C$ match matrix. This bounds working memory, not the time needed for many repeated queries. The ID index uses $O(M)$ memory in addition to text. Exact inclusion takes $O((B-1)bC)$ arithmetic per chunk, so expected counts are the initial choice for large batches.

Half-precision scores are promoted inside the loss to make the calculation more stable. Mixed-precision model training remains unverified.

Cosmea uses Sentence Transformers 6.3 collator helpers and GradCache. Supported ranges are ST `>=6.3,<6.4` and PyTorch `>=2.6,<3`, with NumPy `>=2,<3`; ST v5 compatibility is not claimed. CPU with one training process is verified. GPU, DDP, cross-device candidate sharing, streaming counts and multiple datasets need additional sampling definitions or validation. In-batch logQ with duplicate-constrained, weighted or grouped row sampling requires a proposal matched to that sampler. See [batch sampling and logQ](api.md#choose-a-batch-sampler).

## Formula reference

You can use the library without implementing these formulas. They explain precisely which weights the loss uses.

### Candidate weights

| Symbol | Meaning |
| --- | --- |
| $N, B$ | Total training rows / rows in the batch |
| $M, K$ | Distinct corpus documents / uniform draws |
| $p_i$ | Positive document for query row $i$ |
| $c_d$ | Number of training rows whose positive is document $d$ |
| $\mathbf{1}[A]$ | 1 when condition $A$ holds, otherwise 0 |

After fixing row $i$, the remaining $B-1$ rows are sampled from $N-1$ rows. The remaining count is $c'_{id}=c_d-\mathbf{1}[d=p_i]$. Corpus draws are independent of the row draw. For plugin/improved, the pooled expected count and inclusion probability are:

$$
e_i(d)=\frac{(B-1)c'_{id}}{N-1}+\frac{K}{M}\mathbf{1}[d\in\mathrm{corpus}]
$$

$$
\pi_i^{\mathrm{batch}}(d)=1-\frac{\binom{N-1-c'_{id}}{B-1}}{\binom{N-1}{B-1}},\qquad
\pi_i(d)=1-\left(1-\pi_i^{\mathrm{batch}}(d)\right)
\left(1-\frac{K}{M}\mathbf{1}[d\in\mathrm{corpus}]\right).
$$

The in-batch term is defined as zero when $B=N=1$, and the corpus term as zero when $K=0$. These row-dependent weights form a $(b,C)$ matrix for a chunk. The negative-total estimate is unbiased only over the supported document set, with positive-document exclusion and the specified sampling procedure. A supported document has a nonzero chance of being drawn.

### Corrected scores and gradients

Let $F_{id}=\mathrm{scale}\cdot\mathrm{similarity}(i,d)$. Scores have shape $(B,C)$, where $C=B(1+H)+K$ and $H$ is the number of extra negative columns. Correction is subtracted **after** scaling:

$$
\widetilde F_{id}=F_{id}-\lambda\log q_i(d),\qquad \lambda=\texttt{logq\_strength}.
$$

For the negative terms, plugin/improved use $e_i(d)$ or $\pi_i(d)$ as $q_i(d)$, according to `q_measure`. The yi expected-count calculation instead uses the empirical independent-draw probability $c_d/N$, combined with source counts and corpus sampling. Extra mined-negative columns receive no correction.

Let $\mathcal V_i$ be the eligible negative entries after masking and, for inclusion mode, deduplication. The corrected negative sum and its logarithm are:

$$
\widehat Z_i^{-}=\sum_{d\in\mathcal V_i}\exp(\widetilde F_{id}),\qquad z_i=\log\widehat Z_i^{-}.
$$

In pooled mode, $\mathcal V_i$ includes all candidate sources and one correction strength applies to the pooled proposal. In separate mode, each source is normalized on its own before averaging active losses. The in-batch and uniform strengths can be set independently. For a uniform sample of $K$ distinct documents from a corpus of $M$, every corpus document has expected count and inclusion probability $K/M$, even when the row sampler constrains duplicates. Setting the in-batch strength to zero therefore skips the row proposal while retaining the uniform correction.

For plugin and improved, with positive score $F_{i,p_i}$:

$$
\ell_i^{\mathrm{plugin}}=\log\left(1+\exp(z_i-F_{i,p_i})\right),\qquad
\ell_i^{\mathrm{improved}}=\operatorname{sg}\!\left[\sigma(z_i-F_{i,p_i})\right](z_i-F_{i,p_i}).
$$

Here $\sigma(x)=1/(1+e^{-x})$ and $\operatorname{sg}$ stops the gradient through the confidence weight. Both give the same first derivative for identical negative sums. This is an adaptation of the confidence-weighted formulation, not a literal reproduction of the paper's Q-prime or author scripts. Empty negative sets are handled separately as zero loss and gradient.
