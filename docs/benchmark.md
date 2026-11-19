# Small SciFact benchmark

[日本語](ja/benchmark.md) · [Design](design.md)

SciFact retrieval ranks scientific abstracts that can support or refute an input claim. This experiment uses the [BEIR SciFact archive](https://github.com/beir-cellar/beir/wiki/Datasets-available) and the original [SciFact annotations](https://github.com/allenai/scifact). Obtain the data separately under the [dataset's terms](https://github.com/allenai/scifact/blob/master/LICENSE.md).

## Run the comparison

Set up the development environment as described in [Contributing](../CONTRIBUTING.md#setup-and-validation), then run:

```bash
mkdir -p outputs/scifact
curl --fail --location \
  --output outputs/scifact/scifact.zip \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

mise exec -- env HF_HUB_OFFLINE=0 uv run --frozen --no-sync python experiments/scifact_benchmark.py \
  --data outputs/scifact/scifact.zip \
  --output outputs/scifact/reviewed-results.json \
  --train-pairs 256 \
  --data-seed 42 47 48 49 50 \
  --seed 42 44 45 46 48 \
  --timing-repeats 4 \
  --threads 4
```

The two seed lists vary one factor at a time. The first data seed fixes the 256 pairs while the training seed changes. The first training seed then stays fixed while the data seed selects different pairs. The shared first combination is evaluated once. Omitting the seed lists and timing repeats runs one combination.

The script checks the MNRL batch schedule before loading a model. With duplicates, the sampler can produce an extra batch beyond the Trainer's planned steps. The first attempt at training seed 43 used 255 of 256 rows in 16 steps. For each sweep, we selected the first five seeds from 42 upward that gave all conditions the same rows and step count, using the sampler schedule alone. Results describe those feasible combinations.

## Data and evaluation

The first experiment's sample contained 74 training pairs whose positive document was also relevant to a test query. This included closely related claims. The revised experiment retains all 300 official test queries and 339 positive judgments. From the 919 training pairs, it removes two with matching query text after whitespace normalization and case folding, then 293 more whose positive document appears in the test judgments. Training samples come from the 624 eligible pairs.

Each condition searches the same 5,183 documents by cosine similarity, using each title followed by its abstract. Test-positive documents remain in the corpus and can occur as unlabeled uniform negatives. Test judgments supply the positive-pair filter and the evaluation labels. This is a positive-document-holdout variant of SciFact. Its filter checks query text and document ID, so semantically related texts under other IDs may still occur.

All conditions load the same [all-MiniLM-L6-v2 revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41). The 128-token limit applies to queries and documents during both training and evaluation, including the initial model. It truncates 5,073 of 5,183 documents with this checkpoint's tokenizer. The checkpoint's [default limit](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#intended-uses) is 256. Metrics use binary relevance and average equally over test queries.

## Training conditions

Each fine-tune uses 256 identical positive pairs within each seed combination, one epoch, batch size 16 and a starting learning rate of `2e-5`.

| Condition | Row sampler | Known-positive masks | Uniform negatives per batch | logQ strength |
| --- | --- | --- | ---: | ---: |
| Base | — | — | — | — |
| A · Standard MNRL | `NO_DUPLICATES` | — | 0 | 0 |
| B · In-batch control | Random | Both | 0 | 0 |
| C · Uniform control | Random | Both | 16 | 0 |
| D · Cosmea | Random | Both | 16 | 1 |

The standard workflow uses Sentence Transformers' default MNRL loss and its [recommended sampler](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss). B, C and D use Cosmea's pooled `plugin` loss with the same positive batches. The script checks the actual number of training rows and optimizer steps. It also verifies that C and D received identical query and candidate IDs in each batch. A to B changes the sampler and masks together. B to C adds uniform negatives. C to D enables logQ while holding the candidate inputs fixed.

The first seed combination repeats training four times, twice in each order. The table reports the median wall time of `Trainer.train()`, including startup, tokenization and optimization. Model loading, collator setup and evaluation are outside the timer. Padded input tokens quantify the amount of text processed. Training time also depends on sequence lengths, batching, and hardware.

The JSON retains the sample IDs, source commit and file hashes, per-query metrics, paired differences, training rows, steps, timing repeats and padded token counts.

## Reviewed benchmark results

Run on 2026-11-19 with macOS arm64, Python 3.12.13, Sentence Transformers 6.3.0, PyTorch 2.15.0, and four CPU threads. Nine distinct seed combinations completed. Each table shows the mean and sample standard deviation of five runs. Both tables use the same base evaluation.

### Fixed training pairs, varied training seeds

| Condition | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6239 | 0.5784 | 0.7743 | 0.9227 |
| A · Standard MNRL | 0.6290 ± 0.0055 | 0.5854 ± 0.0069 | 0.7781 ± 0.0072 | 0.9223 ± 0.0051 |
| B · In-batch control | 0.6281 ± 0.0035 | 0.5846 ± 0.0042 | 0.7769 ± 0.0060 | 0.9220 ± 0.0040 |
| C · Uniform control | 0.6276 ± 0.0015 | 0.5843 ± 0.0019 | 0.7743 ± 0.0064 | 0.9247 ± 0.0099 |
| D · Cosmea | 0.6334 ± 0.0033 | 0.5880 ± 0.0033 | 0.7877 ± 0.0055 | 0.9207 ± 0.0038 |

### Fixed training seed, varied training pairs

| Condition | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6239 | 0.5784 | 0.7743 | 0.9227 |
| A · Standard MNRL | 0.6279 ± 0.0120 | 0.5846 ± 0.0099 | 0.7762 ± 0.0194 | 0.9283 ± 0.0069 |
| B · In-batch control | 0.6281 ± 0.0095 | 0.5841 ± 0.0075 | 0.7789 ± 0.0153 | 0.9277 ± 0.0061 |
| C · Uniform control | 0.6285 ± 0.0074 | 0.5846 ± 0.0068 | 0.7783 ± 0.0106 | 0.9257 ± 0.0066 |
| D · Cosmea | 0.6327 ± 0.0042 | 0.5875 ± 0.0046 | 0.7867 ± 0.0051 | 0.9258 ± 0.0049 |

The paired nDCG@10 difference from enabling logQ (D minus C) averaged +0.0058 with fixed training pairs (range +0.0016 to +0.0098). With varied training pairs it averaged +0.0042 (range −0.0036 to +0.0157), so the direction changed. Cosmea minus standard MNRL also ranged from −0.0077 to +0.0170 when the training pairs varied. These estimates describe the tested seeds and SciFact setting.

The absolute mean paired nDCG@10 difference for A to B and for B to C was below 0.001 in both sweeps. Random batches had one to six repeated positive-document rows and zero to three repeated query rows per epoch. Those counts describe repetitions within the positive rows. Uniform candidates have their own masking.

### Training time and input size

| Condition | Median training time in seconds (min–max) | Padded input tokens per epoch |
| --- | ---: | ---: |
| A · Standard MNRL | 4.89 (4.78–5.15) | 42,192 |
| B · In-batch control | 4.85 (4.81–4.99) | 42,192 |
| C · Uniform control | 9.03 (8.76–9.34) | 74,960 |
| D · Cosmea | 8.82 (8.61–8.96) | 74,960 |

C and D processed about 1.8 times as many input tokens as standard MNRL in this setting. D took about 1.8 times as long as A by median wall time. C and D had equal token counts and overlapping timing ranges. These timings describe batch size 16, 16 uniform negatives, and the 128-token limit.

The data archive SHA-256 was `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
