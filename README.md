# Cosmea

[日本語](README.ja.md)

When a few documents receive most of the clicks, they can appear as negatives across many training batches. Searchable documents with no clicks may never appear in those batches. Cosmea is a Python library for fine-tuning Sentence Transformers search models with this kind of uneven feedback.

Cosmea is most relevant when training positives are concentrated in a small part of your searchable collection. If positive-document counts are already similar, there is less frequency imbalance for logQ to correct among those documents. Sampling from the full collection can still bring in documents missing from the training pairs.

## What changes from standard MNRL

For each query in a batch of positive query–document pairs, standard Multiple Negatives Ranking Loss (MNRL) uses the documents from other rows as negatives. Sentence Transformers recommends `NO_DUPLICATES` for this setup. See the [Sentence Transformers loss reference](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss).

Cosmea's recommended setup samples training rows at random. **Mixed Negative Sampling (MNS)** adds uniformly sampled documents from the full searchable collection to the in-batch candidates. **logQ correction** adjusts the loss using how often those candidates are expected to be sampled. The loss excludes known positives in the batch from the negatives. See [batch sampling and logQ](docs/api.md#choose-a-batch-sampler).

Use the `model`, training arguments `args`, and `train_dataset` from your Trainer script. In this example, `train_dataset` has exactly two text columns, `anchor` followed by `positive`. Supply the search collection as `corpus_texts` with at least four distinct document texts. Use the same text for a document in both inputs.

```python
from sentence_transformers.base.sampler import BatchSamplers

from cosmea import DocumentFrequency, LogQMultipleNegativesRankingLoss, MixedNegativeDataCollator

args.batch_sampler = BatchSamplers.BATCH_SAMPLER

collator = MixedNegativeDataCollator(
    preprocess_fn=model.preprocess,
    document_frequency=DocumentFrequency.from_dataset(train_dataset),
    corpus=corpus_texts,
    num_uniform=4,
)

loss = LogQMultipleNegativesRankingLoss(model)
```

Both the plain and Cached losses default to `plugin`, `pooled`, and both known-positive masks. Specify `variant="yi"` to reproduce the previous defaults. Pass `data_collator=collator` and `loss=loss` when creating the Trainer. If your model needs prompts, set `prompts` on the collator. For the full setup, explicit IDs, and how to use `NO_DUPLICATES` with Cosmea, see the [integration guide](docs/api.md).

## SciFact results

The CPU benchmark used the same `all-MiniLM-L6-v2` checkpoint and evaluated all conditions on the same 300 held-out queries. Each fine-tune used 256 training pairs per run, matched across methods. nDCG@10 measures how well the top 10 results rank relevant documents. Higher is better.

| nDCG@10 | Initial model | Standard MNRL | Cosmea |
| --- | ---: | ---: | ---: |
| Fixed pairs, varied training seed | 0.6239 | 0.6290 | 0.6334 |
| Fixed training seed, varied pairs | 0.6239 | 0.6279 | 0.6327 |

The fine-tuned scores are means over five runs per row. When the training pairs varied, Cosmea's difference from standard MNRL ranged from −0.0077 to +0.0170. On the configuration timed repeatedly, median training time was 8.82 s for Cosmea and 4.89 s for standard MNRL, about 1.8× as long with 16 additional documents sampled per batch. See the [SciFact benchmark](docs/benchmark.md) for the split, timing protocol, and comparison with logQ disabled.

## CPU and GPU validation

| Training setup | Project validation |
| --- | --- |
| CPU, one training process | Verified by integration tests. The quickstart and CSV example default to CPU. |
| Apple MPS, one GPU | Verified for FP32 CSV training with the pooled `plugin` loss, followed by model save and reload. See the [CSV example](examples/README.md#choose-a-device). |
| CUDA, one GPU | The CSV example accepts `--device cuda`. End-to-end training has not been verified. |
| Distributed or mixed-precision training | End-to-end training has not been verified. |

See the [design guide](docs/design.md#resource-and-compatibility-limits) for supported dependency versions and resource limits.

## Quickstart

Fine-tune a small English model on three example pairs, save it, and use the saved model to search six documents. Run these commands in Bash or a compatible shell with [uv](https://docs.astral.sh/uv/), Git and curl installed. The first run downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. On Linux, the default PyTorch package can pull in several GB of GPU libraries even though the example trains on CPU.

```bash
uv init --python 3.12 cosmea-quickstart
cd cosmea-quickstart

uv add "cosmea[train] @ git+https://github.com/tkhshtsh0917/cosmea"
curl --fail --location --output quickstart.py https://raw.githubusercontent.com/tkhshtsh0917/cosmea/main/examples/quickstart.py

uv run python quickstart.py
```

Look for `Saved model:` and `Top result:` in the output. The script saves the model to `outputs/quickstart/model`, reloads it, and prints the top document for a question about bicycle tires. You can read the training pairs and corpus in [quickstart.py](examples/quickstart.py).

The script also prints `nDCG@3` and `Recall@3` for the initial and fine-tuned models on the same two held-out queries and the same corpus. nDCG rewards relevant documents near the top, and Recall measures how many known relevant documents appear in the top three. The example runs one training step. Unchanged scores are normal, and the comparison shows how to evaluate retrieval. Use your own held-out queries and relevance judgments to assess quality.

To use an existing local Sentence Transformers model instead of downloading one, pass `--model <directory>` to the script. If you already have a Python project, run the install command there. The `train` extra provides the Trainer dependencies. uv records the Git revision in `uv.lock`. When pinning a commit with `uv add --rev <commit-sha>`, replace `main` in the curl URL with that same SHA.

## Use your own data

- [Train from CSV](examples/README.md#train-with-your-own-data) takes query–document pairs and a corpus with document IDs.
- [Integrate with Sentence Transformers](docs/api.md) explains how to prepare a dataset, connect the collator and loss, and evaluate retrieval.
- [Mine hard negatives and rerank](examples/README.md#mine-hard-negatives-train-and-rerank) adds hard negatives during training and measures the effect of a CrossEncoder on the trained retriever's top candidates.

## Documentation

- [Public API](docs/reference.md) lists the classes and functions, with links to their source and instructions for rendering the API reference.
- [Design](docs/design.md) explains the correction, sampling assumptions, loss variants and their limits.
- [Contributing](CONTRIBUTING.md) covers setup, checks and documentation builds.

Cosmea is an Apache-2.0 development project. The project name comes from *Cosmea*, another name for cosmos flowers. See [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for licensing and attribution.
