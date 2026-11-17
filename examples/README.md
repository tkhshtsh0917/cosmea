# Examples

[日本語](README.ja.md)

## Train with your own data

Use [train_from_csv.py](train_from_csv.py). It loads an existing model, reads training pairs and a searchable collection, trains on CPU, and saves the result. It does not contain the model/tokenizer-building code used by the offline check.

First [install Cosmea with the train extra](../README.md#installation), then copy `train_from_csv.py` into your project. Prepare UTF-8 CSV files with the following columns. The rows below illustrate the format; they are not a useful training dataset on their own.

`train.csv`:

```csv
anchor,positive,query_id,document_id
python lists,A guide to Python lists,q1,p1
sql joins,An introduction to SQL joins,q2,p2
```

`corpus.csv`:

```csv
document_id,text
p1,A guide to Python lists
p2,An introduction to SQL joins
p3,A guide to garden flowers
```

The `anchor` is the query text, and the `positive` is the matching document text. The script also accepts `query` and `document` for those text columns and `item_id` for `document_id` in older CSVs.

All IDs are read as strings. If a positive ID also appears in the corpus, the texts should describe the same document.

```bash
uv run python train_from_csv.py \
  --model path/to/local/model \
  --train train.csv \
  --corpus corpus.csv \
  --output outputs/retriever \
  --uniform 2
```

The CSV script follows the README workflow: `plugin` with pooled expected counts, `BatchSamplers.BATCH_SAMPLER`, `mask_same_document=True` and `mask_same_query=True`. The library still defaults to yi when no variant is specified.

`--model` accepts a local Sentence Transformers model directory or a Hugging Face model ID. Using a remote ID may download its files. The uniform count cannot exceed the number of distinct corpus IDs. The trained model is saved to `outputs/retriever/model`; run folders can be selected with `--output`.

This is a training example, not a quality benchmark. Evaluate on separate queries against the full corpus before drawing conclusions about retrieval quality. See the [API guide](../docs/api.md) for the sampling assumptions and [Matryoshka usage](../docs/api.md#train-at-multiple-embedding-sizes) for shorter embeddings.

For a prompt-based model, set `--query-prompt "query: "` and `--document-prompt "passage: "` to the actual prefixes required by the model; these are examples, not universal defaults. `--max-length 256` sets the collator token limit. For Router tasks use the API guide and pass `router_mapping` directly to the collator. TrainingArguments alone does not transfer these settings.

The prompt flags configure training preprocessing; they do not rewrite the saved model's default prompts. Apply the same query/document prefixes when encoding after reload.

## Check the integration without downloads

[train_example.py](train_example.py) creates a tiny random BERT and synthetic data locally. It exercises ordinary and Cached losses and is used by CI. The extra tokenizer/model construction is for that purpose. Setup and execution are documented in [CONTRIBUTING.md](../CONTRIBUTING.md#offline-example-and-optional-experiment).

The optional comparison under `experiments/` measures loss choices in a synthetic world. It is a research tool, not needed to use the library; its command and limitations are also in CONTRIBUTING.
