# Examples

[日本語](README.ja.md)

## Train with your own data

Use [train_from_csv.py](train_from_csv.py). It loads an existing model, reads training pairs and a searchable collection, trains on CPU, and saves the result. It does not contain the model/tokenizer-building code used by the offline check.

First [install Cosmea with the train extra](../README.md#installation), then copy `train_from_csv.py` into your project. Prepare UTF-8 CSV files with the following columns. UTF-8 files with a BOM work too. The rows below illustrate the format.

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

## Mine hard negatives, train, and rerank

[mine_train_rerank.py](mine_train_rerank.py) runs the full pipeline on CPU. Install the train extra as above and keep this script next to [train_from_csv.py](train_from_csv.py), which supplies the CSV reader. Pass an existing SentenceTransformer and a pretrained single-score CrossEncoder. Both arguments accept a local model directory or a Hugging Face model ID.

Use the same `corpus.csv` format as above, with a unique string ID per document. Here `train.csv` and `eval.csv` both use `query_id,query,document_id`, one known relevant document per row. Within each file, use one ID per exact query text and one text per ID. Add more rows with the same ID and text for multiple relevant documents. Training and evaluation queries must have distinct IDs and distinct texts. Every `document_id` must refer to a row in `corpus.csv`. For a small smoke run, use the corpus above and these files.

`train.csv`:

```csv
query_id,query,document_id
q1,python list operations,p1
q2,how to join SQL tables,p2
```

`eval.csv`:

```csv
query_id,query,document_id
q3,flower gardening,p3
```

To inspect the mined negatives before training, run the mining step on its own.

```bash
uv run python mine_train_rerank.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --train train.csv \
  --corpus corpus.csv \
  --output outputs/two-stage-preview \
  --mine-only
```

Check the IDs in `outputs/two-stage-preview/mined.csv`. If a negative is actually relevant, add its ID as a known positive in `train.csv`. When you run the full pipeline, use the same mining inputs and settings. It mines again from the input CSVs, so make any judgment edits in `train.csv`. Use a fresh output directory for each run.

```bash
uv run python mine_train_rerank.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --reranker cross-encoder/ms-marco-MiniLM-L6-v2 \
  --train train.csv \
  --eval eval.csv \
  --corpus corpus.csv \
  --output outputs/two-stage \
  --uniform 1 \
  --candidates 2 \
  --k 1
```

1. **Mine.** Use the initial retriever to pick the closest eligible corpus document for each training query. Exclude all of that query's known positives and their exact-text copies. An unjudged near match can still be relevant; review the mined IDs before treating them as reliable negatives.
2. **Train.** Use Cosmea with random row batches and `plugin` / `separate`: correct in-batch and uniform negatives, and give mined negatives a separate uncorrected term. The mined columns are shared across the batch, so mining-time exclusions do not protect positives for every *other* query. Cosmea's known-positive masks at training time are batch-local.
3. **Evaluate.** Search the full corpus with the trained retriever. Reorder its top `--candidates` using the fixed CrossEncoder. Never add missed positives. Compare mean nDCG@k, Recall@k and MRR@k at the same `--k`, with all known relevant IDs as the reference. Candidate recall is the fraction of relevant IDs the first stage retrieved, averaged across queries.

The script saves `mined.csv` with the training query, positive and negative IDs before training starts. The full run also saves `model/`, `metrics.json`, and `rankings.jsonl`. Both rankings use the trained retriever's candidates. The `before` ranking uses cosine similarity, and the `after` ranking uses CrossEncoder logits. Use ranks and metrics to compare the two stages.

Learning masks and ranking metrics use document IDs. Identical text under different IDs can still act as a negative during training or receive different relevance credit in evaluation. If the IDs refer to the same document, deduplicate the corpus and update the judgments before running the example. Cosine ties follow corpus order. CrossEncoder ties keep the candidate order from the retriever.

The example uses exact cosine search for small corpora and binary judgments. Unjudged documents receive no relevance credit. Use a representative held-out set to assess quality. The sample data only checks that the workflow runs.

By default the script uses the model's `query` and `document` prompts, with an empty prefix when the corresponding prompt is absent. If the checkpoint uses a different key such as `passage` or `corpus`, pass its prefix text with `--document-prompt`. Use `--query-prompt` to override the query prefix. An empty string explicitly disables the corresponding prefix. The same prefixes apply during mining, training and evaluation. The saved model retains its configured defaults, so pass the same overrides when encoding after reload.

## Check the integration without downloads

[train_example.py](train_example.py) creates a tiny random BERT and synthetic data locally. It exercises ordinary and Cached losses and is used by CI. The extra tokenizer/model construction is for that purpose. Setup and execution are documented in [CONTRIBUTING.md](../CONTRIBUTING.md#offline-example-and-optional-experiment).

The optional comparison under `experiments/` measures loss choices in a synthetic world. It is a research tool, not needed to use the library; its command and limitations are also in CONTRIBUTING.
