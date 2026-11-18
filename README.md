# Cosmea

[日本語](README.ja.md)

Cosmea adds two tools for fine-tuning a Sentence Transformers model for search. **Mixed Negative Sampling** adds comparison documents drawn from the searchable corpus, including documents without clicks. **logQ correction** adjusts the training loss for differences in how often documents are sampled.

Cosmea is an Apache-2.0 development project. The examples run on CPU, and the [design guide](docs/design.md#resource-and-compatibility-limits) covers the supported setup. The tests verify the loss and training integration. Retrieval gains on real data still need to be measured.

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

The project name comes from *Cosmea*, another name for cosmos flowers. See [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for licensing and attribution. Install the development version from GitHub.
