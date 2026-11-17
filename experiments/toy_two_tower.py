"""Synthetic two-tower experiment with a known ground truth p(document | query).

Purpose: check, with exact metrics, what each loss configuration converges to at a *finite*
batch size. It drives the functional core (`cosmea.functional.mixed_negative_logq_loss`)
with a free embedding-table two-tower model; no pretrained model is downloaded.

World
-----
* M documents with Zipf popularity, U queries, true logits  g(u, d) = <x_u, y_d> + log pop(d).
* Click log: N rows, u uniform, d ~ softmax(g(u, .)). Tail documents end up with zero clicks, which
  is a finite-sample coverage limitation; no causal exposure policy is simulated.
* Model: dot product of dim-`rank` embeddings (default) or scale * cosine (`--cosine`).

Metrics (exact expectations under the true distribution, no sampled evaluation)
* KL:      mean_u KL(p(.|u) || softmax(f(u, .)))                (lower is better; target of lambda=1)
* R@k:     expected recall@k of a fresh click
* tailR@k: the same, restricted to clicks on documents outside the top 10% by popularity
* ipwR@10: recall@10 with clicks weighted by 1/pop(d) (a popularity-debiased test set)
* unseen@10: share of top-10 slots taken by documents that have no click in the log

    python experiments/toy_two_tower.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from cosmea.functional import (
    SOURCE_IN_BATCH,
    SOURCE_UNIFORM,
    mixed_negative_logq_loss,
    negative_log_partition,
)

CONFIGS = {
    # name: (num_uniform / batch_size, loss kwargs)   None => full softmax reference
    "full softmax (reference)": None,
    "in-batch, no correction": (0, dict(logq_strength=0.0, combine="pooled")),
    "in-batch, yi": (0, dict(variant="yi")),
    "in-batch, yi, hits masked": (0, dict(variant="yi", mask_same_document=True)),
    "in-batch, plugin (=improved)": (0, dict(variant="plugin")),
    "MNS, no correction": (1, dict(logq_strength=0.0, combine="pooled")),
    "MNS, yi, separate": (1, dict(variant="yi", combine="separate")),
    "MNS, yi, separate, hits masked": (1, dict(variant="yi", combine="separate", mask_same_document=True)),
    "MNS, yi, pooled": (1, dict(variant="yi", combine="pooled")),
    "MNS, yi, separate, lambda=0.5": (1, dict(variant="yi", combine="separate", logq_strength=0.5)),
    "MNS, plugin (=improved), pooled": (
        1,
        dict(variant="plugin", combine="pooled", mask_same_document=True, mask_same_query=True),
    ),
    "MNS, plugin, separate": (1, dict(variant="plugin", combine="separate")),
    "MNS, plugin, pooled, inclusion": (1, dict(variant="plugin", combine="pooled", q_measure="inclusion")),
    # Diagnostic for sensitivity to Q scale; not a reproduction of the author repository.
    "MNS, uncorrected positive + per-draw Q": (1, dict(per_draw=True)),
}


def make_world(args, seed):
    rng = np.random.default_rng(seed)
    pop = 1.0 / np.arange(1, args.documents + 1) ** args.zipf
    pop /= pop.sum()

    x = rng.normal(0, 1.0, (args.queries, args.true_rank))
    y = rng.normal(0, 1.0, (args.documents, args.true_rank))
    logits = x @ y.T * args.affinity + np.log(pop)[None, :]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)

    row_query = rng.integers(0, args.queries, args.rows)
    cdf = p.cumsum(1)
    row_document = (rng.random(args.rows)[:, None] > cdf[row_query]).sum(1).clip(max=args.documents - 1)

    return pop, p, row_query, row_document


class TwoTower(torch.nn.Module):
    def __init__(self, args, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q = torch.nn.Parameter(torch.randn(args.queries, args.rank, generator=g) * 0.1)
        self.d = torch.nn.Parameter(torch.randn(args.documents, args.rank, generator=g) * 0.1)
        self.cosine, self.scale = args.cosine, args.scale

    def scores(self, query_index, document_index):
        q, d = self.q[query_index], self.d[document_index]
        if self.cosine:
            q, d = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(d, dim=-1)
            return q @ d.T * self.scale
        return q @ d.T

    def all_scores(self):
        return self.scores(torch.arange(self.q.size(0)), torch.arange(self.d.size(0)))


def evaluate(model, pop, p, counts):
    """Evaluate full-corpus ranks; ties in popularity use original document order.

    The head is exactly ceil(10% of documents), leaving a nonempty tail for M >= 2.
    Top-k metrics use at most M slots when the corpus is smaller than k.
    """
    if len(pop) < 2:
        raise ValueError("Evaluation requires at least two documents to define a tail.")

    with torch.no_grad():
        f = model.all_scores().double()
    p_t = torch.tensor(p)
    log_model = torch.log_softmax(f, dim=1)
    kl = (p_t * (torch.log(p_t.clamp(min=1e-300)) - log_model)).sum(1).mean().item()
    order = f.argsort(dim=1, descending=True)
    out = {"KL": kl}
    head_size = math.ceil(len(pop) * 0.1)
    head = np.argsort(-np.asarray(pop), kind="stable")[:head_size]
    tail = torch.ones(len(pop), dtype=torch.bool)
    tail[torch.as_tensor(head)] = False
    ipw = p_t / torch.tensor(pop)[None, :]
    for k in (10, 100):
        top = torch.zeros_like(p_t, dtype=torch.bool).scatter_(1, order[:, :k], True)
        out[f"R@{k}"] = (p_t * top).sum(1).mean().item()
        out[f"tailR@{k}"] = ((p_t * top)[:, tail].sum() / p_t[:, tail].sum()).item()
        if k == 10:
            out["ipwR@10"] = ((ipw * top).sum() / ipw.sum()).item()
            unseen = torch.tensor(counts == 0)
            out["unseen@10"] = top[:, unseen].sum(1).double().mean().item() / min(k, len(pop))
    return out


def train(name, args, seed, pop, p, row_query, row_document):
    config = CONFIGS[name]
    torch.manual_seed(seed)
    model = TwoTower(args, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    counts = np.bincount(row_document, minlength=args.documents)
    counts_t = torch.tensor(counts, dtype=torch.int64)
    query_ids, document_ids = torch.tensor(row_query), torch.tensor(row_document)
    rng = np.random.default_rng(seed + 1)
    steps = args.epochs * (args.rows // args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0 - 0.9 * s / steps)
    all_documents = torch.arange(args.documents)

    for epoch in range(args.epochs):
        perm = torch.tensor(rng.permutation(args.rows))
        for start in range(0, args.rows - args.batch_size + 1, args.batch_size):
            rows = perm[start : start + args.batch_size]
            if config is None:
                loss = torch.nn.functional.cross_entropy(
                    model.scores(query_ids[rows], all_documents), document_ids[rows]
                )
            else:
                factor, kwargs = config
                num_uniform = factor * args.batch_size
                uniform = torch.tensor(rng.choice(args.documents, size=num_uniform, replace=False), dtype=torch.long)
                cand = torch.cat([document_ids[rows], uniform])
                label = {
                    "query_id": query_ids[rows],
                    "cand_id": cand,
                    "cand_count": counts_t[cand],
                    "cand_in_corpus": torch.ones_like(cand, dtype=torch.bool),
                    "cand_source": torch.cat(
                        [
                            torch.full((args.batch_size,), SOURCE_IN_BATCH),
                            torch.full((num_uniform,), SOURCE_UNIFORM),
                        ]
                    ),
                    "num_rows": torch.tensor(args.rows),
                    "corpus_size": torch.tensor(args.documents),
                }
                scores = model.scores(query_ids[rows], cand)
                if kwargs.get("per_draw"):
                    log_z = negative_log_partition(scores, label, mask_same_document=True) + math.log(len(cand) - 1)
                    pos = scores.diagonal()
                    loss = (torch.logaddexp(pos, log_z) - pos).mean()
                else:
                    loss = mixed_negative_logq_loss(scores, label, **kwargs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    return evaluate(model, pop, p, counts)


def validate_args(parser, args, names):
    """Reject configurations with no training steps or an impossible negative draw."""
    for name in ("queries", "rows", "true_rank", "rank", "batch_size", "epochs", "seeds"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.documents < 2:
        parser.error("--documents must be at least 2 to define tail metrics")
    if args.rows < args.batch_size:
        parser.error("--rows must be at least --batch-size")
    for name in ("zipf", "affinity", "lr", "scale"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    if args.lr <= 0 or args.scale <= 0:
        parser.error("--lr and --scale must be positive")

    for name in names:
        if name not in CONFIGS:
            parser.error(f"Unknown configuration: {name}")
        config = CONFIGS[name]
        if config is not None and config[0] * args.batch_size > args.documents:
            parser.error("Uniform negative count must not exceed --documents; reduce --batch-size")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--documents", "--items", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--zipf", type=float, default=1.0)
    parser.add_argument("--affinity", type=float, default=1.0)
    parser.add_argument("--true-rank", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    names = args.only or list(CONFIGS)
    validate_args(parser, args, names)

    results = {name: [] for name in names}
    for seed in range(args.seeds):
        pop, p, row_query, row_document = make_world(args, seed)
        unseen = (np.bincount(row_document, minlength=args.documents) == 0).mean()
        for name in names:
            start = time.time()
            results[name].append(train(name, args, seed, pop, p, row_query, row_document))
            print(f"seed {seed} {name:40s} {time.time() - start:5.1f}s {results[name][-1]}", flush=True)

    print(
        f"\nmodel: {'cosine x %g' % args.scale if args.cosine else 'dot product'}; "
        f"documents without clicks: {unseen:.1%}; mean +- std over {args.seeds} seeds\n"
    )
    metrics = ["KL", "R@10", "R@100", "tailR@10", "tailR@100", "ipwR@10", "unseen@10"]
    print("| config | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for name in names:
        cells = []
        for metric in metrics:
            values = np.array([run[metric] for run in results[name]])
            cells.append(f"{values.mean():.3f} ± {values.std():.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle, indent=1)


if __name__ == "__main__":
    main()
