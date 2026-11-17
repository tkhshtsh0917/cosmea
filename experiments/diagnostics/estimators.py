"""Monte Carlo check that Z_hat_neg is an unbiased estimate of the negative partition function.

The simulation reproduces the actual batch construction: one fixed row plus B-1 other rows drawn
without replacement from the training set, plus K corpus documents drawn uniformly without
replacement. For the fixed row with query u and positive p the target is

    Z_neg = sum_{d != p} exp(f(u, d))      over every document of the training set or the corpus.
"""

import numpy as np
import torch

from cosmea.functional import (
    SOURCE_IN_BATCH,
    SOURCE_UNIFORM,
    _first_occurrence,
    candidate_log_q,
    negative_log_partition,
)

NUM_ROWS, CORPUS_SIZE, NUM_TRAIN_DOCUMENTS = 4000, 300, 210  # train documents 200..209 are not in the corpus
BATCH_SIZE, NUM_UNIFORM = 48, 16


def build_world(seed=0):
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, NUM_TRAIN_DOCUMENTS + 1) ** 1.1  # Zipf: head document holds ~19% of rows
    popularity /= popularity.sum()
    row_document = rng.choice(NUM_TRAIN_DOCUMENTS, size=NUM_ROWS, p=popularity)
    num_documents = max(CORPUS_SIZE + 20, NUM_TRAIN_DOCUMENTS)
    in_corpus = np.zeros(num_documents, dtype=bool)
    in_corpus[:200] = True  # train documents 0..199
    in_corpus[220:320] = True  # 100 documents that never occur in training
    counts = np.bincount(row_document, minlength=num_documents)
    scores = rng.normal(0.0, 1.5, size=num_documents)
    return row_document, counts, in_corpus, scores


def simulate(fixed_row, q_measure, trials, mismatched=False, seed=1):
    row_document, counts, in_corpus, scores = build_world()
    corpus_document_ids = np.flatnonzero(in_corpus)
    rng = np.random.default_rng(seed)
    others = np.delete(np.arange(NUM_ROWS), fixed_row)
    estimates = np.empty(trials)
    score_tensor = torch.tensor(scores)
    for t in range(trials):
        rows = np.concatenate([[fixed_row], rng.choice(others, size=BATCH_SIZE - 1, replace=False)])
        uniform = rng.choice(corpus_document_ids, size=NUM_UNIFORM, replace=False)
        cand = np.concatenate([row_document[rows], uniform])
        label = {
            "query_id": torch.arange(BATCH_SIZE),
            "cand_id": torch.tensor(cand),
            "cand_count": torch.tensor(counts[cand], dtype=torch.float32),
            "cand_in_corpus": torch.tensor(in_corpus[cand]),
            "cand_source": torch.cat(
                [torch.full((BATCH_SIZE,), SOURCE_IN_BATCH), torch.full((NUM_UNIFORM,), SOURCE_UNIFORM)]
            ),
            "num_rows": torch.tensor(NUM_ROWS),
            "corpus_size": torch.tensor(len(corpus_document_ids)),
        }
        row_scores = score_tensor[label["cand_id"]][None, :]
        if mismatched:
            # De-duplicate the pool but keep dividing by the expected count.
            log_q = candidate_log_q(label, "expected_count", per_source=False).double()
            keep = _first_occurrence(label["cand_id"], torch.ones_like(label["cand_id"], dtype=torch.bool))
            keep &= label["cand_id"] != label["cand_id"][0]
            estimates[t] = torch.exp(row_scores[0] - log_q)[keep].sum().item()
        else:
            log_z = negative_log_partition(
                row_scores,
                label,
                rows=torch.tensor([0]),
                q_measure=q_measure,
                mask_same_document=True,
                mask_same_query=False,
            )
            estimates[t] = log_z.exp().item()
    positive = row_document[fixed_row]
    reachable = (counts > 0) | in_corpus
    reachable[positive] = False
    truth = np.exp(scores[reachable]).sum()
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(trials), truth


def _rows_for_head_and_tail():
    row_document, counts, _, _ = build_world()
    head = int(np.flatnonzero(row_document == np.argmax(counts))[0])
    tail = int(np.flatnonzero(counts[row_document] == 1)[0])
    return head, tail


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo illustration, not a CI regression test")
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    for row in _rows_for_head_and_tail():
        for measure, mismatched in (("expected_count", False), ("inclusion", False), ("expected_count", True)):
            mean, stderr, truth = simulate(row, measure, args.trials, mismatched=mismatched)
            print(f"row={row} {measure} mismatched={mismatched}: mean={mean:.4f} stderr={stderr:.4f} truth={truth:.4f}")
