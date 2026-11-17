"""Monte Carlo check that Z_hat_neg is an unbiased estimate of the negative partition function.

The simulation reproduces the actual batch construction: one fixed row plus B-1 other rows drawn
without replacement from the training set, plus K corpus items drawn uniformly without
replacement. For the fixed row with query u and positive p the target is

    Z_neg = sum_{d != p} exp(f(u, d))      over every item of the training set or the corpus.
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

NUM_ROWS, CORPUS_SIZE, NUM_TRAIN_ITEMS = 4000, 300, 210  # train items 200..209 are not in the corpus
BATCH_SIZE, NUM_UNIFORM = 48, 16


def build_world(seed=0):
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, NUM_TRAIN_ITEMS + 1) ** 1.1  # Zipf: head item holds ~19% of rows
    popularity /= popularity.sum()
    row_item = rng.choice(NUM_TRAIN_ITEMS, size=NUM_ROWS, p=popularity)
    num_items = max(CORPUS_SIZE, NUM_TRAIN_ITEMS) + 10
    in_corpus = np.zeros(num_items, dtype=bool)
    in_corpus[:200] = True  # train items 0..199
    in_corpus[220:320] = True  # 100 items that never occur in training
    counts = np.bincount(row_item, minlength=num_items)
    scores = rng.normal(0.0, 1.5, size=num_items)
    return row_item, counts, in_corpus, scores


def simulate(fixed_row, q_measure, trials, mismatched=False, seed=1):
    row_item, counts, in_corpus, scores = build_world()
    corpus_items = np.flatnonzero(in_corpus)
    rng = np.random.default_rng(seed)
    others = np.delete(np.arange(NUM_ROWS), fixed_row)
    estimates = np.empty(trials)
    score_tensor = torch.tensor(scores)
    for t in range(trials):
        rows = np.concatenate([[fixed_row], rng.choice(others, size=BATCH_SIZE - 1, replace=False)])
        uniform = rng.choice(corpus_items, size=NUM_UNIFORM, replace=False)
        cand = np.concatenate([row_item[rows], uniform])
        label = {
            "query_id": torch.arange(BATCH_SIZE),
            "cand_id": torch.tensor(cand),
            "cand_count": torch.tensor(counts[cand], dtype=torch.float32),
            "cand_in_corpus": torch.tensor(in_corpus[cand]),
            "cand_source": torch.cat(
                [torch.full((BATCH_SIZE,), SOURCE_IN_BATCH), torch.full((NUM_UNIFORM,), SOURCE_UNIFORM)]
            ),
            "num_rows": torch.tensor(NUM_ROWS),
            "corpus_size": torch.tensor(len(corpus_items)),
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
                mask_same_item=True,
                mask_same_query=False,
            )
            estimates[t] = log_z.exp().item()
    positive = row_item[fixed_row]
    reachable = (counts > 0) | in_corpus
    reachable[positive] = False
    truth = np.exp(scores[reachable]).sum()
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(trials), truth


def _rows_for_head_and_tail():
    row_item, counts, _, _ = build_world()
    head = int(np.flatnonzero(row_item == np.argmax(counts))[0])
    tail = int(np.flatnonzero(counts[row_item] == 1)[0])
    return head, tail


def test_expected_count_estimator_is_unbiased():
    for row in _rows_for_head_and_tail():
        mean, stderr, truth = simulate(row, "expected_count", trials=6000)

        assert abs(mean - truth) < 4 * stderr, (mean, stderr, truth)
        assert abs(mean / truth - 1) < 0.03


def test_inclusion_estimator_is_unbiased():
    for row in _rows_for_head_and_tail():
        mean, stderr, truth = simulate(row, "inclusion", trials=6000)
        # Hypergeometric inclusion is exact for the finite-population sampling design.
        assert abs(mean / truth - 1) < 0.03, (mean, stderr, truth)


def test_dedup_with_expected_count_is_biased():
    """De-duplicating while dividing by the expected count under-weights head items."""
    _, tail = _rows_for_head_and_tail()
    mean, stderr, truth = simulate(tail, "expected_count", trials=6000, mismatched=True)

    assert mean < truth - 6 * stderr, (mean, stderr, truth)


if __name__ == "__main__":
    head, tail = _rows_for_head_and_tail()
    for name, row in (("head positive", head), ("tail positive", tail)):
        for q_measure in ("expected_count", "inclusion"):
            mean, stderr, truth = simulate(row, q_measure, trials=20000)
            print(
                f"{name:14s} {q_measure:15s} E[Z_hat]={mean:9.3f} +- {stderr:.3f}  Z_neg={truth:9.3f}  ratio={mean / truth:.4f}"
            )
        mean, stderr, truth = simulate(row, "expected_count", trials=20000, mismatched=True)
        print(
            f"{name:14s} {'dedup+exp.count':15s} E[Z_hat]={mean:9.3f} +- {stderr:.3f}  Z_neg={truth:9.3f}  ratio={mean / truth:.4f}"
        )
