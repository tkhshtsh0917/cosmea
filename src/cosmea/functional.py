"""Score-level objectives for sampling-aware retrieval.

``scores`` has shape (b, C), with B positive columns first. ``rows`` maps
these b query rows to the full batch. Corrections follow temperature scaling.

The partition estimators assume uniform sampling of B training rows without
replacement and K distinct corpus items without replacement, independently.
Only the pooled, positive-masked estimator with full proposal support and
no query-dependent filtering is an unbiased estimate of the negative partition.
This does NOT make its log, normalized probabilities, or gradient unbiased.

Yi-style ranking NCE uses the empirical marginal proposal (an iid approximation
to the finite-data batch design). Its consistency theorem does not cover the
actual without-replacement collator, filtering, deduplication, or mined negatives.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

Variant = Literal["yi", "plugin", "improved"]
QMeasure = Literal["expected_count", "inclusion"]
Combine = Literal["auto", "pooled", "separate"]
SOURCE_IN_BATCH = 0
SOURCE_UNIFORM = 1
SOURCE_OTHER = 2


def resolve_combine(combine: Combine, variant: Variant) -> Combine:
    if combine == "auto":
        return "separate" if variant == "yi" else "pooled"
    return combine


def resolve_mask_same_item(mask_same_item: bool | None, variant: str) -> bool:
    return variant != "yi" if mask_same_item is None else mask_same_item


def _first_occurrence(ids: Tensor, subset: Tensor) -> Tensor:
    """Keep the first *eligible* occurrence of each ID in a one-dimensional subset."""
    keep = torch.zeros_like(subset)
    index = subset.nonzero(as_tuple=True)[0]
    if index.numel():
        sorted_ids, order = torch.sort(ids[index], stable=True)
        first = torch.ones_like(sorted_ids, dtype=torch.bool)
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
        keep[index[order[first]]] = True
    return keep


def _inclusion_probability(count: Tensor, population: int, draws: int) -> Tensor:
    """Exact hypergeometric inclusion: 1 - choose(L-c,n)/choose(L,n).

    Product of miss probabilities avoids cancellation of large log-gammas.
    No tensor of shape (B, C, B) is allocated. Runtime is O(draws * count.numel()).
    """
    log_miss = torch.zeros_like(count)
    for t in range(draws):
        hit = (count / (population - t)).clamp(0, 1)
        log_miss = log_miss + torch.log1p(-hit)
    return -torch.expm1(log_miss)


def candidate_log_q(
    label: dict[str, Tensor],
    q_measure: QMeasure,
    per_source: bool,
    *,
    rows: Tensor | None = None,
    iid_proposal: bool = False,
) -> Tensor:
    """Return log expected multiplicity or log inclusion probability.

    With ``rows``: shape (b, C), conditioned on each row being present; its
    positive occurrence is removed from the N-1 available training rows.
    Without ``rows``: shape (C,), suitable for items distinct from the positive.
    ``iid_proposal=True`` uses count/N and binomial inclusion instead, for the
    ranking-NCE approximation. SOURCE_OTHER has log q=0 (uncorrected heuristic).
    Zero support is represented by -inf, never an invented epsilon probability.
    """
    source = label["cand_source"]
    batch_size = label["query_id"].numel()
    n, k = batch_size - 1, int((source == SOURCE_UNIFORM).sum())
    total, corpus = int(label["num_rows"]), int(label["corpus_size"])
    if total < batch_size or batch_size < 1:
        raise ValueError("Expected 1 <= batch size <= num_rows for random row sampling.")
    if k and not 0 < k <= corpus:
        raise ValueError("Uniform sampling requires 0 < K <= corpus_size.")

    # float64 preserves finite-population probabilities even for large row counts.
    count = label["cand_count"].detach().to(torch.float64)
    if not torch.isfinite(count).all() or (count < 0).any() or (count > total).any() or (count != count.floor()).any():
        raise ValueError("cand_count must contain integer frequencies in [0, num_rows].")

    if rows is not None:
        count = count[None, :].expand(rows.numel(), -1)
        if not iid_proposal:
            own = label["cand_id"][rows, None] == label["cand_id"][None, :]
            count = (count - own.to(count.dtype)).clamp_min(0)

    population = total if iid_proposal else total - 1
    count = count.clamp(max=max(population, 0))
    q_un = label["cand_in_corpus"].to(torch.float64) * (k / corpus if k else 0.0)

    if q_measure == "expected_count":
        q_in = n * count / max(population, 1)
        q_pool = q_in + q_un
    elif q_measure == "inclusion":
        if iid_proposal:
            q_in = -torch.expm1(n * torch.log1p(-(count / total).clamp(max=1))) if n else count * 0
        else:
            q_in = _inclusion_probability(count, population, n)
        q_pool = q_in + (1 - q_in) * q_un
    else:
        raise ValueError(f"Unknown q_measure {q_measure!r}")

    q = torch.where(source == SOURCE_UNIFORM, q_un, q_in) if per_source else q_pool
    # Avoid log(0) for deterministic extra columns; they receive no correction.
    return torch.where(source == SOURCE_OTHER, torch.ones_like(q), q).log()


def known_positive_mask(
    label: dict[str, Tensor],
    rows: Tensor,
    mask_same_item: bool,
    mask_same_query: bool,
) -> Tensor:
    """Mask the own column and optional known-positive matches, shape (b, C).

    Query matching only knows positives present in this batch. It is a heuristic,
    not a complete relevance set or a correction for exposure propensities.
    """
    query_id, cand_id = label["query_id"], label["cand_id"]
    b, c = query_id.numel(), cand_id.numel()
    own = torch.zeros(rows.numel(), c, dtype=torch.bool, device=cand_id.device)
    own[torch.arange(rows.numel(), device=cand_id.device), rows] = True
    if not mask_same_item and not mask_same_query:
        return own

    if mask_same_query:
        same_query = query_id[rows, None] == query_id[None, :]
    else:
        same_query = own[:, :b]

    if mask_same_item:
        equal = cand_id[:b, None] == cand_id[None, :]
        return ((same_query.float() @ equal.float()) > 0) | own

    mask = torch.zeros_like(own)
    mask[:, :b] = same_query
    return mask | own


def _loss_from_parts(pos: Tensor, log_z_neg: Tensor, pos_log_q: Tensor, variant: Variant) -> Tensor:
    has_negatives = torch.isfinite(log_z_neg)
    safe_log_z = torch.where(has_negatives, log_z_neg, torch.zeros_like(log_z_neg))

    if variant == "plugin":
        loss = torch.nn.functional.softplus(safe_log_z - pos)
    elif variant == "yi":
        # For an inactive source, q(pos)=0 is irrelevant; avoid inf arithmetic.
        pos_log_q = torch.where(has_negatives, pos_log_q, torch.zeros_like(pos_log_q))
        loss = torch.nn.functional.softplus(safe_log_z - pos + pos_log_q)
    elif variant == "improved":
        # Confidence-weighted objective; identical *first*
        # derivatives to plugin when both reuse this same partition estimate.
        weight = torch.sigmoid(safe_log_z - pos).detach()
        loss = weight * (safe_log_z - pos)
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    return torch.where(has_negatives, loss, torch.zeros_like(loss))


def negative_log_partition(
    scores: Tensor,
    label: dict[str, Tensor],
    *,
    rows: Tensor | None = None,
    logq_strength: float = 1.0,
    q_measure: QMeasure = "expected_count",
    sources: tuple[int, ...] | None = None,
    mask_same_item: bool = True,
    mask_same_query: bool = True,
    iid_proposal: bool = False,
) -> Tensor:
    """Log corrected negative sum per row; -inf when a row has no negatives.

    Duplicates are removed *after* row-specific masking in inclusion mode.
    iid_proposal is reserved for the approximate Yi-style objective.
    """
    if rows is None:
        rows = torch.arange(scores.size(0), device=scores.device)

    # AMP-safe loss arithmetic, while retaining float64 oracle inputs.
    scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    source = label["cand_source"]
    subset = (
        torch.ones_like(source, dtype=torch.bool) if sources is None else torch.zeros_like(source, dtype=torch.bool)
    )
    for s in sources or ():
        subset |= source == s

    valid = subset[None, :] & ~known_positive_mask(label, rows, mask_same_item, mask_same_query)
    if q_measure == "inclusion":
        valid = torch.stack(
            [
                _first_occurrence(label["cand_id"], row & (source != SOURCE_OTHER)) | (row & (source == SOURCE_OTHER))
                for row in valid
            ]
        )

    if logq_strength == 0:
        adjusted = scores
    else:
        log_q = candidate_log_q(label, q_measure, sources is not None, rows=rows, iid_proposal=iid_proposal).to(
            scores.dtype
        )
        if ((~torch.isfinite(log_q)) & valid).any():
            raise ValueError("A negative has zero proposal support; check frequency, corpus IDs and sampling design.")
        adjusted = scores - logq_strength * torch.where(valid, log_q, torch.zeros_like(log_q))

    masked = adjusted.masked_fill(~valid, -torch.inf)
    active = valid.any(dim=1)

    # Avoid all--inf logsumexp in the autograd graph, including under AMP.
    safe = torch.where(active[:, None], masked, torch.zeros_like(masked))
    return torch.where(active, torch.logsumexp(safe, dim=1), torch.full_like(scores[:, 0], -torch.inf))


def mixed_negative_logq_loss(
    scores: Tensor,
    label: dict[str, Tensor],
    *,
    rows: Tensor | None = None,
    variant: Variant = "yi",
    logq_strength: float = 1.0,
    q_measure: QMeasure = "expected_count",
    combine: Combine = "auto",
    mask_same_item: bool | None = None,
    mask_same_query: bool = True,
) -> Tensor:
    """Return (b,) losses for already temperature-scaled scores of shape (b,C).

    yi: all logits corrected, empirical iid proposal approximation.
    plugin: positive uncorrected, negatives corrected by expected counts/inclusion.
    improved: stop-gradient confidence-weighted adaptation, same first derivative
        as plugin, different scalar values and higher-order derivatives.
    separate: average only sources with at least one valid negative for each row.
    pooled: one softmax; expected counts combine via multiple importance sampling.
    No configuration is a general causal popularity/selection debiasing method.
    """
    if variant not in ("yi", "plugin", "improved"):
        raise ValueError(f"Unknown variant {variant!r}")
    if q_measure not in ("expected_count", "inclusion"):
        raise ValueError(f"Unknown q_measure {q_measure!r}")
    if not math.isfinite(logq_strength) or logq_strength < 0:
        raise ValueError("logq_strength must be finite and non-negative.")
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating (b, C) matrix.")

    b, c = label["query_id"].numel(), label["cand_id"].numel()
    if rows is None:
        rows = torch.arange(scores.size(0), device=scores.device)
    if scores.shape != (rows.numel(), c) or rows.numel() == 0 or ((rows < 0) | (rows >= b)).any():
        raise ValueError("scores, rows and candidate metadata have incompatible shapes.")

    scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    combine = resolve_combine(combine, variant)
    if combine not in ("separate", "pooled"):
        raise ValueError(f"Unknown combine {combine!r}")
    mask_same_item = resolve_mask_same_item(mask_same_item, variant)
    if variant == "yi" and q_measure == "inclusion":
        raise ValueError("yi + inclusion has no supported ranking-NCE interpretation; use expected_count or plugin.")
    pos = scores[torch.arange(rows.numel(), device=scores.device), rows]

    def partition(sources: tuple[int, ...] | None) -> Tensor:
        return negative_log_partition(
            scores,
            label,
            sources=sources,
            rows=rows,
            logq_strength=logq_strength,
            q_measure=q_measure,
            mask_same_item=mask_same_item,
            mask_same_query=mask_same_query,
            iid_proposal=variant == "yi",
        )

    def positive_log_q(per_source: bool) -> Tensor:
        if variant != "yi" or logq_strength == 0:
            return torch.zeros_like(pos)
        return logq_strength * candidate_log_q(label, q_measure, per_source, iid_proposal=True)[rows].to(scores.dtype)

    if combine == "pooled":
        log_z = partition(None)
        pos_q = positive_log_q(False)
        if variant == "yi" and (~torch.isfinite(pos_q) & torch.isfinite(log_z)).any():
            raise ValueError("Positive has zero proposal support.")
        return _loss_from_parts(pos, log_z, pos_q, variant)

    groups: list[tuple[int, ...]] = [(SOURCE_IN_BATCH, SOURCE_OTHER)]
    k = int((label["cand_source"] == SOURCE_UNIFORM).sum())
    if k:
        groups.append((SOURCE_UNIFORM,))

    losses, active = [], []
    for group in groups:
        log_z = partition(group)
        if group == (SOURCE_UNIFORM,) and variant == "yi" and logq_strength:
            if not label["cand_in_corpus"][rows].all():
                raise ValueError("yi + separate requires every positive ID in the uniform corpus.")
            pos_q = pos.new_full(pos.shape, logq_strength * math.log(k / int(label["corpus_size"])))
        else:
            pos_q = positive_log_q(True)
        if variant == "yi" and (~torch.isfinite(pos_q) & torch.isfinite(log_z)).any():
            raise ValueError("Positive has zero proposal support.")
        losses.append(_loss_from_parts(pos, log_z, pos_q, variant))
        active.append(torch.isfinite(log_z))

    return torch.stack(losses).sum(0) / torch.stack(active).sum(0).clamp_min(1)
