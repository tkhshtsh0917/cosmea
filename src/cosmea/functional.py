"""Score-level objectives for sampling-aware retrieval.

``scores`` has shape (b, C), with B positive columns first. ``rows`` maps
these b query rows to the full batch. Corrections follow temperature scaling.

The partition estimators assume uniform sampling of B training rows without
replacement and K distinct corpus documents without replacement, independently.
With positive-document masking and no query-dependent filtering, each estimator
is unbiased over its proposal support (the pooled union or an individual source).
This does NOT make its log, normalized probabilities, or gradient unbiased.

Yi-style ranking NCE uses the empirical marginal proposal (an iid approximation
to the finite-data batch design). Its consistency theorem does not cover the
actual without-replacement collator, filtering, deduplication, or mined negatives.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from cosmea._compat import renamed_option
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


def resolve_logq_strengths(
    logq_strength: float,
    in_batch_logq_strength: float | None,
    uniform_logq_strength: float | None,
    combine: Combine,
) -> tuple[float, float]:
    """Resolve source overrides after resolving the combination mode."""
    for name, strength in (
        ("logq_strength", logq_strength),
        ("in_batch_logq_strength", in_batch_logq_strength),
        ("uniform_logq_strength", uniform_logq_strength),
    ):
        if strength is not None and (not math.isfinite(strength) or strength < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
    if combine != "separate" and (in_batch_logq_strength is not None or uniform_logq_strength is not None):
        raise ValueError('Source-specific logQ strengths require combine="separate".')
    return (
        logq_strength if in_batch_logq_strength is None else in_batch_logq_strength,
        logq_strength if uniform_logq_strength is None else uniform_logq_strength,
    )


def _uniform_probability(draws: int, corpus_size: int) -> float:
    if draws and not 0 < draws <= corpus_size:
        raise ValueError("Uniform sampling requires 0 < K <= corpus_size.")
    return draws / corpus_size if draws else 0.0


def resolve_mask_same_document(mask_same_document: bool | None, variant: str) -> bool:
    return variant != "yi" if mask_same_document is None else mask_same_document


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

    Args:
        label: Sampling metadata from the collator.
        q_measure: Use `expected_count` for log expected occurrences or `inclusion` for log probability of appearing.
        per_source: Use weights for each candidate's own source. False combines the in-batch and uniform proposals.
        rows: Original batch row indices. With indices, condition on the row by removing its positive occurrence
            from the remaining training rows, except in the iid approximation.
        iid_proposal: Use count/N and binomial inclusion for the ranking-NCE approximation.

    Returns:
        Shape `(b, C)` with `b` requested rows, or `(C,)` without `rows`. Mined candidates have log weight 0.
        Zero proposal support gives negative infinity.
    """
    source = label["cand_source"]
    batch_size = label["query_id"].numel()
    n, k = batch_size - 1, int((source == SOURCE_UNIFORM).sum())
    total, corpus = int(label["num_rows"]), int(label["corpus_size"])
    if total < batch_size or batch_size < 1:
        raise ValueError("Expected 1 <= batch size <= num_rows for random row sampling.")
    uniform_q = _uniform_probability(k, corpus)

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
    q_un = label["cand_in_corpus"].to(torch.float64) * uniform_q

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
    mask_same_document: bool,
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
    if not mask_same_document and not mask_same_query:
        return own

    if mask_same_query:
        same_query = query_id[rows, None] == query_id[None, :]
    else:
        same_query = own[:, :b]

    if mask_same_document:
        # Keep memory proportional to the chunk, not the full B-by-C batch.
        mask = cand_id[rows, None] == cand_id[None, :]
        if mask_same_query:
            repeated = (same_query.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
            for local_row in repeated.tolist():
                positive_ids = cand_id[:b][same_query[local_row]]
                mask[local_row] |= torch.isin(cand_id, positive_ids)
        return mask | own

    mask = torch.zeros_like(own)
    mask[:, :b] = same_query
    return mask | own


def _loss_from_parts(pos: Tensor, log_z_neg: Tensor, pos_log_q: Tensor, variant: Variant) -> Tensor:
    has_negatives = log_z_neg != -torch.inf
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
    mask_same_document: bool | None = None,
    mask_same_query: bool = True,
    iid_proposal: bool = False,
    mask_same_item: bool | None = None,
) -> Tensor:
    """Return the corrected log sum of eligible negatives for each requested row.

    Args:
        scores: Temperature-scaled scores of shape `(b, C)` with `C` candidates.
        label: Candidate and row metadata from the collator.
        rows: Original row indices for the first axis of `scores`. Defaults to indices starting at zero.
        sources: Restrict candidates to the named source codes in `cand_source`. None pools all sources.
        q_measure: In inclusion mode, keep one occurrence per eligible document after masking.
        iid_proposal: Use the approximation from the Yi-style objective.

    Returns:
        Shape `(b,)`. Rows without eligible negatives return negative infinity.

    Raises:
        ValueError: Scores contain a NaN or infinity, or correction requires a proposal with zero support.
    """
    if rows is None:
        rows = torch.arange(scores.size(0), device=scores.device)
    mask_same_document = renamed_option(mask_same_document, mask_same_item, "mask_same_document", "mask_same_item")
    log_z, _ = _negative_log_partition(
        scores,
        label,
        rows=rows,
        logq_strength=logq_strength,
        q_measure=q_measure,
        sources=sources,
        mask_same_document=True if mask_same_document is None else mask_same_document,
        mask_same_query=mask_same_query,
        iid_proposal=iid_proposal,
    )
    return log_z


def _negative_log_partition(
    scores: Tensor,
    label: dict[str, Tensor],
    *,
    rows: Tensor,
    logq_strength: float,
    q_measure: QMeasure,
    sources: tuple[int, ...] | None,
    mask_same_document: bool,
    mask_same_query: bool,
    iid_proposal: bool,
) -> tuple[Tensor, Tensor]:
    """Return the log sum and the exact mask used to compute it."""
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite; NaN/Inf indicates invalid embeddings or numerical overflow.")

    # AMP-safe loss arithmetic, while retaining float64 oracle inputs.
    scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    source = label["cand_source"]
    subset = (
        torch.ones_like(source, dtype=torch.bool) if sources is None else torch.zeros_like(source, dtype=torch.bool)
    )
    for s in sources or ():
        subset |= source == s

    valid = subset[None, :] & ~known_positive_mask(label, rows, mask_same_document, mask_same_query)
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
        if q_measure not in ("expected_count", "inclusion"):
            raise ValueError(f"Unknown q_measure {q_measure!r}")
        if sources == (SOURCE_UNIFORM,):
            # Distinct corpus draws have the same expected count and inclusion probability.
            # This proposal depends only on the corpus, regardless of the row sampler.
            draws = int((source == SOURCE_UNIFORM).sum())
            uniform_q = _uniform_probability(draws, int(label["corpus_size"]))
            log_q = (label["cand_in_corpus"].to(torch.float64) * uniform_q).log()
        else:
            log_q = candidate_log_q(label, q_measure, sources is not None, rows=rows, iid_proposal=iid_proposal)
        log_q = log_q.to(scores.dtype)
        if ((~torch.isfinite(log_q)) & valid).any():
            raise ValueError("A negative has zero proposal support; check frequency, corpus IDs and sampling design.")
        adjusted = scores - logq_strength * torch.where(valid, log_q, torch.zeros_like(log_q))

    masked = adjusted.masked_fill(~valid, -torch.inf)
    active = valid.any(dim=1)

    # Avoid all--inf logsumexp in the autograd graph, including under AMP.
    safe = torch.where(active[:, None], masked, torch.zeros_like(masked))
    return torch.where(active, torch.logsumexp(safe, dim=1), torch.full_like(scores[:, 0], -torch.inf)), valid


def mixed_negative_logq_loss(
    scores: Tensor,
    label: dict[str, Tensor],
    *,
    rows: Tensor | None = None,
    variant: Variant = "yi",
    logq_strength: float = 1.0,
    in_batch_logq_strength: float | None = None,
    uniform_logq_strength: float | None = None,
    q_measure: QMeasure = "expected_count",
    combine: Combine = "auto",
    mask_same_document: bool | None = None,
    mask_same_query: bool = True,
    mask_same_item: bool | None = None,
) -> Tensor:
    """Return per-query losses for scores that already include the temperature scale.

    - `yi` corrects positive and negative scores using an empirical iid proposal approximation.
    - `plugin` corrects negatives by expected counts or inclusion probabilities and keeps the positive unchanged.
    - `improved` uses a stop-gradient confidence weight. It has the same first derivative as `plugin`
      and different scalar values and higher-order derivatives.
    - `separate` averages active in-batch, uniform and uncorrected extra objectives. Each sampled source uses
      `logq_strength` unless overridden. A zero override skips that proposal calculation. The uniform
      correction cancels in separate `yi`, which still checks its support.
    - `pooled` computes one softmax. Expected counts combine via multiple importance sampling. A `yi` row
      with only extra negatives uses an uncorrected softmax.

    Args:
        scores: A floating matrix of shape `(b, C)` with `C` candidates.
        label: Candidate and row metadata from the collator. The initial candidate columns are the
            positives in original batch order.
        rows: Original batch row indices for the first axis of `scores`. Defaults to indices starting at zero.

    Returns:
        Shape `(b,)`, one loss per requested row.

    Raises:
        ValueError: Scores or row indices do not match the candidate metadata, or the correction is invalid.
    """
    if variant not in ("yi", "plugin", "improved"):
        raise ValueError(f"Unknown variant {variant!r}")
    if q_measure not in ("expected_count", "inclusion"):
        raise ValueError(f"Unknown q_measure {q_measure!r}")
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
    in_batch_strength, uniform_strength = resolve_logq_strengths(
        logq_strength, in_batch_logq_strength, uniform_logq_strength, combine
    )
    mask_same_document = renamed_option(mask_same_document, mask_same_item, "mask_same_document", "mask_same_item")
    mask_same_document = resolve_mask_same_document(mask_same_document, variant)
    if variant == "yi" and q_measure == "inclusion":
        raise ValueError("yi + inclusion has no supported ranking-NCE interpretation; use expected_count or plugin.")
    pos = scores[torch.arange(rows.numel(), device=scores.device), rows]
    k = int((label["cand_source"] == SOURCE_UNIFORM).sum())

    def partition(sources: tuple[int, ...] | None, strength: float) -> tuple[Tensor, Tensor]:
        return _negative_log_partition(
            scores,
            label,
            sources=sources,
            rows=rows,
            logq_strength=strength,
            q_measure=q_measure,
            mask_same_document=mask_same_document,
            mask_same_query=mask_same_query,
            iid_proposal=variant == "yi",
        )

    def positive_log_q(strength: float, *, per_source: bool) -> Tensor:
        if variant != "yi" or strength == 0:
            return torch.zeros_like(pos)
        return strength * candidate_log_q(label, q_measure, per_source, iid_proposal=True)[rows].to(scores.dtype)

    if combine == "pooled":
        log_z, valid = partition(None, logq_strength)
        pos_q = torch.zeros_like(pos)
        if variant == "yi" and logq_strength:
            has_sampled = (valid & (label["cand_source"] != SOURCE_OTHER)).any(dim=1)
            if has_sampled.any():
                pos_q = torch.where(has_sampled, positive_log_q(logq_strength, per_source=False), pos_q)
                if (~torch.isfinite(pos_q) & has_sampled).any():
                    raise ValueError("Positive has zero proposal support.")
        return _loss_from_parts(pos, log_z, pos_q, variant)

    groups = [(SOURCE_IN_BATCH, in_batch_strength)]
    if k:
        groups.append((SOURCE_UNIFORM, uniform_strength))
    if (label["cand_source"] == SOURCE_OTHER).any():
        groups.append((SOURCE_OTHER, 0.0))

    losses, active = [], []
    for source, strength in groups:
        if source == SOURCE_UNIFORM and variant == "yi":
            log_z, valid = partition((source,), 0.0)
            pos_q = torch.zeros_like(pos)
            if strength:
                in_corpus = label["cand_in_corpus"]
                _uniform_probability(k, int(label["corpus_size"]))
                if not in_corpus[rows].all():
                    raise ValueError("yi + separate requires every positive ID in the uniform corpus.")
                if (valid & ~in_corpus).any():
                    raise ValueError("A uniform negative has zero proposal support; check corpus IDs.")
        else:
            log_z, _ = partition((source,), strength)
            pos_q = positive_log_q(strength, per_source=True)
        if variant == "yi" and (~torch.isfinite(pos_q) & torch.isfinite(log_z)).any():
            raise ValueError("Positive has zero proposal support.")
        losses.append(_loss_from_parts(pos, log_z, pos_q, variant))
        active.append(log_z != -torch.inf)

    return torch.stack(losses).sum(0) / torch.stack(active).sum(0).clamp_min(1)
