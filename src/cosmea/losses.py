"""Sentence Transformers losses that apply the sampling-bias correction.

Both losses expect the batch metadata of :class:`cosmea.collator.MixedNegativeDataCollator` as
``labels`` and are drop-in replacements for ``MultipleNegativesRankingLoss`` and
``CachedMultipleNegativesRankingLoss`` in the query-to-document direction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import Any

import torch
from sentence_transformers import util
from torch import Tensor, nn

from cosmea._compat import renamed_option
from cosmea.functional import (
    Combine,
    QMeasure,
    Variant,
    mixed_negative_logq_loss,
    resolve_combine,
    resolve_logq_strengths,
    resolve_mask_same_document,
)

from sentence_transformers.base.losses.gradcache import CachedLossMixin

_LABEL_KEYS = ("query_id", "cand_id", "cand_count", "cand_in_corpus", "cand_source", "num_rows", "corpus_size")


def _check_labels(labels: Any) -> dict[str, Tensor]:
    if not isinstance(labels, dict) or any(key not in labels for key in _LABEL_KEYS):
        raise ValueError(
            "This loss needs the sampling metadata produced by cosmea.MixedNegativeDataCollator. "
            "Pass it to the trainer: SentenceTransformerTrainer(..., data_collator=collator)."
        )
    return labels


class _LogQConfig:
    """Hyperparameters shared by the plain and the cached loss."""

    def _init_config(
        self,
        scale: float,
        similarity_fct: Callable[[Tensor, Tensor], Tensor],
        variant: Variant,
        logq_strength: float,
        q_measure: QMeasure,
        combine: Combine,
        mask_same_document: bool | None,
        mask_same_query: bool,
        *,
        in_batch_logq_strength: float | None,
        uniform_logq_strength: float | None,
    ) -> None:
        if not math.isfinite(scale) or not scale > 0:
            raise ValueError("scale must be positive.")
        if variant not in ("plugin", "yi", "improved"):
            raise ValueError(f"variant must be 'plugin', 'yi' or 'improved', got {variant!r}")
        if q_measure not in ("expected_count", "inclusion"):
            raise ValueError(f"q_measure must be 'expected_count' or 'inclusion', got {q_measure!r}")
        if combine not in ("auto", "pooled", "separate"):
            raise ValueError(f"combine must be 'auto', 'pooled' or 'separate', got {combine!r}")
        combine = resolve_combine(combine, variant)
        resolve_logq_strengths(logq_strength, in_batch_logq_strength, uniform_logq_strength, combine)

        self.scale = scale
        self.similarity_fct = similarity_fct
        self.variant = variant
        self.logq_strength = logq_strength
        self.in_batch_logq_strength = in_batch_logq_strength
        self.uniform_logq_strength = uniform_logq_strength
        self.q_measure = q_measure
        self.combine = combine
        self.mask_same_document = resolve_mask_same_document(mask_same_document, variant)
        self.mask_same_query = mask_same_query

        if variant == "yi" and q_measure == "inclusion":
            raise ValueError("yi + inclusion is unsupported; use expected_count or plugin.")

    @property
    def mask_same_item(self) -> bool:
        return self.mask_same_document

    @mask_same_item.setter
    def mask_same_item(self, value: bool) -> None:
        self.mask_same_document = value

    def _row_losses(self, anchors: Tensor, candidates: Tensor, labels: dict[str, Tensor], rows: Tensor) -> Tensor:
        num_candidates = labels["cand_id"].numel()
        if candidates.size(0) != num_candidates:
            raise ValueError(
                f"The batch has {candidates.size(0)} candidate embeddings but metadata for {num_candidates}. "
                "Losses that gather embeddings across devices are not supported."
            )
        # Temperature first, correction second (inside mixed_negative_logq_loss).
        scores = self.similarity_fct(anchors, candidates) * self.scale
        return mixed_negative_logq_loss(
            scores,
            labels,
            rows=rows,
            variant=self.variant,
            logq_strength=self.logq_strength,
            in_batch_logq_strength=self.in_batch_logq_strength,
            uniform_logq_strength=self.uniform_logq_strength,
            q_measure=self.q_measure,
            combine=self.combine,
            mask_same_document=self.mask_same_document,
            mask_same_query=self.mask_same_query,
        )

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": getattr(self.similarity_fct, "__name__", type(self.similarity_fct).__name__),
            "variant": self.variant,
            "logq_strength": self.logq_strength,
            "in_batch_logq_strength": self.in_batch_logq_strength,
            "uniform_logq_strength": self.uniform_logq_strength,
            "q_measure": self.q_measure,
            "combine": self.combine,
            "mask_same_document": self.mask_same_document,
            "mask_same_query": self.mask_same_query,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{khrylchenko2025correcting,
    title={Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval},
    author={Khrylchenko, Kirill and Baikalov, Vladimir and Makeev, Sergei and Matveev, Artem and Liamaev, Sergei},
    booktitle={Proceedings of the 19th ACM Conference on Recommender Systems},
    year={2025}
}
@inproceedings{yang2020mixed,
    title={Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations},
    author={Yang, Ji and Yi, Xinyang and Cheng, Derek Zhiyuan and Hong, Lichan and Li, Yang and Wang, Simon Xiaoming and Xu, Taibai and Chi, Ed H.},
    booktitle={Companion Proceedings of the Web Conference 2020},
    year={2020}
}
@inproceedings{yi2019sampling,
    title={Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations},
    author={Yi, Xinyang and Yang, Ji and Hong, Lichan and Cheng, Derek Zhiyuan and Heldt, Lukasz and Kumthekar, Aditee and Zhao, Zhe and Wei, Li and Chi, Ed},
    booktitle={Proceedings of the 13th ACM Conference on Recommender Systems},
    year={2019}
}
"""


class LogQMultipleNegativesRankingLoss(_LogQConfig, nn.Module):
    """InfoNCE over in-batch and uniformly sampled negatives with sampling-bias correction.

    Args:
        model: the SentenceTransformer.
        scale: inverse temperature applied to the similarity (20 = temperature 0.05).
        similarity_fct: pairwise similarity, default cosine.
        variant: `"yi"` (default, independent-draw approximation), `"plugin"` or `"improved"`
            (confidence-weighted objective with the same first gradient as `"plugin"`).
        logq_strength: default lambda in ``f - lambda * log q``; 0 disables correction.
        in_batch_logq_strength: Override for in-batch candidates in separate mode. None inherits `logq_strength`.
        uniform_logq_strength: Override for uniform candidates in separate mode. None inherits `logq_strength`.
        q_measure: ``"expected_count"`` (duplicates kept) or ``"inclusion"`` (pool de-duplicated).
        combine: ``"separate"`` or ``"pooled"`` softmax over the negative sources; ``"auto"``
            uses "separate" for "yi" and "pooled" otherwise.
        mask_same_document: mask accidental hits; ``None`` = keep them for "yi", mask otherwise.
        mask_same_query: mask the positives of other rows that share the row's query.
    """

    def __init__(
        self,
        model: nn.Module,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        variant: Variant = "yi",
        logq_strength: float = 1.0,
        q_measure: QMeasure = "expected_count",
        combine: Combine = "auto",
        mask_same_document: bool | None = None,
        mask_same_query: bool = True,
        *,
        in_batch_logq_strength: float | None = None,
        uniform_logq_strength: float | None = None,
        mask_same_item: bool | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        mask_same_document = renamed_option(mask_same_document, mask_same_item, "mask_same_document", "mask_same_item")
        self._init_config(
            scale,
            similarity_fct,
            variant,
            logq_strength,
            q_measure,
            combine,
            mask_same_document,
            mask_same_query,
            in_batch_logq_strength=in_batch_logq_strength,
            uniform_logq_strength=uniform_logq_strength,
        )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Any) -> Tensor:
        labels = _check_labels(labels)
        # One forward per column: the uniform-negative column has its own batch size, so the
        # columns cannot be merged into a single rectangular forward pass.
        embeddings = [self.model(features)["sentence_embedding"] for features in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: dict[str, Tensor]) -> Tensor:
        """Return mean row loss from anchor and candidate columns, each shaped (n, dim)."""
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embedding columns, got {len(embeddings)}")

        anchors = embeddings[0]
        candidates = torch.cat(embeddings[1:], dim=0)
        rows = torch.arange(anchors.size(0), device=anchors.device)
        return self._row_losses(anchors, candidates, labels, rows).mean()


class CachedLogQMultipleNegativesRankingLoss(_LogQConfig, CachedLossMixin, nn.Module):
    """GradCache version of `LogQMultipleNegativesRankingLoss` for large batches.

    The correction is a constant with respect to the embeddings, so it is compatible with
    gradient caching. It accepts the same loss options as `LogQMultipleNegativesRankingLoss`.

    Args:
        mini_batch_size: Positive number of query rows per chunk of the cached loss calculation.
        show_progress_bar: Show progress while replaying the chunks.
    """

    def __init__(
        self,
        model: nn.Module,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        variant: Variant = "yi",
        logq_strength: float = 1.0,
        q_measure: QMeasure = "expected_count",
        combine: Combine = "auto",
        mask_same_document: bool | None = None,
        mask_same_query: bool = True,
        show_progress_bar: bool = False,
        *,
        in_batch_logq_strength: float | None = None,
        uniform_logq_strength: float | None = None,
        mask_same_item: bool | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(mini_batch_size, int) or mini_batch_size < 1:
            raise ValueError("mini_batch_size must be a positive integer.")
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = None
        self.show_progress_bar = show_progress_bar
        mask_same_document = renamed_option(mask_same_document, mask_same_item, "mask_same_document", "mask_same_item")
        self._init_config(
            scale,
            similarity_fct,
            variant,
            logq_strength,
            q_measure,
            combine,
            mask_same_document,
            mask_same_query,
            in_batch_logq_strength=in_batch_logq_strength,
            uniform_logq_strength=uniform_logq_strength,
        )

    def calculate_loss(self, reps: list[list[Tensor]], labels: Any = None, *, with_backward: bool = False) -> Tensor:
        """Reduce anchor chunks over the full candidate pool to a scalar mean loss.

        With ``with_backward``, accumulate embedding gradients per chunk and return
        the detached loss; otherwise preserve its autograd graph for evaluation.
        """
        labels = _check_labels(labels)
        anchors = torch.cat(reps[0])
        candidates = torch.cat([torch.cat(column) for column in reps[1:]], dim=0)
        batch_size = anchors.size(0)
        losses: list[Tensor] = []

        for begin in range(0, batch_size, self.mini_batch_size):
            rows = torch.arange(begin, min(begin + self.mini_batch_size, batch_size), device=anchors.device)
            chunk = self._row_losses(anchors[rows], candidates, labels, rows).sum() / batch_size
            if with_backward:
                chunk.backward()
                chunk = chunk.detach()
            losses.append(chunk)

        return torch.stack(losses).sum()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Any) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        return self.forward_cached(sentence_features, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {**super().get_config_dict(), "mini_batch_size": self.mini_batch_size}
