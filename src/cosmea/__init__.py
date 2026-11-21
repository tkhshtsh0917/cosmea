"""Sampling-bias-corrected contrastive training for Sentence Transformers.

* :class:`MixedNegativeDataCollator` - Mixed Negative Sampling + logQ metadata (collate_fn).
* :class:`LogQMultipleNegativesRankingLoss` / :class:`CachedLogQMultipleNegativesRankingLoss` -
  InfoNCE with logQ correction over in-batch and uniform negatives.
* :class:`DocumentFrequency` - exact document counts of the training positives.
"""

from cosmea.collator import MixedNegativeDataCollator
from cosmea.frequency import DocumentFrequency, ItemFrequency, stable_id
from cosmea.functional import (
    candidate_log_q,
    mixed_negative_logq_loss,
    negative_log_partition,
)
from cosmea.losses import CachedLogQMultipleNegativesRankingLoss, LogQMultipleNegativesRankingLoss

__all__ = [
    "CachedLogQMultipleNegativesRankingLoss",
    "DocumentFrequency",
    "ItemFrequency",
    "LogQMultipleNegativesRankingLoss",
    "MixedNegativeDataCollator",
    "candidate_log_q",
    "mixed_negative_logq_loss",
    "negative_log_partition",
    "stable_id",
]
__version__ = "0.1.0"
