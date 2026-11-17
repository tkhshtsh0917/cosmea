"""Data collator that adds Mixed Negative Sampling and logQ metadata to a training batch.

In Sentence Transformers the training DataLoader is built by the trainer; the collator is the
``collate_fn`` of that DataLoader and the one place that sees the raw texts of a whole batch.
This collator therefore does the two things the loss cannot do by itself:

1. **Mixed Negative Sampling**: draw ``K`` items uniformly from the
   serving corpus and append them as an extra candidate column that is shared by every query in
   the batch. This is what lets items that never occur as a positive act as negatives.
2. **logQ metadata**: attach, for every candidate, its item id, its training-set frequency and
   whether it is in the corpus, so the loss can compute the correct proposal probability.

The metadata travels in ``batch["label"]`` as a dict of tensors, the only channel the trainer
forwards to the loss untouched.
"""

from __future__ import annotations

import logging
import numbers
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator

from cosmea.frequency import ItemFrequency, stable_id
from cosmea.functional import SOURCE_IN_BATCH, SOURCE_OTHER, SOURCE_UNIFORM

logger = logging.getLogger(__name__)


@dataclass
class MixedNegativeDataCollator(SentenceTransformerDataCollator):
    """Collator for (anchor, positive[, extra columns]) datasets with uniform corpus negatives.

    Args:
        preprocess_fn: ``model.preprocess`` (as for the stock collator).
        item_frequency: counts of the training positives, ``ItemFrequency.from_dataset(...)``.
            Must be built from the *training* split with the same id convention.
        corpus: texts of the serving corpus to sample uniform negatives from. Should cover
            everything that can be retrieved at serving time, including items without feedback.
        corpus_ids: optional item ids aligned with ``corpus``; defaults to the hash of each text.
        num_uniform: number K of uniform negatives added to every batch (0 = logQ only).
        query_id_column / item_id_column: optional dataset columns holding ids for the anchor and
            the positive. They are removed before tokenization. Without them, ids are text hashes,
            so two different texts of one item are treated as different items.
        uniform_column: name of the appended candidate column.
        extra_item_id_columns: extra text column -> ID column mapping; required for each
            extra column when ``item_id_column`` is set.
        prompts / router_mapping / max_length: inherited preprocessing settings.
            Pass them here, not only to TrainingArguments: a custom collator does
            not inherit the trainer's settings. Uniforms reuse the positive settings.

    The first two text columns of the dataset are the anchor and the positive, as for
    ``MultipleNegativesRankingLoss``. Further text columns (e.g. mined hard negatives) are kept as
    candidates but are not treated as samples from a known distribution: they get no correction.

    The item counts assume batches are a uniform random sample of the training rows (the default
    ``BatchSamplers.BATCH_SAMPLER``). With ``NO_DUPLICATES`` or grouped samplers the in-batch
    proposal is no longer frequency-proportional and the correction becomes approximate.
    """

    item_frequency: ItemFrequency | None = field(default=None, repr=False)
    corpus: Sequence[str] | None = field(default=None, repr=False)
    corpus_ids: Sequence[Any] | None = field(default=None, repr=False)
    num_uniform: int = 0
    query_id_column: str | None = None
    item_id_column: str | None = None
    extra_item_id_columns: dict[str, str] = field(default_factory=dict)
    uniform_column: str = "uniform_negative"

    _corpus_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _corpus_unique_ids: np.ndarray | None = field(default=None, init=False, repr=False)
    _corpus_id_set: frozenset[int] = field(default_factory=frozenset, init=False, repr=False)
    _rng: random.Random | None = field(default=None, init=False, repr=False)
    _rng_pid: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.item_frequency is None:
            raise ValueError("item_frequency is required: build it with ItemFrequency.from_dataset(train_dataset).")
        if not isinstance(self.num_uniform, int) or self.num_uniform < 0:
            raise ValueError("num_uniform must be >= 0.")
        if self.num_uniform > 0 and self.corpus is None:
            raise ValueError("num_uniform > 0 requires a corpus to sample from.")

        if self.corpus is not None:
            if self.corpus_ids is not None:
                if len(self.corpus_ids) != len(self.corpus):
                    raise ValueError("corpus_ids must be aligned with corpus.")
                ids = np.fromiter((stable_id(i) for i in self.corpus_ids), dtype=np.int64, count=len(self.corpus))
            else:
                ids = np.fromiter((stable_id(t) for t in self.corpus), dtype=np.int64, count=len(self.corpus))

            # Sample over distinct items: a duplicated corpus entry would otherwise be drawn twice
            # as often as the "uniform" proposal claims.
            unique_ids, first_positions = np.unique(ids, return_index=True)
            order = np.argsort(first_positions)
            self._corpus_positions = first_positions[order]
            self._corpus_unique_ids = unique_ids[order]
            self._corpus_id_set = frozenset(int(i) for i in unique_ids)
            if len(unique_ids) < len(ids):
                logger.warning(
                    "Corpus has %d entries but %d distinct item ids; sampling over distinct items.",
                    len(ids),
                    len(unique_ids),
                )
            if self.num_uniform > len(unique_ids):
                raise ValueError(f"num_uniform={self.num_uniform} exceeds the corpus size {len(unique_ids)}.")

    @property
    def corpus_size(self) -> int:
        """Number of distinct corpus IDs, not the number of input text rows."""
        return 0 if self._corpus_unique_ids is None else len(self._corpus_unique_ids)

    def _process_rng(self) -> random.Random:
        # One generator per process, seeded from torch's RNG. DataLoader seeds torch differently in
        # every worker, so workers do not replay each other's uniform negatives, while a fixed
        # trainer seed keeps runs reproducible.
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng = random.Random(int(torch.randint(0, 2**62, (1,)).item()))
            self._rng_pid = pid
        return self._rng

    def _ids(self, values: list[Any]) -> list[int]:
        if values and not isinstance(values[0], (str, bytes, numbers.Integral)):
            raise TypeError(
                "Cannot derive ids from non-text inputs; provide query_id_column / item_id_column in the dataset."
            )
        return [stable_id(v) for v in values]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Preprocess a batch, advance its process RNG, and attach candidate metadata.

        Empty input returns an empty dict. Inconsistent columns or unknown positive IDs
        are rejected; every non-ID column must belong to the text input contract.
        """
        if not features:
            return {}

        frequency = self.item_frequency
        if frequency is None:
            raise ValueError("item_frequency is required.")

        id_columns = {c for c in (self.query_id_column, self.item_id_column) if c}
        id_columns.update(self.extra_item_id_columns.values())
        if any(set(row) != set(features[0]) for row in features):
            raise ValueError("All rows must have the same columns.")
        missing = id_columns - set(features[0])
        if missing:
            raise KeyError(f"Id column(s) {sorted(missing)} not found in the dataset columns {list(features[0])}.")
        rows = [{k: v for k, v in row.items() if k not in id_columns} for row in features]

        text_columns = [c for c in rows[0] if c != "dataset_name" and c not in self.valid_label_columns]
        if len(text_columns) < 2:
            raise ValueError(f"Expected at least an anchor and a positive column, got {text_columns}.")
        if any(c in rows[0] for c in self.valid_label_columns):
            raise ValueError(
                "The dataset has a label column, but MixedNegativeDataCollator uses the label slot for "
                "its sampling metadata. Remove the label column."
            )
        anchor_column, positive_column, *other_columns = text_columns
        if set(self.extra_item_id_columns) - set(other_columns):
            raise ValueError("extra_item_id_columns keys must name additional candidate text columns.")
        if self.item_id_column and set(other_columns) - set(self.extra_item_id_columns):
            raise ValueError("Explicit item IDs require extra_item_id_columns for every extra text column.")
        if self.uniform_column in text_columns:
            raise ValueError(f"Column name {self.uniform_column!r} is reserved for the uniform negatives.")

        batch = super().__call__(rows)

        batch_size = len(rows)

        if self.query_id_column:
            query_ids = [stable_id(row[self.query_id_column]) for row in features]
        else:
            query_ids = self._ids([row[anchor_column] for row in rows])

        if self.item_id_column:
            positive_ids = [stable_id(row[self.item_id_column]) for row in features]
        else:
            positive_ids = self._ids([row[positive_column] for row in rows])
        if any(frequency.count(i) == 0 for i in positive_ids):
            raise ValueError("Positive ID absent from item_frequency; check the split and ID convention.")

        cand_ids = list(positive_ids)
        cand_source = [SOURCE_IN_BATCH] * batch_size
        for column in other_columns:
            if column in self.extra_item_id_columns:
                cand_ids.extend(stable_id(row[self.extra_item_id_columns[column]]) for row in features)
            else:
                cand_ids.extend(self._ids([row[column] for row in rows]))
            cand_source.extend([SOURCE_OTHER] * batch_size)

        if self.num_uniform > 0:
            if self.corpus is None or self._corpus_positions is None or self._corpus_unique_ids is None:
                raise ValueError("Uniform sampling requires an initialized corpus.")

            picks = self._process_rng().sample(range(self.corpus_size), self.num_uniform)
            texts = [self.corpus[int(self._corpus_positions[i])] for i in picks]
            cand_ids.extend(int(self._corpus_unique_ids[i]) for i in picks)
            cand_source.extend([SOURCE_UNIFORM] * self.num_uniform)

            # Tokenize exactly like the positive column (same prompt, same Router task).
            router_mapping = self._resolve_router_mapping(batch)
            prompts = self._resolve_prompts(batch)
            task = self._get_task_for_column(positive_column, 1, router_mapping)
            prompt = None if prompts is None else self._get_prompt_for_column(prompts, positive_column)
            max_length = self._get_max_length_for_task(task)
            if max_length is not None:
                preprocessed = self.preprocess_fn(texts, prompt=prompt, task=task, max_length=max_length)
            else:
                preprocessed = self.preprocess_fn(texts, prompt=prompt, task=task)
            for key, value in preprocessed.items():
                batch[f"{self.uniform_column}_{key}"] = value
            if task is not None:
                batch[f"{self.uniform_column}_task"] = task

        batch["label"] = {
            "query_id": torch.tensor(query_ids, dtype=torch.long),
            "cand_id": torch.tensor(cand_ids, dtype=torch.long),
            "cand_count": torch.tensor([frequency.count(i) for i in cand_ids], dtype=torch.long),
            "cand_in_corpus": torch.tensor([i in self._corpus_id_set for i in cand_ids], dtype=torch.bool),
            "cand_source": torch.tensor(cand_source, dtype=torch.long),
            "num_rows": torch.tensor(frequency.num_rows, dtype=torch.long),
            "corpus_size": torch.tensor(self.corpus_size, dtype=torch.long),
        }

        return batch
