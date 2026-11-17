"""Item identity and item-frequency bookkeeping.

The proposal distribution of in-batch negatives is the empirical distribution of *positive items*
in the training set, so everything the logQ correction needs is a count per item. For a finite
`datasets.Dataset` those counts are exact; no streaming estimator is required.
"""

from __future__ import annotations

import hashlib
import numbers
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def stable_id(value: Any) -> int:
    """Deterministic signed 64-bit ID for strings, bytes or signed int64 integers.

    Python's built-in ``hash`` is salted per process, which would give every DataLoader worker a
    different id for the same text. blake2b is stable across processes and machines.
    """
    if isinstance(value, numbers.Integral):
        value = int(value)
        if not -(2**63) <= value < 2**63:
            raise ValueError("Integer IDs must fit in signed int64.")
        return value
    if not isinstance(value, (str, bytes)):
        raise TypeError("IDs must be strings, bytes or signed int64 integers.")
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=True)


@dataclass
class ItemFrequency:
    """Exact count of how often each item occurs as a positive in the training set.

    Attributes:
        counts: item id -> number of training rows whose positive is that item.
        num_rows: number of training rows N.
    """

    counts: dict[int, int] = field(default_factory=dict)
    num_rows: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_rows, numbers.Integral)
            or self.num_rows < 1
            or any(not isinstance(c, numbers.Integral) or c < 1 for c in self.counts.values())
            or sum(self.counts.values()) != self.num_rows
        ):
            raise ValueError("Frequency requires positive integer counts summing to num_rows > 0.")
        for item_id in self.counts:
            if not isinstance(item_id, numbers.Integral):
                raise TypeError("counts keys must be stable integer IDs; use from_ids for raw IDs.")
            stable_id(item_id)

    @classmethod
    def from_ids(cls, item_ids: Iterable[Any]) -> ItemFrequency:
        """Consume raw IDs once; reject an empty collection or unsupported ID values."""
        counter: Counter[int] = Counter(stable_id(item) for item in item_ids)
        return cls(counts=dict(counter), num_rows=sum(counter.values()))

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        positive_column: str = "positive",
        item_id_column: str | None = None,
    ) -> ItemFrequency:
        """Count positives of a ``datasets.Dataset`` (or anything indexable by column name).

        If ``item_id_column`` is given, items are identified by that column (recommended when the
        same item can have several text renderings); otherwise by the hash of the positive text.
        """
        column: Sequence[Any] = dataset[item_id_column or positive_column]
        return cls.from_ids(column)

    def count(self, item_id: int) -> int:
        """Return a stable integer ID's count, or zero when it is outside the log."""
        return self.counts.get(item_id, 0)

    def __len__(self) -> int:
        return len(self.counts)
