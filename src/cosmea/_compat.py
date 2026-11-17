"""Resolve renamed options while accepting the previous keyword spelling."""

from typing import TypeVar

T = TypeVar("T")


def renamed_option(value: T | None, legacy_value: T | None, name: str, legacy_name: str) -> T | None:
    if value is not None and legacy_value is not None and value != legacy_value:
        raise ValueError(f"{name} and {legacy_name} must agree when both are supplied.")
    return value if value is not None else legacy_value
