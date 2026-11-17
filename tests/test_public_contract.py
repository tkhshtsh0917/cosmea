"""Regression checks for portable, bilingual, self-contained public documentation."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit(tmp_path, monkeypatch):
    """Load the development checker against an isolated public document tree."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_public.py"
    spec = importlib.util.spec_from_file_location("cosmea_public_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in module.PUBLIC_DOCS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Public documentation\n")
    return module


def test_relative_bilingual_navigation_is_portable(audit):
    (audit.ROOT / "README.md").write_text("[Japanese](README.ja.md)\n[API](docs/api.md)\n")
    (audit.ROOT / "docs/ja/api.md").write_text("[English](../api.md)\n")

    assert audit.inspect_public(audit.public_files()) == []


def test_personal_path_is_rejected(audit):
    personal_path = "/".join(["", "Users", "example-user", "notes", "README.md"])
    errors = audit.inspect_text("README.md", personal_path.encode())

    assert any("machine-specific" in error for error in errors)


def test_missing_japanese_document_is_rejected(audit):
    (audit.ROOT / "README.ja.md").unlink()
    errors = audit.inspect_public(audit.public_files())

    assert "missing public document: README.ja.md" in errors


def test_existing_but_unpublished_link_is_rejected(audit):
    hidden = audit.ROOT / "local-note.md"
    hidden.write_text("Private notes\n")
    (audit.ROOT / "README.md").write_text("[Details](local-note.md)\n")
    files = [path for path in audit.public_files() if path != hidden]
    errors = audit.inspect_public(files)

    assert any("link target is not published" in error for error in errors)


def test_link_outside_checkout_is_rejected(audit):
    (audit.ROOT / "README.md").write_text("[Details](../personal-note.md)\n")
    errors = audit.inspect_public(audit.public_files())

    assert any("outside public root" in error for error in errors)


def test_display_math_is_not_a_link_but_surrounding_links_are_checked(audit):
    formula = "$$\n" + r"\operatorname{sg}[\sigma(z-F)](z-F)" + "\n$$\n"
    valid = formula + "\n[Japanese](README.ja.md)\n"

    assert audit.inspect_text("README.md", valid.encode(), published=audit.PUBLIC_DOCS) == []

    invalid = valid + "\n[Missing](missing.md)\n" + formula
    errors = audit.inspect_text("README.md", invalid.encode(), published=audit.PUBLIC_DOCS)

    assert errors == ["README.md: link target is not published missing.md"]


def test_display_math_does_not_hide_personal_paths(audit):
    personal_path = "/".join(["", "Users", "example-user", "notes", "README.md"])
    text = f"$$\n{personal_path}\n$$\n"
    errors = audit.inspect_text("README.md", text.encode(), published=audit.PUBLIC_DOCS)

    assert any("machine-specific" in error for error in errors)
