"""Check public navigation and distribution contents without private workspace data.

Run after building. File discovery respects Git ignores in checkouts and uses
explicit exclusions in source archives. This checks paths and packaging rules,
not the absence of every possible secret or vulnerability.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/|/private/(?:tmp|var)/")
PRIVATE_LINK = re.compile(r"\]\([^\n)]*(?:\.private/|research/|file://)")
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".txt", ".json", ".in", ".lock"}
EXCLUDED = {".git", ".private", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__", "build", "dist"}
PUBLIC_DOCS = {
    "README.md",
    "README.ja.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING.ja.md",
    "docs/api.md",
    "docs/ja/api.md",
    "docs/design.md",
    "docs/ja/design.md",
    "examples/README.md",
    "examples/README.ja.md",
    "NOTICE.md",
}


def public_files() -> list[Path]:
    """List files eligible for publication, including not-yet-committed files."""
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return sorted({ROOT / name.decode() for name in result.stdout.split(b"\0") if name})

    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not (set(p.relative_to(ROOT).parts) & EXCLUDED)
        and not any(part.endswith(".egg-info") for part in p.parts)
    )


def inspect_text(name: str, data: bytes, *, published: set[str] | None = None) -> list[str]:
    """Reject personal paths and private links; optionally check local Markdown targets."""
    text = data.decode("utf-8", errors="replace")
    errors = []
    if PERSONAL_PATH.search(text):
        errors.append(f"{name}: machine-specific home or temporary path")
    if PRIVATE_LINK.search(text):
        errors.append(f"{name}: private documentation dependency")
    if published is not None and name.endswith(".md"):
        # Display math can contain ](...) without defining a Markdown link.
        # Privacy checks above still inspect the complete, unmodified content.
        navigation = re.sub(r"(?ms)^\$\$[ \t]*\n.*?^\$\$[ \t]*$", "", text)
        for target in re.findall(r"\]\(([^\n)]+)\)", navigation):
            target = target.strip("<>").split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            destination = (ROOT / name).parent.joinpath(target).resolve()
            try:
                relative = destination.relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                errors.append(f"{name}: link outside public root {target}")
                continue
            if relative not in published:
                errors.append(f"{name}: link target is not published {target}")
    return errors


def inspect_public(files: list[Path]) -> list[str]:
    """Require both documentation languages and links within the public inventory."""
    published = {path.relative_to(ROOT).as_posix() for path in files}
    errors = [f"missing public document: {name}" for name in sorted(PUBLIC_DOCS - published)]
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if set(path.relative_to(ROOT).parts) & {".private", "research"}:
            errors.append(f"{name}: private material is eligible for publication")
        if path.suffix in TEXT_SUFFIXES:
            errors.extend(inspect_text(name, path.read_bytes(), published=published))
    return errors


def inspect_artifacts() -> list[str]:
    """Require wheel and sdist, reject private material, and validate sdist test inputs."""
    errors = []
    wheels = list((ROOT / "dist").glob("*.whl"))
    sdists = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return ["dist must contain exactly one wheel and one sdist for this build"]

    with zipfile.ZipFile(wheels[0]) as archive:
        entries = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        if "cosmea/functional.py" not in entries:
            errors.append("wheel: missing runtime package")
        for name in entries:
            if not (name.startswith("cosmea/") or ".dist-info/" in name):
                errors.append(f"wheel: unexpected payload {name}")
        for name, data in entries.items():
            errors.extend(inspect_text(name, data))

    with tarfile.open(sdists[0]) as archive:
        entries = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            name = member.name.split("/", 1)[1]
            entries[name] = stream.read()
        required = PUBLIC_DOCS | {
            "mise.toml",
            ".python-version",
            "uv.lock",
            "scripts/check_public.py",
            "examples/train_example.py",
            "examples/train_from_csv.py",
            "tests/test_integration.py",
        }
        for name in sorted(required - entries.keys()):
            errors.append(f"sdist: missing {name}")
        for name, data in entries.items():
            if set(Path(name).parts) & {".private", "research", ".venv"}:
                errors.append(f"sdist: private/development environment payload {name}")
            errors.extend(inspect_text(name, data, published=set(entries)))
    return errors


def main() -> None:
    """Exit nonzero on public-content or packaging contract violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args()
    files = public_files()
    errors = inspect_public(files)
    if args.artifacts:
        errors.extend(inspect_artifacts())
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Public content checks passed ({len(files)} files; artifacts={args.artifacts}).")


if __name__ == "__main__":
    main()
