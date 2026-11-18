"""Expose public Markdown to MkDocs using its original repository paths."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

ROOT = Path(__file__).resolve().parents[1]


@event_priority(100)
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Keep relative GitHub links valid without copying documents on disk."""
    # Scope the site to documentation, the linked licence and downloadable examples.
    root_docs = (
        "README.md",
        "README.ja.md",
        "CONTRIBUTING.md",
        "CONTRIBUTING.ja.md",
        "AGENTS.md",
        "NOTICE.md",
        "LICENSE",
    )
    docs = [file for file in files if Path(file.abs_src_path).is_relative_to(ROOT / "docs")]
    examples = [
        path.relative_to(ROOT).as_posix() for path in (ROOT / "examples").iterdir() if path.suffix in {".md", ".py"}
    ]
    for file in docs:
        files.remove(file)

    paths = (*root_docs, *(f"docs/{file.src_uri}" for file in docs), *sorted(examples))
    for path in (*paths, "experiments/diagnostics/stationarity.py"):
        files.append(File(path, str(ROOT), config.site_dir, config.use_directory_urls))
    return files


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Append the generated sections to the API index used on GitHub."""
    if page.file.src_uri == "docs/reference.md":
        return f"{markdown.rstrip()}\n\n{(ROOT / 'scripts/api_reference.md').read_text(encoding='utf-8')}"
    return markdown


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Declare the primary language for Japanese pages of the default theme."""
    uri = page.file.src_uri
    if uri.endswith(".ja.md") or uri.startswith("docs/ja/"):
        return output.replace('<html lang="en"', '<html lang="ja"', 1)
    return output
