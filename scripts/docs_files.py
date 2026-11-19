"""Expose public Markdown to MkDocs using its original repository paths."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = (
    ("en", "English", "README.md"),
    ("ja", "日本語", "README.ja.md"),
)


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


def language_of(uri: str) -> str:
    """Recognize Japanese Markdown in repository paths."""
    return "ja" if uri.endswith(".ja.md") or uri.startswith("docs/ja/") else "en"


def translated_uri(uri: str, language: str) -> str:
    """Find the peer using repository naming conventions for translations."""
    if language == language_of(uri):
        return uri
    if language == "en":
        if uri.startswith("docs/ja/"):
            return uri.replace("docs/ja/", "docs/", 1)
        return uri.removesuffix(".ja.md") + ".md"
    if uri.startswith("docs/"):
        return uri.replace("docs/", "docs/ja/", 1)
    return uri.removesuffix(".md") + ".ja.md"


def language_context(pages: list[File], uri: str) -> dict:
    """Use the matching page if available and the localized quickstart otherwise."""
    current = language_of(uri)
    files = {file.src_uri: file for file in pages}
    _, label, home_uri = next(locale for locale in LANGUAGES if locale[0] == current)
    return {
        "language": current,
        "language_label": label,
        "language_home": files[home_uri].page,
        "language_choices": [
            (code, label, files.get(translated_uri(uri, code), files[home_uri]).url)
            for code, label, home_uri in LANGUAGES
        ],
    }


def on_page_context(context: dict, page: Page, config: MkDocsConfig, nav) -> dict:
    """Set language links for the documentation header."""
    return {**context, **language_context(context["pages"], page.file.src_uri)}


def on_template_context(context: dict, template_name: str, config: MkDocsConfig) -> dict:
    """Keep language selection available when GitHub Pages serves a 404."""
    if template_name == "404.html":
        return {**context, **language_context(context["pages"], "README.md")}
    return context


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Declare the primary language for Japanese pages of the default theme."""
    if language_of(page.file.src_uri) == "ja":
        return output.replace('<html lang="en"', '<html lang="ja"', 1)
    return output
