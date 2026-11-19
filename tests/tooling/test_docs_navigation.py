"""Check language links in the generated MkDocs site."""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin

import pytest

SITE_URL = "https://tkhshtsh0917.github.io/cosmea/"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    destination = tmp_path_factory.mktemp("mkdocs")
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(destination)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    return destination


def language_links(site, page):
    document = (site / page / "index.html").read_text(encoding="utf-8")
    options = document.split('id="language-options"', 1)[1].split("</ul>", 1)[0]
    links = re.findall(r'<a href="([^"]+)" lang="(en|ja)" hreflang="\2"', options)
    return {language: urljoin(SITE_URL + page, href) for href, language in links}


@pytest.mark.parametrize(
    ("english", "japanese"),
    [
        ("", "README.ja/"),
        ("examples/", "examples/README.ja/"),
        ("docs/api/", "docs/ja/api/"),
        ("docs/design/", "docs/ja/design/"),
        ("docs/benchmark/", "docs/ja/benchmark/"),
        ("CONTRIBUTING/", "CONTRIBUTING.ja/"),
    ],
)
def test_language_picker_links_to_the_same_guide(site, english, japanese):
    expected = {"en": SITE_URL + english, "ja": SITE_URL + japanese}

    assert language_links(site, english) == expected
    assert language_links(site, japanese) == expected


@pytest.mark.parametrize("page", ["docs/reference/", "NOTICE/"])
def test_shared_pages_link_to_the_japanese_quickstart(site, page):
    assert language_links(site, page) == {"en": SITE_URL + page, "ja": SITE_URL + "README.ja/"}
