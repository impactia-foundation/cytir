"""Regression checks for the result-link presentation contract.

The API's ``url`` is already the complete fetch target.  ``anchor`` is an
identity/metadata field and must not be concatenated by the UI: Fedlex records
carry both ``...#art_32`` and ``anchor=art_32``, which historically rendered as
the broken ``...#art_32art_32`` link.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGES = ["webapp/index.html"]


@pytest.mark.parametrize("page", PAGES)
def test_result_links_use_complete_api_url_without_identity_anchor(page: str) -> None:
    html = (ROOT / page).read_text(encoding="utf-8")

    assert 'href="${h.url}"' in html
    assert '<div class="url">${esc(h.url)}</div>' in html
    assert '${h.url}${h.anchor' not in html
    assert '${esc(h.url)}${h.anchor' not in html


@pytest.mark.parametrize("page", PAGES)
def test_inline_javascript_parses(page: str, tmp_path: Path) -> None:
    """Catch template-literal breakage that static HTML assertions cannot."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    html = (ROOT / page).read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "inline.js"
    path.write_text(script, encoding="utf-8")

    subprocess.run([node, "--check", str(path)], check=True)
