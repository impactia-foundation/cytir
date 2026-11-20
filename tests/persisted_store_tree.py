"""The sealed corpus store tree the persisted-input tests share (one home for the fixture).

A minimal, sealed store: one L1 page and one L2-only page, written into `tmp_path`. Nothing is
fetched. Used by the persisted-index tests here and by the build-tree suite that tests the same
module (it puts this directory on `sys.path`, like `throwaway_qdrant`).
"""
from __future__ import annotations

import gzip
import json

import pytest

# -- the real record -------------------------------------------------------
REAL_URL = "https://developers.openai.com/api/docs/guides/function-calling"
REAL_PROD_ROWS = 41
REAL_L2_WORDS = 6731
# The page prod holds 41 rows of, and the only text of it anyone ever stored:
# L2 passages. No L0, no L1. Under the ADR-105 bar it is NOT loadable; under
# the ADR-093 route (require="l1b") it is, and the verdict says which.
PERSISTED_URL = "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006219126"
PERSISTED_SHA = "06cfcd0a62a3e60ffeb579637d4c0ffdb261b949a76104b42cdeadbeef000000"


@pytest.fixture()
def store_tree(tmp_path):
    """A minimal, sealed corpus store: one L1 page, one L2-only page."""
    root = tmp_path / "corpus"
    l1 = root / "l1" / "release=test" / "source=legi"
    l1.mkdir(parents=True)
    with gzip.open(l1 / "l1-markdown-00000.jsonl.gz", "wt",
                   encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "aw.l1-markdown.v1",
            "canonical_url": PERSISTED_URL,
            "url": PERSISTED_URL,
            "markdown": "# Code de commerce\n\n## Article L110-2\n",
            "markdown_sha256": PERSISTED_SHA}) + "\n")
    l2 = root / "wave=cc_phase2"
    l2.mkdir(parents=True)
    _write_parquet(l2 / "part-0000.parquet", [REAL_URL])
    return root


def _write_parquet(path, urls):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"url": list(urls),
                             "content_digest": ["d" * 40] * len(urls)}),
                   str(path))
