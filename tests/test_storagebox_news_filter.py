"""The news off-box leg must select exactly the non-rederivable evidence.

Task 450 found `stage-*/pages/` — the only copy of the exact publisher
responses `news_article.original` is judged against — missing from the rules,
while `raw/` (decoded, truncated text, and NOT a substitute) was selected. A
rule that is never exercised is a rule nobody can trust, so this test parses the
`NEWS_FILTER` array out of the real script and hands it to real rsync.
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "scripts" / "storagebox_sync.sh"


from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("deploy/scripts/storagebox_sync.sh")  # the public cut withholds this file; skip there (task 2382)


def news_filter() -> list[str]:
    body = re.search(r"^NEWS_FILTER=\((.*?)^\)", SCRIPT.read_text(),
                     re.S | re.M)
    assert body, "NEWS_FILTER array not found in storagebox_sync.sh"
    args = []
    for line in body.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        flag, _, value = line.partition("=")
        args.append(f"{flag}={ast.literal_eval(value)}")
    return args


def build_tree(root: Path) -> None:
    files = [
        # non-rederivable news evidence
        "stage-2026-08-28/raw/www.srf.ch.raw.jsonl.gz",
        "stage-2026-08-28/pages/receipts/abc.json",
        "stage-2026-08-28/pages/original/publisher_pages/ab/abc.html.gz",
        "stage-2026-08-28/pages/original/publisher_transport/ab/abc.json.gz",
        "stage-2026-08-28/hn_linked/receipts/1.json",
        "stage-2026-08-28/preservation/chmedia/RUN.json",
        "stage-2026-08-28/preservation/chmedia/chunks.jsonl.gz",
        "stage-2026-08-28/preservation/chmedia/chunks.receipt.json",
        "stage-2026-08-28/news_documents.jsonl",
        "pack-20260828-task450/news-points.jsonl.gz",
        "delta-20260828T000000Z/promote_plan.json",
        "state/news-point-candidates-bootstrap.jsonl",
        # rederivable scratch that must NOT be shipped
        "stage-2026-08-28/news_rows.jsonl",
        "stage-2026-08-28/all.enriched.jsonl",
        "stage-2026-08-28/gated/passages_ok.jsonl",
        "consent-cache/srf.ch.json",
        # a decoy: an unanchored directory of the same name elsewhere
        "scratch/raw/decoy.gz",
        "scratch/pages/decoy.gz",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync absent")
def test_the_news_leg_ships_the_sealed_originals_and_nothing_rederivable(
        tmp_path):
    source, dest = tmp_path / "src", tmp_path / "dst"
    build_tree(source)
    dest.mkdir()
    subprocess.run(
        ["rsync", "-a", *news_filter(), f"{source}/", f"{dest}/"], check=True)
    shipped = {p.relative_to(dest).as_posix()
               for p in dest.rglob("*") if p.is_file()}

    assert "stage-2026-08-28/pages/receipts/abc.json" in shipped
    assert ("stage-2026-08-28/pages/original/publisher_pages/ab/abc.html.gz"
            in shipped)
    assert ("stage-2026-08-28/pages/original/publisher_transport/ab/abc.json.gz"
            in shipped)
    assert "stage-2026-08-28/raw/www.srf.ch.raw.jsonl.gz" in shipped
    assert "stage-2026-08-28/hn_linked/receipts/1.json" in shipped
    assert "stage-2026-08-28/preservation/chmedia/RUN.json" in shipped
    assert "stage-2026-08-28/preservation/chmedia/chunks.jsonl.gz" in shipped
    assert "stage-2026-08-28/preservation/chmedia/chunks.receipt.json" in shipped
    assert "stage-2026-08-28/news_documents.jsonl" in shipped
    assert "pack-20260828-task450/news-points.jsonl.gz" in shipped
    assert "delta-20260828T000000Z/promote_plan.json" in shipped
    assert "state/news-point-candidates-bootstrap.jsonl" in shipped

    assert not {name for name in shipped
                if name.endswith(("news_rows.jsonl", "all.enriched.jsonl",
                                  "passages_ok.jsonl"))}
    assert not {name for name in shipped if name.startswith("consent-cache/")}
    # Anchored rules: a same-named directory elsewhere is not swept in.
    assert not {name for name in shipped if name.startswith("scratch/")}


def test_the_news_leg_never_deletes_on_the_box():
    """Retention on the box is a policy decision, not local disk hygiene."""
    text = SCRIPT.read_text()
    news_line = [line for line in text.splitlines()
                 if line.startswith("sync_leg news ")]
    assert len(news_line) == 1
    assert "--delete" not in news_line[0]


def test_the_exports_leg_is_bounded_never_deletes_and_skips_part_files():
    """Task 716: `exports/` (task705, removed-rows) ships to `aw-exports/` at the sync's own <= 8M rate, with
    no --delete (a restore copy outlives local disk hygiene) and without a half-written `.part` file."""
    text = SCRIPT.read_text()
    legs = [line for line in text.splitlines() if line.startswith("sync_leg exports ")]
    assert len(legs) == 1
    assert "aw-exports/" in legs[0]
    assert "--delete" not in legs[0]
    assert "--exclude='*.part'" in legs[0]
    assert ": \"${AW_SRC_EXPORTS:=$(aw_setting_in \"$APP\" work_root)/exports/}\"" in text
    assert ': "${AW_BWLIMIT:=8M}"' in text and '--bwlimit="$AW_BWLIMIT"' in text
