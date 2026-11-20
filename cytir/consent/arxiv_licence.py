"""arXiv per-paper licences, read from the L0 OAI-PMH pages (ADR-115 r1).

The twin's arXiv rows carry `license = arXiv-metadata-CC0`. That string licenses the
METADATA snapshot, not the paper, so it cannot decide `train_ok`. The per-paper
licence exists only in the harvested OAI `arXiv` records (`<license>` element) under
`_dumps/arxiv/arxiv_raw/page_*.xml.gz`. Older papers have no `<license>` element:
they are "missing" and fail closed (train_ok=false).

    python -m cytir.consent.backfill arxiv-licences --raw-dir <arxiv_raw> --out <tsv>

The TSV is `arxiv_id<TAB>licence_url`, only for records that carry a licence.
"""
from __future__ import annotations

import collections
import glob
import gzip
import os
import re

_ID = re.compile(r"<id>([^<]+)</id>")
_LIC = re.compile(r"<license>([^<]+)</license>")


def parse_page(text: str):
    """Yield (arxiv_id, licence_url_or_empty) for each record in one OAI page."""
    for block in text.split("<record>")[1:]:
        m = _ID.search(block)
        if not m:
            continue                       # a deleted record: header only, no metadata
        lic = _LIC.search(block)
        yield m.group(1).strip(), (lic.group(1).strip() if lic else "")


def build_map(raw_dir: str, out_path: str) -> dict:
    """Parse every page under `raw_dir` (globbed, never a typed list). Returns stats."""
    pages = sorted(glob.glob(os.path.join(raw_dir, "page_*.xml.gz")))
    if not pages:
        raise SystemExit(f"no page_*.xml.gz under {raw_dir}: refusing an empty roster")
    n = with_lic = 0
    by_lic: collections.Counter = collections.Counter()
    with open(out_path + ".tmp", "w") as out:
        for p in pages:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for aid, lic in parse_page(f.read()):
                    n += 1
                    by_lic[lic or "(none)"] += 1
                    if lic:
                        with_lic += 1
                        out.write(f"{aid}\t{lic}\n")
    os.replace(out_path + ".tmp", out_path)
    return {"pages": len(pages), "records": n, "with_licence": with_lic,
            "by_licence": dict(by_lic.most_common())}


def load_map(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            aid, _, lic = line.rstrip("\n").partition("\t")
            out[aid] = lic
    return out
