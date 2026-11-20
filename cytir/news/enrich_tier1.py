#!/usr/bin/env python3
"""Stamp the retrieval facets required by the tier-1/news projection.

Usage: python3 -m cytir.news.enrich_tier1 INPUT.jsonl HOST_META.json OUTPUT.jsonl

The implementation lives in the principal pipeline so immutable releases do
not depend on a mutable task-workspace copy.

**Task 638 (2026-09-22) — license citation is now required.** Task 636 found
that `cytir.core.license`'s host-inheritance table had drifted from an
uncited entry (canada.ca); task 638 chasing that found this module had the
SAME defect one layer earlier and worse: `HOST_META.json`'s `license` field
was fed straight into `record["license"]` with no check at all, and
task 286's `build_tier1_queue` script's `norm_license_seed(topic)` had been
populating it from a bare topic class (gov/ai-tech/other -> "gov-open" /
"permissive-or-fairuse-snippet" / "reference-open") for every one of Tier-1's
138 curated seed domains — never an instrument read. Because this module runs
BEFORE `build_aw_b1.py`'s `aw_b1_license.resolve_license` and that resolver's
rule 1 trusts any already-present `license` forever, a heuristic stamp landed
here pre-empted resolve_license's carefully-cited `_HOST_LICENSE` table
entirely, even for hosts that table already gets right (fedlex.admin.ch,
gesetze-im-internet.de, ...). See the task 638 licence-stamp audit.

Fix: `metadata["license"]` is honoured only when the SAME host_meta entry also
carries a non-empty `license_citation` (a URL + operative-sentence quote, or a
statutory citation — same shape as `aw_b1_license._Entry.citation`). No
citation -> no stamp. `record["license"]` stays unset, and downstream
resolve_license's `None` fallback then applies (the row is search-only under
ADR-039's fair-use/snippet posture; never a fabricated permissive grant).
`license_note` (freeform commentary, already used by some host_meta entries)
is NOT a citation and does not satisfy this requirement — memory
`licence-basis-from-the-instrument`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from cytir.core.country import KEEP_EXISTING, country_for_record


VALID_COUNTRIES = {"CH", "FR", "DE", "CA", "US", "BE", "INTL"}


def hint_to_country(hint: Any) -> str | None:
    """Map an explicit feeder hint to the indexed country facet."""
    if not isinstance(hint, str) or not hint.strip():
        return None
    normalized = hint.strip().upper()
    if normalized == "EU":
        return "INTL"
    if normalized in VALID_COUNTRIES:
        return normalized
    return None


def enrich_record(record: dict[str, Any], host_meta: dict[str, Any]) -> dict[str, Any]:
    domain = str(record.get("domain") or "")
    metadata = host_meta.get(domain, {})
    if not isinstance(metadata, dict):
        raise ValueError(f"host metadata for {domain!r} must be an object")

    # The sentinel is `aw_country.KEEP_EXISTING` == "__KEEP_EXISTING__", not the
    # literal "KEEP_EXISTING" this function compared against until task 448. The
    # comparison could never match, so a registered entity_card had the sentinel
    # itself written into its `country` facet. Import the constant instead of
    # restating it: one home for the value, no way to drift again.
    country = country_for_record(domain, record.get("source_type"))
    if country in (None, KEEP_EXISTING):
        country = hint_to_country(metadata.get("country_hint"))
    if country and country != KEEP_EXISTING:
        record["country"] = country
    else:
        record.setdefault("country", None)

    if not record.get("license") and metadata.get("license"):
        citation = metadata.get("license_citation")
        if isinstance(citation, str) and citation.strip():
            record["license"] = metadata["license"]
        # else: fail closed. A `license` with no `license_citation` is a
        # topic-class guess (or any other uncited value), not a declared
        # grant -- leave record["license"] unset so the ADR-039 licensing
        # exception applies downstream rather than a fabricated one.

    if not record.get("anchor"):
        anchor = str(record.get("title") or "")
        if not anchor:
            lines = str(record.get("text") or "").strip().splitlines()
            anchor = lines[0][:120] if lines else ""
        record["anchor"] = anchor.lstrip("# ").strip()

    record.setdefault("snippet", "")
    return record


def enrich_file(input_path: Path, metadata_path: Path, output_path: Path) -> dict[str, int]:
    host_meta = json.loads(metadata_path.read_text())
    if not isinstance(host_meta, dict):
        raise ValueError("host metadata root must be an object")

    counts = {"rows": 0, "country": 0, "license": 0}
    with input_path.open() as source, output_path.open("w") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"input line {line_number} must be an object")
            enriched = enrich_record(record, host_meta)
            counts["rows"] += 1
            counts["country"] += int(bool(enriched.get("country")))
            counts["license"] += int(bool(enriched.get("license")))
            target.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return counts


def main(argv: list[str] | None = None, prog: str = "enrich_tier1.py") -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(f"usage: {prog} INPUT.jsonl HOST_META.json OUTPUT.jsonl", file=sys.stderr)
        return 2
    counts = enrich_file(*(Path(arg) for arg in args))
    print(
        f"enriched {counts['rows']} rows -> {args[2]} "
        f"(country stamped {counts['country']}, license {counts['license']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
