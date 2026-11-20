#!/usr/bin/env python3
# S3: literal uuid5 is the subject under test (this file pins the retired and card id expressions byte-for-byte against cytir.core)
"""Task 436 -- where point ids are allowed to be minted, and what must not move.

THE SHAPE OF THE PROBLEM (measured 2026-08-20, and it is not what was filed).
This repo does not have one identity contract with seven rogue copies. It has
THREE schemes, and each is really in use:

  S1  passage      aw_common.mint_passage_point_id(url, anchor, text)
                   = uuid5(NAMESPACE_URL, f"{url}#{anchor or ''}:{sha1(text)}")
                   -- the scheme most of the LIVE corpus is minted under.
  S2  entity card  uuid5(NAMESPACE_URL, url)   -- url-addressed so a card
                   upserts in place across refreshes. Written as a LITERAL in
                   ~10 card builders; it has no shared home at all. The hygiene
                   deduper depends on its collision property on purpose.
  S3  aw_b1        aw_b1_identity.mint_point_id(canonical_url, anchor_component)
                   = uuid5(AW_NAMESPACE, canonical_url + "#" + anchor)
                   -- the target contract (specs/identity.md), a DIFFERENT
                   namespace and a different input from S1 and S2.

So "migrate everything to aw_b1_identity" is not a refactor: for an
already-indexed record it is a re-id. These tests exist to keep that
distinction executable -- to let the safe migration happen and to make the
unsafe one fail loudly rather than quietly.
"""
from __future__ import annotations

import hashlib
import random
import uuid

import pytest

from cytir.core import common as aw_common  # noqa: E402
from cytir.core import ids as aw_b1_identity  # noqa: E402


# --------------------------------------------------------------------------
# The one migration that was safe: the N2 OpenAlex driver (deleted by task 1250)
# --------------------------------------------------------------------------
def _retired_openalex_formula(url: str, text: str) -> str:
    """Verbatim the expression the N2 OpenAlex driver carried before task 436."""
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#:{text_hash}"))


def _sample():
    """Records in the shapes this ingester actually emits, plus fuzz.

    `text` is stripped from the index at build time, so the ids of the indexed
    OpenAlex points cannot be re-derived from the corpus -- there is no real
    stored sample to replay. The proof therefore has to be over the input
    domain instead: the real URL shapes the picker produces (DOI resolver,
    publisher landing page, OpenAlex work URL) crossed with real abstract
    content, then 2000 randomized unicode pairs to cover the rest.
    """
    fixed = [
        ("https://doi.org/10.1038/s41586-021-03819-2",
         "Highly accurate protein structure prediction with AlphaFold\n"
         "Proteins are essential to life, and understanding their structure..."),
        ("https://doi.org/10.1016/j.cell.2020.02.052", "SARS-CoV-2 Cell Entry..."),
        ("https://link.springer.com/article/10.1007/s00190-021-01512-w", "Abstract"),
        ("https://openalex.org/W2741809807", ""),
        ("https://doi.org/10.1051/0004-6361/202039657", "Ünïcødé — em–dash ✓ 中文"),
        ("https://example.org/a?b=c&d=e#frag", "x" * 20000),
    ]
    rng = random.Random(436)
    alphabet = "abcXYZ0/:?#&=._-%é中✓ \n\t"
    for _ in range(2000):
        url = "https://" + "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
        fixed.append((url, text))
    return fixed


def test_openalex_migration_did_not_move_a_single_id():
    """The precondition for migrating it at all. If this ever fails, the
    OpenAlex points in the index are no longer addressable by this builder."""
    for url, text in _sample():
        assert _retired_openalex_formula(url, text) == \
            aw_common.mint_passage_point_id(url, None, text), (url, text[:40])


def test_openalex_content_digest_did_not_move_either():
    """`content_digest` was the same local sha1 literal. It answers "did the
    content change" and is compared across builds; moving it silently would
    invalidate every reuse decision for these records."""
    for url, text in _sample()[:50]:
        assert hashlib.sha1(text.encode("utf-8")).hexdigest() == aw_common.sha1(text)


# --------------------------------------------------------------------------
# The migrations that were NOT safe, pinned so nobody performs them by reflex
# --------------------------------------------------------------------------
#: Card builders minting S2 (uuid5(NAMESPACE_URL, url)) whose records are in the
#: index. Migrating any of them to S3 re-ids every card it ever wrote. That is a
#: corpus operation with a supersede plan, never a tidy-up commit.
#: Task 1407 moved the last one (the ctgov psy family) into `cytir/sources/families/ctgov_psy.py`, where it mints
#: through `cytir.core.common.mint_card_point_id` (the same S2 expression, in the legal home).
S2_CARD_BUILDERS: tuple[str, ...] = ()


def test_the_moved_card_builder_still_mints_the_scheme_its_records_are_under():
    """Do not fix a divergence into existence (task 436).

    The ctgov_psy registry-record builder writes S2 ids, `uuid5(NAMESPACE_URL, url)`, and its records
    are in the index under them. Task 1407 moved it into the package and routed the expression through
    `cytir.core.common.mint_card_point_id`. Pin that the helper is exactly the old expression and that the
    builder uses it, not the S3 minter (a different namespace would orphan every card)."""
    import uuid
    from cytir.core import common
    from cytir.sources.ctgov_psy import harvest
    url = "https://clinicaltrials.gov/study/NCT00000000"
    assert common.mint_card_point_id(url) == str(uuid.uuid5(uuid.NAMESPACE_URL, url))
    rec = harvest.make_record(url, "clinicaltrials.gov", "t", "t", "t", "B", 0.9, "registry_record", "en", "2026-10-30")
    assert rec["point_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, url))
    src = open(harvest.__file__, encoding="utf-8").read()
    assert "aw_b1_identity" not in src


def test_the_three_schemes_really_are_different():
    """The reason none of the above is a refactor. If two of these ever agree,
    something has been quietly redefined. Card scheme (S2) now minted via
    cytir.core.common.mint_card_point_id (task 1407); the expression is unchanged."""
    url, anchor, text = "https://example.org/a", None, "hello"
    s1 = aw_common.mint_passage_point_id(url, anchor, text)
    s2 = aw_common.mint_card_point_id(url)
    assert s2 == str(uuid.uuid5(uuid.NAMESPACE_URL, url))
    s3 = aw_b1_identity.mint_point_id(url, "")
    assert len({s1, s2, s3}) == 3, "identity schemes collapsed -- re-check specs/identity.md"
    assert aw_b1_identity.AW_NAMESPACE != uuid.NAMESPACE_URL


# --------------------------------------------------------------------------
# Stop the class growing
# --------------------------------------------------------------------------
#: Acquisition scripts (the private ingest scripts) that mint a point id from a
#: local `uuid5` literal, as measured 2026-08-20. This is a CEILING, not a
#: target: entries may leave it (with a corpus plan), none may be added. A new
#: ingester mints through `aw_common` or `aw_b1_identity`.
KNOWN_LOCAL_UUID5_INGESTERS: set[str] = set()   # emptied by task 1407 (ctgov_psy moved into cytir/)


def _calls_uuid5(path) -> bool:
    """True if the FILE CALLS uuid5, by AST.

    Not a substring match: half these files mention `uuid5(...)` in a docstring
    precisely to document which scheme their records are under, and a test that
    counts prose as code trains people to delete the prose.
    """
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "uuid5":
            return True
        if isinstance(fn, ast.Name) and fn.id == "uuid5":
            return True
    return False


def test_the_ratchet_can_actually_see_a_violation(tmp_path):
    """A guard that has never fired is an untested code path pointed at your
    data. Prove it detects a mint and ignores a docstring that mentions one."""
    good = tmp_path / "ingest_prose.py"
    good.write_text('"""point_id = uuid5(NAMESPACE_URL, url)."""\nx = 1\n')
    bad = tmp_path / "ingest_mints.py"
    bad.write_text("import uuid\npid = uuid.uuid5(uuid.NAMESPACE_URL, 'u')\n")
    assert _calls_uuid5(str(bad)) is True
    assert _calls_uuid5(str(good)) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
