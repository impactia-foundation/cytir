"""The fedlex door mints the I1 id, tier B (task 1035, T18-b; ADR-067 am.3, ruling 31).

The twin's fedlex rows are tier B: `id = uuid5(AW_NS, canonical_url(url) + "#" + sha256(
normalize_digest_text(dense text)))`, and the twin stores the first 32 hex of that sha256 as
`content_digest`. `fedlex_pack_v3_twin_ids.json` holds seven real pack-v3 rows (dense text with
`{#eId}` markers) and the id and digest the live twin `aw_e10_twin` holds for the same url (one row
per url, read by keyed url filter on 2026-10-21). The door's mint over those texts must give the
twin's ids, through the same functions the drain uses, and every other id check must agree.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core import ids as ID  # noqa: E402
from cytir.core import projection as PROJ  # noqa: E402
from cytir.core.canon import canonical_url  # noqa: E402
from cytir.index import compose  # noqa: E402
from cytir.index import id_rule as IR  # noqa: E402
from cytir.sources.families import fedlex as FX  # noqa: E402

FIXTURE = json.loads((HERE / "fixtures" / "fedlex_pack_v3_twin_ids.json").read_text(encoding="utf-8"))
ROWS = FIXTURE["rows"]

#: The twin ids, pinned in the test file: a change to the mint shows here, not only in the fixture.
PINNED = {
    "https://www.fedlex.admin.ch/eli/cc/2022/491/fr#art_24": "65b518ae-68e8-5344-8512-d0483795c64a",
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en#art_341": "e0a26760-a365-57ea-978f-ce6a5e9a00ce",
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/fr#art_656_b": "10f77a62-ed4d-53cd-8524-a57405b98875",
    "https://www.fedlex.admin.ch/eli/cc/11/529_488_529/it#art_271": "4d72a799-fa5a-51d5-94bd-f5bc3794d156",
}


def _door_row(r: dict) -> dict:
    """A row shaped as `parse_akn` builds it before the identity stamp: curation `tier` A, an
    `anchor`, the dense text in `text`."""
    anchor = r["url"].split("#", 1)[1]
    return {"url": r["url"], "title": r["title"], "text": r["text"], "dense_text": r["text"],
            "lang": r["lang"], "anchor": anchor, "base_anchor": anchor, "tier": "A",
            "source_type": "official", "domain": "fedlex.admin.ch", "site_name": "Fedlex",
            "trust_tier": 1.0}


def test_the_fixture_holds_seven_rows_and_the_pins_match_it():
    assert len(ROWS) == 7
    by_url = {r["url"]: r["twin_id"] for r in ROWS}
    for url, pid in PINNED.items():
        assert by_url[url] == pid


@pytest.mark.parametrize("r", ROWS, ids=[r["url"].rsplit("/", 2)[-2] + r["url"].rsplit("#", 1)[1] for r in ROWS])
def test_the_doors_mint_equals_the_twin_id_and_digest(r):
    row = FX.stamp_i1_identity(_door_row(r))
    assert row["point_id"] == r["twin_id"]
    assert ID.content_digest(r["text"])[:32] == r["twin_content_digest"]      # the twin's stored prefix
    assert "anchor" not in row and row["base_anchor"]                         # tier B: no native anchor
    # the formula, spelled out once, as the cross-check (the door itself calls id_rule.mint_i1)
    assert ID.mint_point_id(canonical_url(r["url"]), ID.content_digest(r["text"])) == r["twin_id"]


@pytest.mark.parametrize("r", ROWS, ids=[r["url"].rsplit("#", 1)[1] + r["lang"] for r in ROWS])
def test_the_drains_own_projection_mints_the_same_id(r):
    """`project_record` classifies a row with an `anchor` as tier A (the b1 id). The door's row has
    none, so the projection is tier B and `mint_i1` of the projected payload is the twin id."""
    row = FX.stamp_i1_identity(_door_row(r))
    pid, _dense, _sparse, payload = PROJ.project_record({**row, "consent_search_ok": True})
    assert payload["tier"] == "B" and payload["content_digest"] == ID.content_digest(r["text"])
    assert pid == IR.mint_i1(payload)[0] == IR.rekey({**payload, "point_id": r["twin_id"]})[0] == r["twin_id"]


def test_a_row_that_kept_its_anchor_would_mint_the_old_b1_id_not_the_twin_id():
    """Why the door drops `anchor`: the projection's tier rule reads it."""
    r = ROWS[0]
    row = _door_row(r)
    _pid, _d, _s, payload = PROJ.project_record({**row, "consent_search_ok": True})
    assert payload["tier"] == "A"
    assert IR.mint_i1(payload)[0] != r["twin_id"]


def test_compose_checks_the_door_output_under_the_i1_tier_digest_scheme(tmp_path):
    assert compose.id_scheme_for_family("fedlex_exact") == compose.ID_SCHEME_I1_TIER
    rows = [FX.stamp_i1_identity(_door_row(r)) for r in ROWS]
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    got = compose.verify_identity(str(chunks), scheme=compose.id_scheme_for_family("fedlex_exact"))
    assert got == {"checked": 7, "matched": 7, "mismatches": []}
    rows[2]["point_id"] = "00000000-0000-5000-8000-000000000000"
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(compose.ComposeError, match="identity check FAILED"):
        compose.verify_identity(str(chunks), scheme=compose.ID_SCHEME_I1_TIER)


def test_a_changed_text_is_a_new_id_and_a_windowed_article_has_distinct_ids():
    r = ROWS[0]
    a = FX.stamp_i1_identity(_door_row(r))["point_id"]
    changed = dict(r, text=r["text"] + " Neu.")
    assert FX.stamp_i1_identity(_door_row(changed))["point_id"] != a
    body = " ".join(f"w{i}" for i in range(700))
    windows = FX.window(body, "art_9")
    assert len(windows) == 3
    ids = {FX.stamp_i1_identity(_door_row({**r, "url": f"https://www.fedlex.admin.ch/eli/cc/x/y/de#{an}",
                                           "text": t}))["point_id"] for an, t in windows}
    assert len(ids) == 3


def test_the_id_label_matches_the_contract_and_the_splitter_refusal_is_lifted():
    contract = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text(encoding="utf-8"))
    row = next(f for f in contract["families"] if f["family_id"] == "fedlex_exact")
    # the splitter refusal was ON by design (review 1037) until the per-gold zero-loss gate ran;
    # the gate passed and ADR-067 amendment 4 lifts it: the door now agrees with its contract
    assert FX.DOOR_SPLITTER == FX.DECLARED_SPLITTER == "window-240-words"
    assert FX.door_declaration_refusal(row["identity"]["minted_by"],
                                       (row.get("licence") or {}).get("spdx")) is None


def test_the_target_refusal_is_gone():
    assert not hasattr(FX, "target_id_rule_refusal")
