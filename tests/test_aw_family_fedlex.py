#!/usr/bin/env python3
"""Exemplar 1 of the source-family interface (task 683, S7): the fedlex dump, ported.

`cyt source run fedlex_exact` drives the SAME loader the old fedlex dump script always ran
(now `cytir.sources.families.fedlex`), through the family framework. The proof is a golden, not
a claim: the sealed replies #442 recorded (one small federal act, three languages) go through
the family run, and the rows it composes must equal, byte for byte, what the old script door
writes for the same replies.

  * the rows, serialised the way `fedlex_dump.main()` writes them, hash to the SAME
    `out_sha256` that `test_fedlex_offline_golden.py` pins (`2acab3ee...`, `last_seen` pinned);
  * with `last_seen` normalised they hash to the golden's `out_sha256_last_seen_X`;
  * the framework's own properties hold for a real family: the L0 seal holds the exact XML
    bytes and the resolved law metadata, every unit reaches one terminal, the manifest names
    the declaration it was made under, and the pack refuses to embed (fedlex's consent basis is
    still `unknown` in the contracts file: that is the point of `assert_embeddable`).

No network: every request the tape does not hold fails the test.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIX = HERE / "fixtures" / "fedlex_offline"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}


def _tape() -> dict:
    return json.loads((FIX / "cc-1948-195.tape.json").read_text(encoding="utf-8"))


def _expected() -> dict:
    return json.loads((FIX / "cc-1948-195.expected.json").read_text(encoding="utf-8"))


def _client(unexpected: list):
    """#442's sealed tape, replayed by the package's own tape transport."""
    from cytir.acquire.tape import Tape
    tape = Tape.load(_tape())
    tape.unexpected = unexpected
    return tape.client()


def _run(monkeypatch, tmp_path, *, today, consent=lambda h, d: {**OK, "host": h}):
    from cytir.sources import run_family

    monkeypatch.setattr("time.sleep", lambda _s: None)       # the politeness floor
    # The golden proves the ported loader, not the splitter guard (it stays ON, tested below):
    # let the splitter refusal through for the fixture run only.
    import cytir.sources.families.fedlex as FX
    monkeypatch.setattr(FX, "DOOR_SPLITTER", FX.DECLARED_SPLITTER)
    unexpected: list = []
    tape = _tape()
    manifest = run_family.run_family(
        "fedlex_exact", out_dir=tmp_path, today=today, http=_client(unexpected),
        consent=consent, options={"eli": tape["eli"], "langs": tape["langs"]},
        log=lambda m: None)
    return manifest, unexpected


def _fedlex_dump_bytes(pack_dir: pathlib.Path) -> bytes:
    """The pack's `records.jsonl` IS the file `fedlex_dump.main()` writes: the pack keeps the
    adapter's key order and `ensure_ascii=False`, one row per line."""
    return _strip_family_id((pack_dir / "records.jsonl").read_bytes())


def _strip_family_id(raw: bytes) -> bytes:
    """ADR-124: the door stamps a master-only `family_id` on every row; the legacy driver never
    wrote it. Drop it (key order and `ensure_ascii=False` kept) before comparing to the golden."""
    lines = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    return b"".join((json.dumps({k: v for k, v in r.items() if k != "family_id"},
                                ensure_ascii=False) + "\n").encode("utf-8") for r in lines)


def test_family_run_equals_the_old_door_on_the_sealed_golden(monkeypatch, tmp_path):
    """Same replies, two doors. The rows the family composes are the rows `fedlex_dump.main`
    writes: compared as parsed rows against a real `fedlex_dump.main()` run over the same tape,
    and as bytes against the hash #442 pinned."""
    want = _expected()
    manifest, unexpected = _run(monkeypatch, tmp_path / "family", today=want["last_seen_pinned"])
    assert not unexpected, unexpected
    assert manifest["records"]["count"] == want["rows"]

    # old door: the real main(), same tape
    import cytir.sources.families.fedlex as fd
    old_unexpected: list = []
    monkeypatch.setattr(fd, "H", _client(old_unexpected))
    monkeypatch.setattr(fd, "TODAY", want["last_seen_pinned"])
    monkeypatch.setitem(fd.COMMON, "last_seen", want["last_seen_pinned"])
    out = tmp_path / "old.jsonl"
    tape = _tape()
    monkeypatch.setattr(sys, "argv", ["fedlex_dump.py", "--eli", tape["eli"], "--langs",
                                      *tape["langs"], "--out", str(out)])
    fd.main()
    assert not old_unexpected
    old_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    from cytir.sources import run_family
    _m, new_rows = run_family.load_pack(tmp_path / "family" / "pack")
    assert {r.get("family_id") for r in new_rows} == {"fedlex_exact"}     # ADR-124 stamp, every row
    stripped = [{k: v for k, v in r.items() if k != "family_id"} for r in new_rows]
    assert stripped == old_rows                      # same rows, same order
    assert _fedlex_dump_bytes(tmp_path / "family" / "pack") == out.read_bytes()

    data = _fedlex_dump_bytes(tmp_path / "family" / "pack")
    assert hashlib.sha256(data).hexdigest() == want["out_sha256"]          # #442's golden


def test_normalised_last_seen_hash_is_the_goldens_and_is_date_independent(monkeypatch, tmp_path):
    want = _expected()
    a, _ = _run(monkeypatch, tmp_path / "a", today=want["last_seen_pinned"])
    b, _ = _run(monkeypatch, tmp_path / "b", today="2026-08-20")
    da, db = (_fedlex_dump_bytes(tmp_path / x / "pack") for x in ("a", "b"))
    assert hashlib.sha256(da).hexdigest() != hashlib.sha256(db).hexdigest()
    norm = lambda raw, day: raw.replace(f'"last_seen": "{day}"'.encode(), b'"last_seen": "X"')  # noqa: E731
    assert hashlib.sha256(norm(da, want["last_seen_pinned"])).hexdigest() == \
        want["out_sha256_last_seen_X"]
    assert norm(da, want["last_seen_pinned"]) == norm(db, "2026-08-20")
    assert a["run"]["today"] != b["run"]["today"]


def test_the_seal_holds_the_exact_bytes_and_the_resolved_metadata(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, today="2026-09-30")
    index = [json.loads(line) for line in
             (tmp_path / "l0" / "index.jsonl").read_text().splitlines()]
    assert len(index) == 3 and {r["meta"]["lang"] for r in index} == {"de", "fr", "it"}
    tape_xml = {t["url"]: t["body"].encode("utf-8") for t in _tape()["tape"]
                if t["url"].endswith("-xml.xml")}
    for row in index:
        sealed = (tmp_path / "l0" / row["sha256"][:2] / row["sha256"]).read_bytes()
        assert sealed == tape_xml[row["meta"]["source_version_url"]]
        assert hashlib.sha256(sealed).hexdigest() == row["sha256"]
        assert row["meta"]["law_meta"]          # resolved metadata sealed beside the bytes


def test_manifest_names_its_declaration_and_balances(monkeypatch, tmp_path):
    manifest, _ = _run(monkeypatch, tmp_path, today="2026-09-30")
    t = manifest["terminals"]
    assert (t["units_in"], t["units_admitted"], t["units_excluded"],
            t["units_quarantined"]) == (1, 1, 0, 0)
    assert manifest["schema"] == "aw.family-pack.v1"
    assert manifest["family_id"] == "fedlex_exact"
    c = manifest["contract"]
    assert c["status"] == "backfilled" and len(c["row_sha256"]) == 64
    assert c["chunk"]["declared"] == "native-chunked"   # contract and adapter agree (ruling 27)
    assert manifest["consent"]["checked"] is True
    assert manifest["consent"]["verdicts"][0]["host"] == "fedlex.data.admin.ch"


def test_a_declined_host_excludes_the_unit_before_anything_is_fetched(monkeypatch, tmp_path):
    declined = lambda h, d: {"host": h, "bucket": "declined", "status": "ok",  # noqa: E731
                             "search_ok": False, "train_ok": False}
    manifest, unexpected = _run(monkeypatch, tmp_path, today="2026-09-30", consent=declined)
    assert unexpected == []                         # nothing was requested
    assert manifest["terminals"]["excluded_reasons"] == {
        "consent_declined:fedlex.data.admin.ch": 1}
    assert manifest["records"]["count"] == 0


def test_an_unresolved_consent_is_held_not_declined(monkeypatch, tmp_path):
    def boom(host, direct):
        raise TimeoutError("our egress is down")
    manifest, unexpected = _run(monkeypatch, tmp_path, today="2026-09-30", consent=boom)
    assert unexpected == []
    assert manifest["terminals"]["quarantined_reasons"] == {
        "consent_unresolved:fedlex.data.admin.ch": 1}
    assert manifest["terminals"]["excluded_reasons"] == {}


def test_the_pack_clears_the_consent_basis_refusal_now_that_fedlex_declares_one(monkeypatch, tmp_path):
    """S6 left the generic embed door without a consent check. The family pack has one: a
    declared basis of `unknown` is a refusal (corpus-doctrine #7). The POLICY PR of 2026-10-03
    (ruling 23.10) declares fedlex_exact's basis (dump route; URG Art. 5, explicit-grant), so the
    refusal on the basis no longer fires. The pack is still refused for any other reason the
    gate finds (an unchecked consent, a recorded decline)."""
    from cytir.sources import contract as contract_mod
    from cytir.sources import run_family
    _run(monkeypatch, tmp_path, today="2026-09-30")
    manifest, recs = run_family.load_pack(tmp_path / "pack")
    assert manifest["contract"]["consent_basis"] not in (None, contract_mod.UNKNOWN)
    run_family.assert_embeddable(manifest, recs)


def test_the_splitter_refusal_is_lifted_by_the_task_1040_gate_and_still_bites_on_a_mismatch(
        monkeypatch):
    """Task 1035c + review 1037 kept the refusal ON until the per-gold zero-loss gate of 1038 section 6
    ran. Task 1040 ran it (PASS on 3 legs x 2 instruments) and ADR-067 amendment 4 lifts it: the
    declaration is the splitter the door runs. The check itself stays: a door that splits otherwise
    than declared is refused before anything is fetched."""
    import cytir.sources.families.fedlex as FX
    assert FX.DECLARED_SPLITTER == FX.DOOR_SPLITTER == "window-240-words"
    assert FX.door_declaration_refusal(FX.DOOR_ID_RULE) is None
    monkeypatch.setattr(FX, "DECLARED_SPLITTER", "law-paragraph-splitter")
    msg = FX.door_declaration_refusal(FX.DOOR_ID_RULE)
    assert msg and "\n" not in msg and "splitter" in msg and "id rule" not in msg


def test_a_contract_edit_of_minted_by_is_what_flips_the_id_rule_check(monkeypatch):
    """The preflight reads `identity.minted_by` from the contract row it is given (the contract
    file's row in a real run); the door's constant is only compared with it."""
    import json
    import cytir.sources.families.fedlex as FX
    from cytir.sources import contract as contract_mod, run_family
    monkeypatch.setattr(FX, "DOOR_SPLITTER", FX.DECLARED_SPLITTER)          # isolate the id rule
    rows = json.loads((REPO / contract_mod.DEFAULT_PATH).read_text(encoding="utf-8"))["families"]
    row = next(r for r in rows if r["family_id"] == "fedlex_exact")
    ctx = type("C", (), {"contract": row, "options": {}})()
    FX.FedlexFamily().preflight(ctx)                                        # file as committed: passes
    edited = {**row, "identity": {**row["identity"], "minted_by": "text-anchor-v0"}}
    ctx = type("C", (), {"contract": edited, "options": {}})()
    with pytest.raises(run_family.RunError, match="id rule `text-anchor-v0`"):
        FX.FedlexFamily().preflight(ctx)


# ---- ruling 31 / task 1035 (ADR-067 am.3): the door mints the I1 id, tier B ---------------------------
def test_ruling_31_real_fedlex_rows_carry_the_i1_id_through_the_real_projection(monkeypatch, tmp_path):
    """The door's `point_id` is what the drain mints for the same row: `project_record` (which
    classifies by `anchor` and digests `text`) then `id_rule.mint_i1` of the projected payload."""
    from cytir.core import projection as PROJ
    from cytir.index import id_rule as IR
    want = _expected()
    _run(monkeypatch, tmp_path, today=want["last_seen_pinned"])
    from cytir.sources import run_family
    _m, rows = run_family.load_pack(tmp_path / "pack")
    assert len(rows) == want["rows"] > 0
    for r in rows:
        assert "anchor" not in r and r["base_anchor"]          # tier B: no native anchor
        pid, _dense, _sparse, payload = PROJ.project_record(
            {**r, "consent_search_ok": True}, require_license=False)
        assert payload["tier"] == "B" and len(payload["content_digest"]) == 64
        assert IR.mint_i1(payload)[0] == pid == r["point_id"]




# ---- task 1038 (RULING 52 + 53): the licence, and the door gives the stage guard its source text ---
def test_every_row_carries_the_public_domain_licence_and_the_door_is_not_blind(monkeypatch, tmp_path):
    """RULING 53: `license` = `public-domain` (Art. 5 URG), stamped by the door and equal to the
    contract row's `licence.spdx`. The guard block: every (act, language) doc is CHECKED against its
    source text at threshold 1.0 (a family that gave no source text is BLIND and the url-group
    supersede refuses it)."""
    from cytir.sources import contract as contract_mod, run_family
    want = _expected()
    manifest, unexpected = _run(monkeypatch, tmp_path, today=want["last_seen_pinned"])
    assert not unexpected
    _m, rows = run_family.load_pack(tmp_path / "pack")
    assert rows and {r["license"] for r in rows} == {"public-domain"}
    crow = next(r for r in json.loads((REPO / contract_mod.DEFAULT_PATH).read_text(encoding="utf-8"))["families"]
                if r["family_id"] == "fedlex_exact")
    assert crow["licence"]["spdx"] == "public-domain" and crow["licence"]["basis"] == "statute"
    assert crow["licence"]["inferred_from_hosting"] is False
    g = manifest["staged_text_guard"]
    assert (g["docs_checked"], g["docs_unchecked"], g["docs_shortfall"]) == (len(_tape()["langs"]), 0, 0)
    assert g["threshold_declared"] is True and g["min_coverage"] == 1.0 and g["lowest_coverage"] == 1.0


def test_the_preflight_names_a_licence_the_contract_does_not_declare(monkeypatch):
    import cytir.sources.families.fedlex as FX
    from cytir.sources import contract as contract_mod, run_family
    monkeypatch.setattr(FX, "DOOR_SPLITTER", FX.DECLARED_SPLITTER)          # isolate the licence
    row = next(r for r in json.loads((REPO / contract_mod.DEFAULT_PATH).read_text(encoding="utf-8"))["families"]
               if r["family_id"] == "fedlex_exact")
    FX.FedlexFamily().preflight(type("C", (), {"contract": row, "options": {}})())     # as committed
    edited = {**row, "licence": {**row["licence"], "spdx": "SwissGov-Fedlex-Terms"}}
    with pytest.raises(run_family.RunError, match="licence `SwissGov-Fedlex-Terms`"):
        FX.FedlexFamily().preflight(type("C", (), {"contract": edited, "options": {}})())
