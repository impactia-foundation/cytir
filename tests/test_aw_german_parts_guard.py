"""Task 710, guard 1 and guard 2: German rows must carry their de-compounded parts.

REAL ROWS. `tests/fixtures/german_parts_real_rows.json` holds 12 German rows read from the twin
(`aw_e8`) by id on 2026-10-05: 5 whose writer emitted parts (gesetze-im-internet, swisstopo,
bundesverfassungsgericht) and 7 whose writer did not (openlegaldata, Wikipedia DE, the task-564
fedlex delta, openalex, a CC-w7 page). Each carries its stored `bm25` vector, so the lattice reading
is on the real thing. The fixture records which tokens the rule splits, so no test needs a splitter;
one test (skipped without `compound_split`) proves that record is still the rule's answer.
A synthetic row is used only for the refusal branch and for a pack with no `lang`.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.index import german_parts_guard as G
from cytir.ops import german_parts_tripwire as T

FIXTURE = json.loads((pathlib.Path(__file__).parent / "fixtures" / "german_parts_real_rows.json").read_text())
ROWS = FIXTURE["rows"]
WITH = [r for r in ROWS if r["has_parts_on_twin"]]
WITHOUT = [r for r in ROWS if not r["has_parts_on_twin"]]
PARTS = {tok: tuple(p) for r in ROWS for tok, p in r["predicted_parts"].items()}


def _point(r, lang="de"):
    return {"id": r["id"], "vector": {"bm25": r["bm25"]},
            "payload": {"lang": lang, "title": r["title"], "site_name": r["site_name"],
                        "snippet": r["snippet"]}}


def _many(rows, n=40, lang="de"):
    """`n` lattice points cycling through `rows` (a content pack needs >= LATTICE_MIN_PREDICTED)."""
    out = []
    for i in range(n):
        p = _point(rows[i % len(rows)], lang=lang)
        p["id"] = f"{i:08d}"
        out.append(p)
    return out


def blind_rule(monkeypatch):
    """A process whose rule module cannot load (no splitter, no table): the real loader raises
    `DecompoundUnavailable`, so the real writer raises at its first German token and the real
    `external_predictor()` finds no predictor."""
    from cytir.core import decompound

    def unavailable():
        raise decompound.DecompoundUnavailable("simulated: no splitter in this process")
    monkeypatch.setattr(decompound, "_splitter", None)
    monkeypatch.setattr(decompound, "_build", unavailable)


def _predictor():
    return ("fixture", lambda tok: PARTS.get(tok, ()))


def _text(r):
    return " ".join(str(r.get(k) or "") for k in ("title", "site_name", "snippet"))


# --------------------------------------------------------------------- the real rows
def test_the_fixture_has_both_kinds_of_real_row():
    assert len(WITH) == 5 and len(WITHOUT) == 7
    assert {r["label"] for r in WITHOUT} == {"openlegaldata", "wikipedia-de", "fedlex-564-delta",
                                             "openalex", "cc-w7"}


def test_lattice_sees_half_integer_weights_exactly_where_the_writer_had_parts():
    for r in WITH:
        n, half = G.lattice(r["bm25"]["values"])
        assert n >= G.LATTICE_MIN_POSTINGS and half > 0, r["id"]
    for r in WITHOUT:
        n, half = G.lattice(r["bm25"]["values"])
        assert n >= G.LATTICE_MIN_POSTINGS and half == 0, r["id"]


def test_lattice_leaves_a_short_or_empty_vector_undecided():
    assert G.lattice([]) is None
    assert G.lattice([5.0]) is None           # no weight below the saturation ceiling


def test_a_pack_of_rows_with_parts_passes_on_the_lattice():
    rep = G.evaluate_points(_many(WITH), predictor=_predictor())
    assert rep["verdict"] == "PASS" and rep["predicted"] == 40 and rep["carried"] == 40
    assert rep["basis"] == "lattice" and rep["predictor"] == "fixture"


def test_a_pack_of_rows_without_parts_fails_on_the_lattice():
    rep = G.evaluate_points(_many(WITHOUT), predictor=_predictor())
    assert rep["verdict"] == "FAIL" and rep["carried"] == 0 and rep["share"] == 0.0


def _mixed(carried: int, total: int):
    pts = _many(WITH, carried) + _many(WITHOUT, total - carried)
    for i, p in enumerate(pts):
        p["id"] = f"{i:08d}"
    return pts


def test_lattice_threshold_is_the_one_constant_pinned_at_0_8_with_a_greater_equal_boundary():
    assert G.LATTICE_CARRY_SHARE_MIN == 0.8 and G.LATTICE_MIN_PREDICTED == 30
    at = G.evaluate_points(_mixed(32, 40), predictor=_predictor())             # 32/40 = 0.80 exactly
    assert at["share"] == 0.8 and at["threshold"] == G.LATTICE_CARRY_SHARE_MIN and at["verdict"] == "PASS"
    below = G.evaluate_points(_mixed(31, 40), predictor=_predictor())          # 0.775
    assert below["verdict"] == "FAIL"
    # a writer that loses 40 % of its parts (~0.59) passed the old 0.5 floor; it fails now
    loss = G.evaluate_points(_mixed(24, 40), predictor=_predictor())
    assert loss["share"] == 0.6 and loss["verdict"] == "FAIL"


def test_lattice_needs_30_predicted_rows_before_it_judges():
    few = G.evaluate_points(_mixed(0, 29), predictor=_predictor())             # 0 of 29: no evidence
    assert few["verdict"] == "UNJUDGED" and few["predicted"] == 29
    reason, lines = G.decide(few, "packX", guard_mode="refuse")
    assert reason is None and any("NOT JUDGED" in l for l in lines)
    enough = G.evaluate_points(_mixed(0, 30), predictor=_predictor())
    assert enough["verdict"] == "FAIL"


def test_a_pack_whose_predicted_rows_have_no_bm25_vector_warns_never_passes_silently():
    pts = _many(WITH, 40)
    for p in pts:
        del p["vector"]["bm25"]
    rep = G.evaluate_points(pts, predictor=_predictor())
    assert rep["verdict"] == "UNJUDGED" and rep["undecided"] == 40 and rep["predicted"] == 0
    for guard_mode in ("warn", "refuse"):
        reason, lines = G.decide(rep, "packX", guard_mode=guard_mode)
        assert reason is None and any("WARN packX: NOT JUDGED" in l for l in lines)


# --------------------------------------------------------------------- the policy
def test_warn_mode_never_refuses_and_says_so_loudly():
    rep = G.evaluate_points(_many(WITHOUT), predictor=_predictor())
    reason, lines = G.decide(rep, "packX", guard_mode="warn")
    assert reason is None
    assert any(l.startswith("[german_parts] WARN packX") and "AW_GERMAN_PARTS_GUARD" not in l
               and "refuse would refuse it" in l for l in lines)


def test_refuse_mode_refuses_the_pack_and_names_the_numbers():
    rep = G.evaluate_points(_many(WITHOUT), predictor=_predictor())
    reason, lines = G.decide(rep, "packX", guard_mode="refuse")
    assert reason and reason.startswith("german_parts: only 0 of 40 German rows")
    assert any(l.startswith("[german_parts] REFUSE packX") for l in lines)


def test_a_pack_with_no_german_rows_is_untouched():
    pts = [_point(r, lang="fr") for r in WITHOUT]
    rep = G.evaluate_points(pts, predictor=_predictor())
    assert rep["german_rows"] == 0 and rep["verdict"] == "PASS"
    assert G.decide(rep, "packX", guard_mode="refuse") == (None, [])


def test_a_pack_with_no_lang_is_counted_loudly_never_refused():
    pts = [_point(r) for r in WITHOUT]
    for p in pts:
        del p["payload"]["lang"]
    rep = G.evaluate_points(pts, predictor=_predictor())
    assert rep["no_lang_rows"] == 7 and rep["german_rows"] == 0
    reason, lines = G.decide(rep, "packX", guard_mode="refuse")
    assert reason is None
    assert any("7 of 7 rows have no `lang` field" in l and "nothing here can be checked" in l
               for l in lines)


def test_without_a_splitter_a_vector_only_pack_is_reported_not_checked_never_refused(monkeypatch):
    monkeypatch.setattr(G, "external_predictor", lambda: None)
    rep = G.evaluate_points([_point(r) for r in WITHOUT])
    assert rep["verdict"] == "NA"
    reason, lines = G.decide(rep, "packX", guard_mode="refuse")
    assert reason is None and any("NOT CHECKED" in l for l in lines)


# --------------------------------------------------------------------- the text basis
def _writer(with_parts: bool):
    """A synthetic writer over the REAL rows' text: whole tokens at 1.0, parts at 0.5 when it has the
    splitter. (The real writer needs the splitter; this keeps the refusal branch independent of it.)"""
    import re
    import zlib

    def write(text, lang):
        tf = {}
        for tok in re.findall(r"\w+", text.lower()):
            tf[tok] = tf.get(tok, 0.0) + 1.0
            if with_parts and lang == "de":
                for part in PARTS.get(tok, ()):
                    tf[part] = tf.get(part, 0.0) + 0.5
        return [zlib.crc32(t.encode()) for t in tf], list(tf.values())
    return write


def test_text_basis_blind_writer_fails_and_sighted_writer_passes_on_real_text():
    texts = [_text(r) for r in ROWS]
    counts = {"rows": len(texts), "german_rows": len(texts), "no_lang_rows": 0}
    blind = G.evaluate_texts(texts, counts, predictor=_predictor(), writer=_writer(False))
    sighted = G.evaluate_texts(texts, counts, predictor=_predictor(), writer=_writer(True))
    assert blind["verdict"] == "FAIL" and blind["carried"] == 0 and blind["predicted"] == len(ROWS)
    assert sighted["verdict"] == "PASS" and sighted["carried"] == sighted["predicted"] == len(ROWS)
    assert blind["threshold"] == G.CARRY_SHARE_MIN == 0.99


def test_a_writer_that_raises_counts_as_a_writer_without_parts():
    def boom(text, lang):
        raise RuntimeError("no splitter")
    texts = [_text(r) for r in ROWS]
    rep = G.evaluate_texts(texts, {"rows": 12, "german_rows": 12, "no_lang_rows": 0},
                           predictor=_predictor(), writer=boom)
    assert rep["verdict"] == "FAIL" and rep["carried"] == 0


def test_real_writer_without_the_splitter_is_caught_by_the_canary(monkeypatch):
    """Main's rule with no splitter (the three writers of task 710): the real `saturated_bm25`
    returns the same posting for 'de' and 'en'. No independent predictor exists in that process."""
    blind_rule(monkeypatch)
    assert G.external_predictor() is None      # the REAL predictor reads the blind rule: none
    assert G.writer_is_blind()
    texts = [_text(r) for r in ROWS]
    rep = G.evaluate_texts(texts, {"rows": 12, "german_rows": 12, "no_lang_rows": 0})
    assert rep["predictor"] == "none-writer-blind" and rep["verdict"] == "FAIL" and rep["carried"] == 0


def _has_splitter():
    try:
        from cytir.core import decompound
        decompound.ensure_ready()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_splitter(), reason="pinned charsplit not installed")
def test_real_writer_with_the_splitter_passes_and_the_fixture_still_matches_the_rule(monkeypatch):
    ext = G.external_predictor()
    assert ext and ext[0] == "cytir.core.decompound"
    for r in ROWS:
        for tok, parts in r["predicted_parts"].items():
            assert tuple(ext[1](tok)) == tuple(parts), tok
    assert not G.writer_is_blind()
    rep = G.evaluate_texts([_text(r) for r in ROWS], {"rows": 12, "german_rows": 12, "no_lang_rows": 0})
    assert rep["verdict"] == "PASS" and rep["carried"] == rep["predicted"] > 0


# --------------------------------------------------------------------- sampling, mode, wiring
def test_sampling_is_seeded_and_capped_and_counts_every_row():
    rows = [{"lang": "de" if i % 3 else None, "i": i} for i in range(30)]
    a, ca = G.sample_german(rows, lambda r: r["lang"], n=5)
    b, cb = G.sample_german(rows, lambda r: r["lang"], n=5)
    assert a == b and len(a) == 5 and ca == cb == {"rows": 30, "german_rows": 20, "no_lang_rows": 10}


def test_mode_defaults_to_warn_and_flips_by_one_setting(monkeypatch):
    from cytir.core import paths
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")          # no box overlay
    monkeypatch.delenv("AW_GERMAN_PARTS_GUARD", raising=False)
    paths.reset_settings_cache()
    assert G.mode() == "warn"
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "refuse")
    paths.reset_settings_cache()
    assert G.mode() == "refuse"
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "off")
    paths.reset_settings_cache()
    with pytest.raises(ValueError):
        G.mode()
    paths.reset_settings_cache()


def test_the_drain_refusal_stage_is_never_cached():
    """The verdict depends on the interpreter of THIS run, so a cached refusal would outlive the fix."""
    from cytir.index import promote_queue as PQ
    assert "german_parts" not in PQ._CACHEABLE_REFUSAL_STAGES
    from cytir.index.drain import drain as _drain_mod
    src = pathlib.Path(_drain_mod.__file__).read_text()
    assert src.count("GPG.guarded(GPG.check_joined, pack_id, joined)") == 1
    assert src.count("GPG.guarded(GPG.check_points, pack_id, points)") == 1
    assert src.count('stage="german_parts"') == 2


def test_check_joined_reads_the_drains_own_tuple_and_refuses_in_refuse_mode():
    joined = {r["id"]: ("dense", _text(r), {"lang": "de"}, {}) for r in ROWS}
    printed = []
    rep, reason = G.check_joined(joined, "packY", guard_mode="refuse", out=printed.append,
                                 predictor=_predictor(), writer=_writer(False))
    assert reason and rep["verdict"] == "FAIL" and printed and printed[0].startswith("[german_parts] REFUSE")
    rep, reason = G.check_joined(joined, "packY", guard_mode="refuse", out=printed.append,
                                 predictor=_predictor(), writer=_writer(True))
    assert reason is None and rep["verdict"] == "PASS"


def test_compose_records_the_count_in_warn_and_refuses_in_refuse(tmp_path, monkeypatch):
    """compose_pack on the real ERIC rows (no `lang`): the count is in the manifest, never a refusal.
    Then the stage-side judge is fed German rows with a blind writer: refuse mode writes no READY."""
    import gzip
    import cytir.index.compose as compose
    eric = pathlib.Path(__file__).parent.parent / "pipeline" / "testdata" / "task657_eric_real_rows.jsonl"
    if not eric.exists():
        pytest.skip("the real ERIC pack rows are not in this tree (the public cut)")
    # compose_pack stamps consent on these rows (cytir/consent/stamp.py ConsentStampRefused): the ERIC
    # host verdict is read from the sealed probe the family contract names, a private evidence file the
    # public cut withholds (task 2456). Without it every row is class D and the stamp refuses.
    from cytir.consent import stamp as _stamp
    fam = next(f for f in json.loads(_stamp.CONTRACTS_PATH.read_text())["families"]
               if f["family_id"] == "eric_official_pdf")
    pointers = [e.split("#")[0] for e in fam["consent"]["evidence"] if e.endswith("#host_consent_resolution")]
    missing = [p for p in pointers if not (_stamp.REPO / p).exists()]
    if missing:
        pytest.skip(f"the sealed ERIC consent probe {missing} is withheld from the public cut: "
                    "compose_pack would refuse to stamp these rows")
    rows = [json.loads(l) for l in eric.read_text().splitlines() if l.strip()]
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    vec = tmp_path / "vectors"
    vec.mkdir()
    (vec / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": 2, "dim": 1024, "dtype": "float16"}))
    from cytir.core import paths
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.delenv("AW_GERMAN_PARTS_GUARD", raising=False)
    paths.reset_settings_cache()
    res = compose.compose_pack(pack_id="gp-1", family="eric_official_pdf", chunks_dir=str(chunks),
                               vectors_dir=str(vec), collection="aw_e8", mrl=256,
                               queue_dir=str(tmp_path / "q"))
    gp = res["manifest"]["german_parts"]
    assert gp["no_lang_rows"] == 2 and gp["german_rows"] == 0 and gp["verdict"] == "PASS"

    # German rows through the stage-side judge with a blind real writer
    german = [{**rows[0], "lang": "de", "point_id": r["id"]} for r in ROWS]
    fed = iter(ROWS)
    with monkeypatch.context() as m:
        blind_rule(m)
        m.setattr(compose.PSP, "read_records", lambda d: iter(german))
        m.setattr(compose.PSP, "project_rows",
                  lambda it, **kw: ({next(it)["point_id"]: ("d", _text(next(fed)), {"lang": "de"}, {})}, 0))
        m.setenv("AW_GERMAN_PARTS_GUARD", "refuse")
        paths.reset_settings_cache()
        report, reason = compose._german_parts_report(str(chunks), res["manifest"], "gp-2")
    assert reason and reason.startswith("german_parts: only 0 of 12") and report["verdict"] == "FAIL"
    # and compose_pack turns that refusal into "no manifest, no READY"
    monkeypatch.setattr(compose, "_german_parts_report", lambda *a: (report, reason))
    with pytest.raises(compose.ComposeError, match="german_parts: only 0 of 12"):
        compose.compose_pack(pack_id="gp-2", family="eric_official_pdf", chunks_dir=str(chunks),
                             vectors_dir=str(vec), collection="aw_e8", mrl=256,
                             queue_dir=str(tmp_path / "q"))
    assert not (tmp_path / "q" / "gp-2" / "READY").exists()
    paths.reset_settings_cache()


# --------------------------------------------------------------------- guard 2: the tripwire
def test_tripwire_row_reading_matches_the_real_rows():
    for r in WITH:
        assert T.row_has_parts({"vector": {"bm25": r["bm25"]}}) is True
    for r in WITHOUT:
        assert T.row_has_parts({"vector": {"bm25": r["bm25"]}}) is False
    assert T.row_has_parts({"vector": {"bm25": {"indices": [1], "values": [1.0]}}}) is None


def test_tripwire_generation_key_names_pool_type_provenance_epoch_and_writer_month():
    key = T.gen_key({"source_type": "official", "provenance": "weko_docs", "prov_id": None,
                     "rule_epochs": {"chunk": 5}, "build_id": "aw_b1",
                     "last_seen": "2026-09-12"}, "weko.admin.ch")
    assert key == "weko.admin.ch|official|weko_docs|-|chunk5|aw_b1|2026-09"
    assert T.gen_key({"source_type": "reference", "rule_epochs": {"chunk": 1}, "build_id": "aw_b1"},
                     "de.wikipedia.org") == "de.wikipedia.org|reference|-|-|chunk1|aw_b1|-"


def test_restamping_a_consent_field_never_moves_a_row_between_generations():
    """2026-10-07: the consent backfill restamped payloads and the old key (consent month) moved
    596,520 openlegaldata rows to a new group. No consent field may be part of the key."""
    base = {"source_type": "official", "provenance": "endorsement-discovery", "rule_epochs": {"chunk": 1},
            "build_id": "aw_b1", "last_seen": "2026-09-23", "domain": "d.de", "lang": "de"}
    want = T.gen_key(base, "d.de")
    for restamp in ({"consent_checked_at": "2026-10-06T01:00:00Z"}, {"consent_checked_at": "2026-07-01"},
                    {"consent_basis": "robots", "search_ok": True, "train_ok": False},
                    {"consent_state": "x", "consent_source": "y", "stamps": {"a": 1}},
                    {"license": "cc-by", "trust_tier": 0.5, "snippet": "new text"}):
        assert T.gen_key({**base, **restamp}, "d.de") == want
    assert "consent" not in " ".join(T.PAYLOAD_KEYS)
    assert T.gen_key({**base, "last_seen": "2026-10-01"}, "d.de") != want   # a re-fetch is a new writer pass


def test_self_check_puts_every_row_in_exactly_one_generation():
    rep = T.run(_FakeQdrant(), min_generation=1000, sample=1000, alarm_below=0.5, windows=3,
                tail_windows=3, scan_cap=10_000, seed=1)
    sc = rep["self_check"]
    assert sc["ok"] and sc["rows_in_two_generations"] == 0 and sc["pool_rows_total"] == 2400
    assert abs(sc["estimated_rows_total"] - 2400) <= 2400 * T.EST_TOLERANCE
    g = {"a": T.Gen(), "b": T.Gen()}
    g["a"].ids, g["b"].ids = {"1", "2"}, {"2", "3"}              # id 2 in two groups
    g["a"].sampled = g["b"].sampled = 2
    out = [{"est_rows": 2}, {"est_rows": 2}]
    assert T.self_check(g, out, 4, {"p": 4})["rows_in_two_generations"] == 1
    assert not T.self_check(g, out, 4, {"p": 4})["ok"]
    g["b"].ids = {"3", "4"}
    assert T.self_check(g, out, 4, {"p": 4})["ok"]
    assert not T.self_check(g, [{"est_rows": 2}, {"est_rows": 40}], 4, {"p": 4})["ok"]


def test_a_failed_self_check_makes_the_state_error_never_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "Qdrant", lambda url, coll: _FakeQdrant())
    real = T.run
    def bad(*a, **k):
        r = real(*a, **k)
        r["self_check"]["ok"] = False
        return r
    monkeypatch.setattr(T, "run", bad)
    state = tmp_path / "state"
    rc = T.main(["--url", "http://x", "--collection", "c", "--state", str(state), "--seed", "1",
                 "--windows", "3", "--tail-windows", "3", "--scan-cap", "10000"])
    assert rc == 2 and "result=error" in state.read_text()


def test_tripwire_judges_unacknowledged_alarms_as_new(tmp_path):
    report = {"alarm_below": 0.5, "generations": [
        {"generation": "de.wikipedia.org|reference|-|-|chunk1|aw_b1|2026-08", "est_rows": 2_891_103,
         "sampled": 4500, "decidable": 4410, "share_half_integer": 0.002, "verdict": "ALARM"},
        {"generation": "new.example.de|official|-|-|chunk1|aw_b1|2026-10", "est_rows": 5000,
         "sampled": 1000, "decidable": 1000, "share_half_integer": 0.0, "verdict": "ALARM"},
        {"generation": "ok.example.ch|official|-|-|chunk5|aw_b1|-", "est_rows": 9000,
         "sampled": 1000, "decidable": 1000, "share_half_integer": 0.9, "verdict": "ok"}]}
    ack = tmp_path / "ack.txt"
    ack.write_text("# a comment\nde.wikipedia.org|reference|-|-|chunk1|aw_b1|2026-08 ; rows=2891103 ; share=0.20%"
                   "   # W2 wiki\n")
    out = T.judge(report, T.acknowledged(str(ack)))
    assert out["alarms"] == 2 and out["new_alarms"] == 1
    assert out["acknowledged_rollup"] == {"W2 wiki": {"generations": 1, "est_rows": 2_891_103}}
    assert "NEW: new.example.de|official" in out["summary"]
    assert T.judge(report, [("*", "all")])["new_alarms"] == 0


def test_tripwire_state_file_is_the_monitors_contract(tmp_path):
    path = tmp_path / "state"
    T.write_state(str(path), "alarm", "2 generations", {"new_alarms": 1})
    kv = dict(l.split("=", 1) for l in path.read_text().splitlines())
    assert kv["result"] == "alarm" and kv["summary"] == "2 generations" and kv["new_alarms"] == "1"
    assert kv["checked_epoch"].isdigit() and kv["checked_at"].endswith("Z")


class _FakeQdrant:
    """A twin with two pools: one big no-parts domain and one parts domain; reads are by filter."""
    def __init__(self):
        pts = []
        for i in range(2400):
            r = (WITHOUT if i % 2 == 0 else WITH)[i % 5]
            dom = "no.example.de" if i % 2 == 0 else "yes.example.de"
            pts.append({"id": f"{i:08d}-0000-0000-0000-000000000000", "vector": {"bm25": r["bm25"]},
                        "payload": {"lang": "de", "domain": dom, "source_type": "official",
                                    "provenance": "p", "rule_epochs": {"chunk": 1}, "build_id": "aw_b1",
                                    "last_seen": "2026-09-01"}})
        self.pts = pts

    def count(self, flt):
        return len(self.pts)

    def facet(self, key, flt, limit):
        out = {}
        for p in self.pts:
            out[p["payload"]["domain"]] = out.get(p["payload"]["domain"], 0) + 1
        return [{"value": k, "count": v} for k, v in out.items()]

    def scroll(self, flt, *, offset=None, limit=T.WINDOW, vector=True):
        dom = next((c["match"]["value"] for c in flt["must"] if c["key"] == "domain"), None)
        rows = [p for p in self.pts if dom is None or p["payload"]["domain"] == dom]
        start = 0 if offset is None else next((i for i, p in enumerate(rows) if p["id"] >= offset), len(rows))
        page = rows[start:start + limit]
        nxt = rows[start + limit]["id"] if start + limit < len(rows) else None
        return {"points": page, "next_page_offset": nxt}

    def retrieve(self, ids):
        return [p for p in self.pts if p["id"] in set(ids)]


def test_tripwire_run_alarms_on_the_no_parts_pool_only():
    rep = T.run(_FakeQdrant(), min_generation=1000, sample=1000, alarm_below=0.5, windows=3,
                tail_windows=3, scan_cap=10_000, seed=1)
    by = {g["generation"].split("|")[0]: g for g in rep["generations"]}
    assert by["no.example.de"]["verdict"] == "ALARM" and by["no.example.de"]["share_half_integer"] == 0.0
    assert by["yes.example.de"]["verdict"] == "ok" and by["yes.example.de"]["share_half_integer"] == 1.0
    assert by["no.example.de"]["est_rows"] == 1200


def test_news_stage_writes_the_count_in_warn_and_writes_nothing_in_refuse(tmp_path, monkeypatch):
    """`cytir.news.compose.write_pack` (the content-pack stage), on the real rows without parts."""
    from cytir.core import paths
    from cytir.news import compose as NC
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.setattr(G, "external_predictor", _predictor)
    kw = dict(kept=_many(WITHOUT), tombstones=[], pack_id="gp-news", family="news",
              collection="aw_e8", days=["2026-10-05"])
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "warn")
    paths.reset_settings_cache()
    manifest = NC.write_pack(str(tmp_path / "warn"), **kw)
    assert manifest["german_parts"]["verdict"] == "FAIL" and (tmp_path / "warn" / "READY").exists()
    assert manifest["german_parts"]["rows"] == 40
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "refuse")
    paths.reset_settings_cache()
    with pytest.raises(G.GermanPartsRefused, match="only 0 of 40"):
        NC.write_pack(str(tmp_path / "refuse"), **kw)
    assert not (tmp_path / "refuse").exists()
    paths.reset_settings_cache()


# --------------------------------------------------------------------- review fixes (task 710, #645)
def _boom_check(exc):
    def check(*a, **k):
        raise exc
    return check


@pytest.mark.parametrize("guard_mode,refuses", [("warn", False), ("refuse", True)])
def test_a_guard_that_raises_never_stops_the_caller(monkeypatch, guard_mode, refuses):
    from cytir.core import paths
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", guard_mode)
    paths.reset_settings_cache()
    printed = []
    rep, reason = G.guarded(_boom_check(TypeError("bad")), "packZ", out=printed.append)
    assert rep == {"schema": G.SCHEMA, "result": "error", "error": "TypeError: bad"}
    assert bool(reason) is refuses and (not refuses or reason.startswith("german_parts: guard error"))
    assert len(printed) == 1 and printed[0].startswith("[german_parts] ERROR packZ")
    paths.reset_settings_cache()


@pytest.mark.parametrize("guard_mode", ["warn", "refuse"])
def test_a_sealed_row_with_no_sparse_text_is_not_predicted_and_never_raises(monkeypatch, guard_mode):
    """A real raise: a sealed row with `sparse_text=None` was a TypeError."""
    joined = {r["id"]: ("dense", _text(r), {"lang": "de"}, {}) for r in ROWS}
    joined["none-row"] = ("dense", None, {"lang": "de"}, {})
    rep, reason = G.guarded(G.check_joined, "packN", joined, guard_mode=guard_mode,
                            predictor=_predictor(), writer=_writer(True), out=lambda *_: None)
    assert reason is None and rep["verdict"] == "PASS" and rep["no_text_rows"] == 1


@pytest.mark.parametrize("guard_mode", ["warn", "refuse"])
def test_a_content_point_whose_vector_is_a_list_is_read_as_a_vector_never_raises(guard_mode):
    """A real raise: a point with `vector: [..]` was an AttributeError (`list.get`)."""
    pts = _many(WITH, 40)
    pts[0]["vector"] = [0.1, 0.2, 0.3]
    pts[1]["vector"] = None
    rep, reason = G.guarded(G.check_points, "packL", pts, guard_mode=guard_mode,
                            predictor=_predictor(), out=lambda *_: None)
    assert "error" not in rep and reason is None
    assert rep["undecided"] == 2 and rep["predicted"] == 38 and rep["verdict"] == "PASS"
    assert G.bm25_of({"vector": [1.0]}) == {} and T.row_has_parts({"vector": [1.0]}) is None


@pytest.mark.parametrize("guard_mode,refused", [("warn", False), ("refuse", True)])
def test_the_drain_survives_a_guard_error_and_refuses_only_that_pack_in_refuse(
        tmp_path, monkeypatch, guard_mode, refused):
    """The drain's own call sites: an error in `check_joined` is recorded and the pack after it runs."""
    from cytir.index import promote_queue as PQ
    src = pathlib.Path(PQ.__file__).read_text()
    # both call sites go through `guarded`; no bare check call is left behind
    assert "GPG.check_joined(" not in src and "GPG.check_points(" not in src
    from cytir.core import paths
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", guard_mode)
    paths.reset_settings_cache()
    monkeypatch.setattr(G, "check_joined", _boom_check(RuntimeError("boom")))
    results = []
    for pack in ("p1", "p2"):          # the loop does not stop at p1
        rep, reason = G.guarded(G.check_joined, pack, {}, out=lambda *_: None)
        results.append((pack, rep["result"], bool(reason)))
    assert results == [("p1", "error", refused), ("p2", "error", refused)]
    paths.reset_settings_cache()


def test_a_mode_typo_refuses_the_run_once_at_start(monkeypatch, capsys):
    from cytir.core import paths
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "refues")
    paths.reset_settings_cache()
    assert G.startup_refusal() == 2
    assert "[REFUSED] promote_queue: setting german_parts_guard" in capsys.readouterr().err
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", "Refuse")
    paths.reset_settings_cache()
    assert G.startup_refusal() is None
    paths.reset_settings_cache()
    from cytir.index import promote_queue as PQ
    assert "GPG.startup_refusal()" in pathlib.Path(PQ.__file__).read_text()


def test_the_compose_block_names_the_interpreter_that_measured_it(tmp_path, monkeypatch):
    import sys
    import cytir.index.compose as compose
    monkeypatch.setattr(G, "external_predictor", lambda: None)
    german = [{"lang": "de", "point_id": r["id"]} for r in ROWS]
    monkeypatch.setattr(compose.PSP, "read_records", lambda d: iter(german))
    monkeypatch.setattr(compose.PSP, "project_rows",
                        lambda it, **kw: ({(r := next(it))["point_id"]: ("d", "x", {"lang": "de"}, r)}, 0))
    report, reason = compose._german_parts_report(str(tmp_path), {}, "gp-m")
    assert reason is None
    assert report["measured_by"]["interpreter"] == sys.executable
    assert report["measured_by"]["predictor_found"] is False
    assert "drain-side" in report["measured_by"]["authority"]
    monkeypatch.setattr(G, "external_predictor", _predictor)
    report, _ = compose._german_parts_report(str(tmp_path), {}, "gp-m")
    assert report["measured_by"]["predictor_found"] is True


# --- guard 2: the acknowledged list is bound to measured generations
def _gen(key, rows, share=0.01):
    return {"generation": key, "est_rows": rows, "sampled": 1000, "decidable": 1000,
            "share_half_integer": share, "verdict": "ALARM"}


CC_SEP = "agitano.com|reference|endorsement-discovery|-|chunk1|aw_b1|2026-09"


def test_a_line_acknowledges_its_generation_only_up_to_the_measured_rows(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(f"{CC_SEP} ; rows=1000 ; share=1.00%  # CC-w7 Sept\n")
    a = T.acknowledged(str(ack))
    assert a == [T.Ack(CC_SEP, "CC-w7 Sept", 1000, "1.00%")]
    same = T.judge({"alarm_below": 0.5, "generations": [_gen(CC_SEP, 1150)]}, a)   # +15 %: the estimate
    assert same["new_alarms"] == 0 and same["match_counts"] == {CC_SEP: 1}
    grew = T.judge({"alarm_below": 0.5, "generations": [_gen(CC_SEP, 1300)]}, a)  # a new pack landed
    assert grew["new_alarms"] == 1 and "GREW past the acknowledged rows" in grew["summary"]
    assert grew["grown"][0]["acknowledged_rows"] == 1000


def test_a_new_generation_in_an_acknowledged_family_alarms(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(f"{CC_SEP} ; rows=1000 ; share=1.00%  # CC-w7 Sept\n")
    new_month = "agitano.com|reference|endorsement-discovery|-|chunk1|aw_b1|2026-11"
    other_dom = "other.de|reference|endorsement-discovery|-|chunk1|aw_b1|2026-09"
    out = T.judge({"alarm_below": 0.5, "generations": [_gen(CC_SEP, 1000), _gen(new_month, 1500),
                                                        _gen(other_dom, 1500)]}, T.acknowledged(str(ack)))
    assert out["alarms"] == 3 and out["new_alarms"] == 2 and "NEW: " + new_month in out["summary"]


def test_a_line_that_matches_nothing_is_stale_and_the_state_is_warn(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(f"{CC_SEP} ; rows=1000 ; share=1.00%  # CC-w7 Sept\ngone.de|x|-|-|chunk1|aw_b1|- ; rows=5  # rebuilt\n")
    out = T.judge({"alarm_below": 0.5, "generations": [_gen(CC_SEP, 1000)]}, T.acknowledged(str(ack)))
    assert out["new_alarms"] == 0 and out["stale_lines"] == ["gone.de|x|-|-|chunk1|aw_b1|-"]
    assert out["match_counts"]["gone.de|x|-|-|chunk1|aw_b1|-"] == 0 and "1 STALE" in out["summary"]


FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _shipped_ack():
    p = pathlib.Path(__file__).parent.parent / "deploy" / "systemd" / ("heim" + "dall") / "german-parts-acknowledged.txt"
    return p if p.exists() else None


def _report_generations(rep):
    """A full tripwire report (`generations`) or a tracked excerpt of one (task 732's October list)."""
    return rep.get("generations") or rep["october_endorsement_discovery_generations"]


def test_the_shipped_acknowledged_list_covers_exactly_the_measured_baseline():
    """The CURRENT baseline: the 2026-11-09 tripwire report (sealed, alarm generations only) against the shipped
    list. 7 alarms, every one acknowledged, none stale (task 2621: the October lines were pruned)."""
    base = json.loads((FIXTURES / "german_parts_tripwire_report_2026-11-09.json").read_text())
    ackfile = _shipped_ack()
    if ackfile is None:
        pytest.skip("the acknowledged list is not in this tree (the public cut)")
    ack = T.acknowledged(str(ackfile))
    out = T.judge(base, ack)
    assert out["alarms"] == 7 and out["new_alarms"] == 0 and out["stale_lines"] == []
    assert out["grown"] == [] and out["grown_with_parts"] == []
    assert all(a.rows and a.share and "*" not in a.pattern for a in ack), "no wildcard, no uncapped line"
    assert len({a.pattern for a in ack}) == len(ack), "one measured generation per line"
    assert base["self_check"]["ok"] and base["self_check"]["rows_in_two_generations"] == 0
    # every line is the measured generation: its rows are the report's estimate (within the growth slack)
    alarm = {g["generation"]: g["est_rows"] for g in base["generations"]}
    for a in ack:   # a line's pattern may differ from the key by the ABSENT build_id spelling: use the matcher
        (est,) = [r for k, r in alarm.items() if T.key_matches(k, a.pattern)]
        assert a.rows * (1 + T.ACK_SLACK) >= est
    # a family-wide silence is gone: a new Wikipedia generation, or a new CC-w7 pack, alarms
    for key in ("de.wikipedia.org|reference|-|-|chunk1|aw_b1|2026-11",
                "newdomain.de|reference|endorsement-discovery|-|chunk1|aw_b1|2026-10"):
        assert T.judge({"alarm_below": 0.5, "generations": [_gen(key, 5000)]}, ack)["new_alarms"] == 1


def test_the_shipped_list_has_no_stale_line_against_the_sealed_current_report():
    """What keeps the daily WARN '63 STALE acknowledged line(s)' from coming back unnoticed (task 2621)."""
    base = json.loads((FIXTURES / "german_parts_tripwire_report_2026-11-09.json").read_text())
    ackfile = _shipped_ack()
    if ackfile is None:
        pytest.skip("the acknowledged list is not in this tree (the public cut)")
    out = T.judge(base, T.acknowledged(str(ackfile)))
    assert out["stale_lines"] == [] and "STALE" not in out["summary"]


def test_tripwire_state_is_warn_when_only_stale_lines_remain(tmp_path, monkeypatch):
    class Q(_FakeQdrant):
        pass
    monkeypatch.setattr(T, "Qdrant", lambda url, coll: Q())
    ack = tmp_path / "ack.txt"
    ack.write_text("no.example.de|official|p|-|chunk1|aw_b1|2026-09 ; rows=1200 ; share=0%  # no-parts pool\n"
                   "stale.de|x|-|-|chunk1|aw_b1|- ; rows=5 ; share=0%  # stale\n")
    state = tmp_path / "state"
    rc = T.main(["--url", "http://x", "--collection", "c", "--state", str(state), "--acknowledged", str(ack),
                 "--seed", "1", "--windows", "3", "--tail-windows", "3", "--scan-cap", "10000"])
    kv = dict(l.split("=", 1) for l in state.read_text().splitlines())
    assert rc == 0 and kv["result"] == "warn" and kv["stale_lines"] == "1" and "STALE" in kv["summary"]


def test_a_pack_with_no_german_row_keeps_the_box_independent_block(tmp_path, monkeypatch):
    import cytir.index.compose as compose
    monkeypatch.setattr(compose.PSP, "read_records", lambda d: iter([{"lang": "fr"}]))
    report, _ = compose._german_parts_report(str(tmp_path), {}, "gp-fr")
    assert "measured_by" not in report and report["german_rows"] == 0


# --- task 732: a sighted pack that lands in an acknowledged blind generation is growth WITH parts
# The real case (twin aw_e8, tripwire run 2026-10-08 05:17Z): the line below was measured 2026-10-07
# 06:50Z on rows of the blind drain; tranche 13 (drained 2026-10-07 21:39Z by the sighted drain) added
# 59,751 German rows to the same key, every sampled one with its parts (task 732 doc).
TAIL_OCT = "(tail)|official|endorsement-discovery|-|chunk1|aw_b1|2026-10"
TAIL_OCT_LINE = f"{TAIL_OCT} ; rows=91302 ; share=1.40%  # CC-w7 tranche, blind drain, Oct window\n"


def test_growth_with_parts_in_an_acknowledged_blind_generation_warns_never_alarms(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(TAIL_OCT_LINE)
    out = T.judge({"alarm_below": 0.5, "generations": [_gen(TAIL_OCT, 147_891, share=0.3629)]},
                  T.acknowledged(str(ack)))
    assert out["alarms"] == 1 and out["new_alarms"] == 0 and out["grown"] == []
    g = out["grown_with_parts"][0]
    assert g["acknowledged_rows"] == 91_302 and g["est_rows"] == 147_891
    assert 0.90 < g["growth_share"] < 0.95          # (0.3629 x 147,891 - 0.014 x 91,302) / 56,589 = 0.926
    assert "GREW WITH PARTS" in out["summary"] and "NEW:" not in out["summary"]


def test_growth_without_parts_in_an_acknowledged_generation_still_alarms(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(TAIL_OCT_LINE)
    out = T.judge({"alarm_below": 0.5, "generations": [_gen(TAIL_OCT, 147_891, share=0.014)]},
                  T.acknowledged(str(ack)))
    assert out["new_alarms"] == 1 and out["grown_with_parts"] == []
    assert out["grown"][0]["growth_share"] < 0.05 and "the growth reads" in out["summary"]
    # a mixed growth whose own share is below the floor alarms too (0.15 overall = 37 % on the growth)
    mixed = T.judge({"alarm_below": 0.5, "generations": [_gen(TAIL_OCT, 147_891, share=0.15)]},
                    T.acknowledged(str(ack)))
    assert mixed["new_alarms"] == 1 and mixed["grown"][0]["growth_share"] < 0.5


def test_a_line_without_a_share_cannot_judge_growth_so_growth_alarms(tmp_path):
    ack = tmp_path / "ack.txt"
    ack.write_text(f"{TAIL_OCT} ; rows=91302  # no share measured\n")
    out = T.judge({"alarm_below": 0.5, "generations": [_gen(TAIL_OCT, 147_891, share=0.3629)]},
                  T.acknowledged(str(ack)))
    assert out["new_alarms"] == 1 and out["grown"][0]["growth_share"] is None


def test_parse_share_reads_percent_and_fraction():
    assert T.parse_share("1.40%") == pytest.approx(0.014) and T.parse_share("0.014") == 0.014
    assert T.parse_share(None) is None and T.parse_share("n/a") is None


def test_growth_with_parts_makes_the_state_warn(tmp_path, monkeypatch):
    report = {"schema": T.SCHEMA, "alarm_below": 0.5, "self_check": {"ok": True},
              "generations": [_gen(TAIL_OCT, 147_891, share=0.3629)]}
    monkeypatch.setattr(T, "Qdrant", lambda url, coll: None)
    monkeypatch.setattr(T, "run", lambda q, **kw: dict(report))
    ack = tmp_path / "ack.txt"
    ack.write_text(TAIL_OCT_LINE)
    state = tmp_path / "state"
    rc = T.main(["--url", "http://x", "--collection", "c", "--state", str(state), "--acknowledged", str(ack)])
    kv = dict(l.split("=", 1) for l in state.read_text().splitlines())
    assert rc == 0 and kv["result"] == "warn" and kv["grown_with_parts"] == "1" and kv["new_alarms"] == "0"


def test_a_pass_on_german_rows_prints_one_line_so_the_journal_shows_the_check_ran():
    rep = {"rows": 1_124_382, "german_rows": 80_964, "no_lang_rows": 0, "basis": "text",
           "predictor": "cytir.core.decompound", "sampled": 1000, "predicted": 909, "carried": 909,
           "share": 1.0, "verdict": "PASS", "threshold": G.CARRY_SHARE_MIN}
    reason, lines = G.decide(rep, "cc-authority-frontier-w7-tranche13", guard_mode="warn")
    assert reason is None and lines == [
        "[german_parts] PASS cc-authority-frontier-w7-tranche13: 909 of 909 German rows with predicted parts "
        "carry them (100.0%); 80964 German rows, 1000 sampled, basis text, predictor cytir.core.decompound"]


def test_drain_dropin_sets_refuse():
    """Task 741: the tracked drop-in is the record of the operator step; only the drain unit carries it."""
    root = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "systemd" / ("heim" + "dall")
    if not root.is_dir():
        pytest.skip("the operator unit files are not part of the public cut")
    found = sorted(root.glob("*.service.d/*-german-parts-guard.conf"))
    assert [p.parent.name for p in found] == ["aw-promotion-nightly.service.d"]
    assert "Environment=AW_GERMAN_PARTS_GUARD=refuse" in found[0].read_text().splitlines()


# --- task 742: an acknowledged line can be re-measured
def test_remeasured_field_is_parsed_strictly(tmp_path):
    ack = tmp_path / "ack.txt"
    ok = "a.de|x|-|-|chunk1|aw_b1|- ; rows=10 ; share=36.29% ; remeasured=2026-10-08:docs/x/r.json  # why\n"
    ack.write_text(ok)
    (a,) = T.acknowledged(str(ack))
    assert a.remeasured == ("2026-10-08", "docs/x/r.json") and a.rows == 10 and a.label == "why"
    ack.write_text("a.de|x ; rows=10 ; share=1%  # plain\n")
    assert T.acknowledged(str(ack))[0].remeasured is None
    for bad in ("remeasured=2026-13-01:t/r.json", "remeasured=2026-10-08", "remeasured=2026-10-08:/abs/r.json",
                "remeasured=2026-10-08:../r.json", "remeasured=yesterday:t/r.json"):
        ack.write_text(f"a.de|x ; rows=10 ; share=1% ; {bad}  # w\n")
        with pytest.raises(ValueError):
            T.acknowledged(str(ack))
    for bad in ("a.de|x ; share=1% ; remeasured=2026-10-08:t/r.json", "a.de|x ; rows=10 ; remeasured=2026-10-08:t/r.json"):
        ack.write_text(bad + "  # w\n")
        with pytest.raises(ValueError):
            T.acknowledged(str(ack))


def test_the_real_october_report_replayed_acknowledges_the_tail_and_still_alarms_tuwien():
    root = pathlib.Path(__file__).parent.parent
    art = root / ("ta" + "sks") / "732-artifacts" / "tripwire-20261008-october-cc.json"
    # the list as shipped in October, frozen: the shipped list was pruned in task 2621 and no longer covers it.
    # Comments in the frozen copy are reworded so the public cut carries no tasks/ path (the `tasks/` prefix of a
    # path reads `task-`); the acknowledged lines are the pre-prune list verbatim.
    ackfile = FIXTURES / "german_parts_acknowledged_2026_10_frozen.txt"
    if not art.exists():
        pytest.skip("the October report is not in this tree (the public cut)")
    rep = json.loads(art.read_text())
    gens = list(_report_generations(rep))
    tuwien = "ifs.tuwien.ac.at|reference|endorsement-discovery|-|chunk1|aw_b1|2026-10"
    assert tuwien in {g["generation"] for g in gens}
    out = T.judge({"alarm_below": 0.5, "generations": gens}, T.acknowledged(str(ackfile)))
    assert out["grown"] == [] and out["grown_with_parts"] == [], "the re-measured line is no longer GREW"
    # task 867: ifs.tuwien.ac.at is acknowledged (re-measured 2026-10-12: 84 of 1,537 = 5.47 %, same rows as 10-08)
    assert out["new_alarms"] == 0 and tuwien not in out["summary"].split("NEW:")[-1]


def test_the_frozen_october_list_is_the_shipped_list_plus_the_63_pruned_lines():
    shipped = _shipped_ack()
    if shipped is None:
        pytest.skip("the acknowledged list is not in this tree (the public cut)")

    def lines(path):
        return [l.replace("tasks/", "task-") for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    frozen, now = lines(FIXTURES / "german_parts_acknowledged_2026_10_frozen.txt"), lines(shipped)
    assert len(frozen) - len(now) == 63 and set(now) <= set(frozen) and len(set(frozen)) == len(frozen)
