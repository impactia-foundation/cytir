"""`cytir.ops.spam_rows` (task 699): casino / pharma spam chunks are found on REAL rows of the CC
tranche deltas (fixture = rows copied verbatim from t3/t5 chunks on 2026-10-03), and legitimate
chunks that mention the same words are left alone."""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.ops import spam_rows as S  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402


@pytest.fixture
def withheld_data():
    repo_file_or_skip("data/spam-host-lists.v1.json")

ROWS = json.loads((HERE / "fixtures" / "spam_rows" / "real_rows.json").read_text())
LEGIT = [r for r in ROWS if r["label"] == "legit"]
SPAM = [r for r in ROWS if r["label"] == "spam"]


def _flags(rows):
    triples = [(r["point_id"], r["url"], S.row_reason(r["title"], r["text"], r["url"]),
                len(S.lexicon_hits(r["title"] + "\n" + r["text"]))) for r in rows]
    return S.page_flags(triples)


def test_legit_chunks_that_mention_slots_poker_viagra_are_not_flagged():
    assert LEGIT, "fixture lost its legit rows"
    assert _flags(LEGIT) == {}


def test_real_spam_chunks_are_flagged_by_the_row_rule():
    # a chunk alone may need its page (R3/R4); the row rules catch the dense ones
    assert SPAM
    row_level = [r for r in SPAM if S.row_reason(r["title"], r["text"], r["url"])]
    assert len(row_level) >= 5
    assert set(_flags(SPAM)) >= {r["point_id"] for r in row_level}


def test_attachment_page_comment_spam_excludes_the_whole_page():
    rows = [("a", "https://x.example/img-65x65", "R4", 0), ("b", "https://x.example/img-65x65", None, 0),
            ("c", "https://x.example/post", None, 0)]
    assert S.page_flags(rows) == {"a": "R4", "b": "R4"}


def test_doorway_page_is_excluded_whole_when_half_its_chunks_carry_the_lexicon():
    rows = [("a", "u", "R1", 5), ("b", "u", None, 1), ("c", "u", None, 0), ("d", "u", None, 2),
            ("e", "v", None, 1)]
    assert S.page_flags(rows) == {"a": "R1", "b": "R3", "c": "R3", "d": "R3"}


def test_one_mention_is_never_spam():
    assert S.row_reason("An essay", "He lost at the casino once.") is None
    assert S.row_reason("Casino royale review", "A film.") is None  # title alone, no density


def test_bbcode_link_is_comment_spam():
    assert S.row_reason("t", "nice [url=http://spam.example]cheap[/url]") == "R6"


def _write_stage(d, rows):
    d.mkdir(parents=True)
    with gzip.open(d / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_scan_writes_the_exclusion_file_in_the_drain_loader_shape(tmp_path):
    from cytir.index import exclude_file as XF
    keys = ("point_id", "url", "title", "text", "domain")
    _write_stage(tmp_path / "chunks", [{k: r[k] for k in keys} for r in ROWS])
    res = S.scan(tmp_path / "chunks", tmp_path / "out")
    assert res["rows"] == len(ROWS) and res["flagged"] >= 5
    f = tmp_path / "out" / "exclude.jsonl"
    ids = XF.load_exclude_ids(str(f), sha256=XF.sha256_file(f), reason="spam")
    assert len(ids) == res["flagged"]
    assert not ({r["point_id"] for r in LEGIT} & set(ids))
    assert (tmp_path / "out" / "hosts.tsv").read_text().startswith("domain\tspam_rows")


def _queued_pack(q, pid):
    import hashlib
    d = q / pid
    d.mkdir(parents=True)
    body = json.dumps({"pack_id": pid, "rows": 3}, sort_keys=True).encode()
    (d / "manifest.json").write_bytes(body)
    (d / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return d


def test_apply_pins_the_exclusion_file_and_records_the_reason(tmp_path):
    ex = tmp_path / "exclude.jsonl"
    ex.write_text(json.dumps({"point_id": "p1", "reason": "R1"}) + "\n")
    d = _queued_pack(tmp_path / "q", "cc-x-delta")
    out = S.apply(str(tmp_path / "q"), "cc-x-delta", ex, log=lambda m: None)
    m = json.loads((d / "manifest.json").read_text())
    assert out["exclude_ids_added"] == 1
    assert m["exclude_ids_file"] == str(ex.resolve())
    assert "spam" in m["exclude_ids_reason"] and m["resealed"]["exclude_ids_added"] == 1


def test_apply_refuses_a_pack_that_is_not_queued(tmp_path):
    ex = tmp_path / "e.jsonl"
    ex.write_text("{}\n")
    with pytest.raises(SystemExit, match="not READY"):
        S.apply(str(tmp_path / "q"), "nope", ex)


def test_scan_of_zero_rows_refuses(tmp_path):
    (tmp_path / "chunks").mkdir()
    with pytest.raises(SystemExit, match="REFUSED"):
        S.scan(tmp_path / "chunks", tmp_path / "out")


# ---- host hold-out (task 701) ---------------------------------------------------------------
HOST_ROWS = json.loads((HERE.parent / "tests" / "fixtures" / "spam_rows" / "real_host_rows.json").read_text())
LISTS = HERE.parent / "data" / "spam-host-lists.v1.json"


def _scan_host(tmp_path, domain, lists=None):
    rows = [r for r in HOST_ROWS if r["domain"] == domain]
    assert rows, f"fixture lost {domain}"
    d = tmp_path / domain.replace(".", "_")
    _write_stage(d / "chunks", rows)
    res = S.scan(d / "chunks", d / "out", host_lists=lists)
    ex = [json.loads(line) for line in (d / "out" / "exclude.jsonl").read_text().splitlines()]
    return res, ex, rows


def test_job_tag_cloud_host_is_held_out_whole_by_the_measured_rule(tmp_path):
    res, ex, rows = _scan_host(tmp_path, "sunlitcentrekenya.co.ke")  # no list: measured only
    assert res["held_out_hosts"] == 1 and res["held_out_rows"] == len(rows)
    assert {e["reason"] for e in ex} == {S.HOST_REASON} and len(ex) == len(rows)
    held = json.loads((tmp_path / "sunlitcentrekenya_co_ke" / "out" / "host_holdout.json").read_text())
    assert held["sunlitcentrekenya.co.ke"]["basis"] == "measured"


def test_a_hijacked_host_the_row_rule_half_misses_is_held_out_on_the_measured_share():
    # tlb.or.tz in the t5 delta: 2,441 rows, 1,541 with the signal (row rule + pharma phrases), 10 URLs
    held = S.host_holdout({"tlb.or.tz": {"rows": 2441, "signal": 1541, "signal_urls": 10}})
    assert held["tlb.or.tz"]["basis"] == "measured" and held["tlb.or.tz"]["share"] == 0.631


def test_host_thresholds_are_declared_and_all_three_must_hold():
    h = S.host_holdout
    assert h({"a": {"rows": 100, "signal": 50, "signal_urls": 3}})  # on the line: held
    assert not h({"a": {"rows": 100, "signal": 49, "signal_urls": 9}})  # share just under
    assert not h({"a": {"rows": 30, "signal": 19, "signal_urls": 9}})  # too few signal rows
    assert not h({"a": {"rows": 40, "signal": 40, "signal_urls": 2}})  # one or two doorway pages
    assert (S.HOST_SHARE_MIN, S.HOST_MIN_SIGNAL_ROWS, S.HOST_MIN_SIGNAL_URLS) == (0.5, 20, 3)


@pytest.mark.usefixtures("withheld_data")
def test_legit_directory_naming_a_drug_company_is_not_held_out(tmp_path):
    # bcbargaining.ca lists a "Cialis ..." employer on every page: ONE drug name is not spam
    res, ex, rows = _scan_host(tmp_path, "bcbargaining.ca", LISTS)
    assert any("cialis" in r["text"].lower() for r in rows) and ex == [] and res["held_out_hosts"] == 0


@pytest.mark.usefixtures("withheld_data")
def test_cyrillic_legit_host_is_not_held_out(tmp_path):
    res, ex, _ = _scan_host(tmp_path, "kungrad.com", LISTS)
    assert ex == []


@pytest.mark.usefixtures("withheld_data")
def test_gambling_industry_host_is_spared_by_the_declared_exception(tmp_path):
    res0, ex0, rows = _scan_host(tmp_path / "a", "betable.com")  # no list: the row rule flags topic pages
    assert ex0, "the false-positive class vanished from the fixture: re-check the exception"
    res1, ex1, _ = _scan_host(tmp_path / "b", "betable.com", LISTS)
    assert ex1 == [] and res1["held_out_hosts"] == 0


#: Real rows (titles and served snippets) of hosts hand-read on 2026-10-03 whose rows the row rule
#: flagged although the content is not spam: the French retailer "Casino", a news article about gambling
#: policy, an operator's own promotion terms. Their hosts are declared exceptions.
FP_HOSTS = {
    "newsroom.accenture.fr": ("Le groupe Casino noue une alliance avec Accenture et Google Cloud pour accélérer sa stratégie digitale",
                              "Le groupe Casino noue une alliance avec Accenture et Google Cloud pour accélérer sa stratégie digitale. Paris, le 6 Juillet 2021."),
    "africa.businessinsider.com": ("America is one big casino now | Business Insider Africa",
                                   "The ground has shifted on gambling in the US in recent years as it's become easier than ever to try your luck at, well, a casino."),
    "sunbingo.co.uk": ("£50 Bingo Bonus Plus 50 Free Spins",
                       "Free spins can be used at any stage, prior to the expiry date. Your winnings from wagering with free spins will be attributed to Pending balance."),
}


def _fp_rows():
    return [{"point_id": f"p{i}", "url": f"https://{d}/x{i}", "domain": d, "title": t, "text": x}
            for i, (d, (t, x)) in enumerate(FP_HOSTS.items())]


@pytest.mark.usefixtures("withheld_data")
def test_hand_read_false_positive_hosts_are_exempt_from_the_row_rule():
    none = {"hold_out": {}, "exceptions": {}}
    assert S.analyse(_fp_rows(), none)["flags"], "the false-positive class vanished: re-check the exceptions"
    lists = S.load_host_lists(LISTS)
    assert set(FP_HOSTS) <= set(lists["exceptions"])
    assert S.analyse(_fp_rows(), lists)["flags"] == {}


@pytest.mark.usefixtures("withheld_data")
def test_sealed_hold_out_takes_a_host_the_row_rule_never_sees(tmp_path):
    res0, ex0, rows = _scan_host(tmp_path / "a", "escorta.com")
    assert ex0 == []  # an escort portal: no casino lexicon at all
    res1, ex1, _ = _scan_host(tmp_path / "b", "escorta.com", LISTS)
    assert len(ex1) == len(rows) and {e["reason"] for e in ex1} == {S.HOST_REASON}


@pytest.mark.usefixtures("withheld_data")
def test_legit_site_with_a_few_hacked_pages_keeps_the_row_rule_only(tmp_path):
    res, ex, rows = _scan_host(tmp_path, "asharedfuture.ca", LISTS)
    assert res["held_out_hosts"] == 0 and 0 < len(ex) < len(rows)
    assert {e["reason"] for e in ex} <= {"R1", "R2", "R3"}


@pytest.mark.usefixtures("withheld_data")
def test_comment_spam_blog_is_held_out_by_the_sealed_list(tmp_path):
    res, ex, rows = _scan_host(tmp_path, "giffconstable.com", LISTS)
    assert len(ex) == len(rows) and {e["reason"] for e in ex} == {S.HOST_REASON}


@pytest.mark.usefixtures("withheld_data")
def test_the_shipped_host_lists_load_and_every_entry_carries_its_evidence():
    lists = S.load_host_lists(LISTS)
    assert len(lists["hold_out"]) >= 20 and len(lists["exceptions"]) >= 7
    assert not set(lists["hold_out"]) & set(lists["exceptions"])
    for d, ev in {**lists["hold_out"], **lists["exceptions"]}.items():
        assert d == S.norm_domain(d) and len(ev) > 30


def test_host_lists_refuse_a_malformed_file(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(SystemExit, match="REFUSED"):
        S.load_host_lists(p)
    p.write_text(json.dumps({"schema": S.HOST_LISTS_SCHEMA, "hold_out": [{"domain": "x.example"}]}))
    with pytest.raises(SystemExit, match="evidence"):
        S.load_host_lists(p)
    p.write_text(json.dumps({"schema": S.HOST_LISTS_SCHEMA,
                             "hold_out": [{"domain": "x.example", "evidence": "e"}],
                             "exceptions": [{"domain": "www.x.example", "reason": "r"}]}))
    with pytest.raises(SystemExit, match="both"):
        S.load_host_lists(p)


@pytest.mark.usefixtures("withheld_data")
def test_screen_jsonl_drops_hijacked_hosts_and_spam_rows_before_staging(tmp_path):
    doms = ("sunlitcentrekenya.co.ke", "escorta.com", "kungrad.com", "asharedfuture.ca", "betable.com")
    rows = [r for r in HOST_ROWS if r["domain"] in doms]
    src = tmp_path / "passages_ok.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = S.screen_jsonl(src, tmp_path / "kept.jsonl", tmp_path / "rep.jsonl", host_lists=LISTS)
    kept = [json.loads(x) for x in (tmp_path / "kept.jsonl").read_text().splitlines()]
    dropped = [json.loads(x) for x in (tmp_path / "rep.jsonl").read_text().splitlines()]
    assert res["rows_in"] == len(rows) == len(kept) + len(dropped) == res["rows_kept"] + res["rows_screened_out"]
    assert {r["domain"] for r in kept} == {"kungrad.com", "betable.com", "asharedfuture.ca"}
    host_dropped = {d["domain"] for d in dropped if d["reason"] == S.HOST_REASON}
    assert host_dropped == {"sunlitcentrekenya.co.ke", "escorta.com"}
    assert res["held_out_hosts"] == ["escorta.com", "sunlitcentrekenya.co.ke"]
    assert 0 < sum(1 for d in dropped if d["domain"] == "asharedfuture.ca") < sum(
        1 for r in rows if r["domain"] == "asharedfuture.ca")


@pytest.mark.usefixtures("withheld_data")
def test_screen_jsonl_refuses_an_empty_file_and_a_screen_that_drops_everything(tmp_path):
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="0 rows"):
        S.screen_jsonl(empty, tmp_path / "o", tmp_path / "r")
    only = [r for r in HOST_ROWS if r["domain"] == "escorta.com"]
    f = tmp_path / "all.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in only))
    with pytest.raises(SystemExit, match="flags all"):
        S.screen_jsonl(f, tmp_path / "o2", tmp_path / "r2", host_lists=LISTS)
