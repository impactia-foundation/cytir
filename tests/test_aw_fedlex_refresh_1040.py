"""Task 1040: the fedlex refresh door on a REAL act (the sealed replies of one federal act, replayed by
the package's tape transport; no network), through the real `run_family`.

* R1: the dense surface is `compose_index_text` = "<title>. Fedlex. <body>" (the surface the served
  vector was made from, task 526 section 1.1), and the projection leaves it unchanged.
* RULING 55: every window keeps the BARE article url; the place is the payload field `window`; the
  ids stay distinct (tier B: url + digest of the text) and the projection carries `window`.
* Prior records (review 1044 R7d): a prior article absent from the raw XML becomes a removed-source
  pointer (licence public-domain); an article still in the raw XML is never "removed"; the declared
  cap refuses a large removal; an uncapped prior-records run is refused before anything is read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core import projection as PROJ  # noqa: E402
from cytir.core.common import compose_index_text  # noqa: E402
from cytir.sources import run_family as RF  # noqa: E402
from cytir.sources.families import fedlex as FX  # noqa: E402
from cytir.sources.families import fedlex_contract as FC  # noqa: E402

TAPE = HERE / "fixtures" / "fedlex_offline" / "cc-1948-195.tape.json"
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}
TODAY = "2026-09-30"
CONTRACTS = REPO / "data" / "source-family-contracts.v1.json"


def _row() -> dict:
    doc = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    return next(f for f in doc["families"] if f.get("family_id") == "fedlex_exact")


def _tape(*, inflate: bool = False) -> dict:
    d = json.loads(TAPE.read_text(encoding="utf-8"))
    if inflate:     # ~600 unique extra words in the first German article: 3 windows
        for t in d["tape"]:
            if t["url"].endswith("de-xml.xml"):
                filler = " ".join(f"zusatzwort{i}" for i in range(600))
                t["body"] = t["body"].replace("</p></content></paragraph>",
                                              f" {filler}</p></content></paragraph>", 1)
    return d


def _run(monkeypatch, tmp_path, tape: dict, *, options=None, contract=None):
    from cytir.acquire.tape import Tape
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(FX, "DOOR_SPLITTER", FX.DECLARED_SPLITTER)
    contracts_path = None
    if contract is not None:        # a contracts file whose fedlex row is `contract`
        doc = json.loads(CONTRACTS.read_text(encoding="utf-8"))
        doc["families"] = [contract if f.get("family_id") == "fedlex_exact" else f
                           for f in doc["families"]]
        contracts_path = tmp_path.parent / f"contracts-{tmp_path.name}.json"
        contracts_path.write_text(json.dumps(doc), encoding="utf-8")
    t = Tape.load(tape)
    t.unexpected = []
    man = RF.run_family("fedlex_exact", out_dir=tmp_path, today=TODAY, http=t.client(),
                        consent=lambda h, d: {**OK, "host": h},
                        options={"eli": tape["eli"], "langs": tape["langs"], **(options or {})},
                        log=lambda m: None, contracts_path=contracts_path)
    rows = [json.loads(x) for x in (tmp_path / "pack" / "records.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    return man, rows


def test_r1_the_dense_surface_is_title_fedlex_body_on_every_real_row(monkeypatch, tmp_path):
    _man, rows = _run(monkeypatch, tmp_path, _tape())
    assert len(rows) == 63
    for r in rows:
        assert r["dense_text"] == f"{r['title']}. Fedlex. {r['text']}"
        assert r["dense_text"] == compose_index_text({**r, "text": r["text"]})
    first = rows[0]       # pinned on the real first row: law name, SR number, article citation
    assert first["dense_text"].startswith(
        "Schweizerisch‑österreichisches Abkommen über den Grenzverkeh (SR 0.631.256.916.31) — Art. 1"
        ". Fedlex. ")
    # the projection composes again; its dedup guard makes that the identity on this surface
    _pid, dense, _sparse, _payload = PROJ.project_record(first, require_license=True)
    assert dense == first["dense_text"]


def test_r1_the_id_is_still_the_digest_of_the_body_not_of_the_composed_surface(monkeypatch, tmp_path):
    _man, rows = _run(monkeypatch, tmp_path, _tape())
    from cytir.core import ids as ID
    from cytir.index import id_rule as IR
    r = rows[0]
    want, _ = IR.mint_i1({**r, "tier": "B", "content_digest": ID.content_digest(r["text"])})
    assert r["point_id"] == want


def test_ruling_55_every_window_keeps_the_bare_url_and_carries_its_place(monkeypatch, tmp_path):
    _man, rows = _run(monkeypatch, tmp_path, _tape(inflate=True))
    de = [r for r in rows if r["lang"] == "de" and FC.split_anchor(r["url"])[1] == "art_1"]
    n = len(de)
    assert n >= 3
    assert {r["url"] for r in de} == {"https://www.fedlex.admin.ch/eli/cc/1948/195_183_155/de#art_1"}
    assert [r["window"] for r in de] == [{"index": i, "count": n} for i in range(1, n + 1)]
    assert len({r["point_id"] for r in de}) == n
    per_url: dict = {}
    for r in rows:
        per_url.setdefault(r["url"], []).append(r)
    for url, rs in per_url.items():       # a one-row article has no field; windows count themselves
        assert ([r.get("window") for r in rs] == [None] if len(rs) == 1 else
                [r["window"] for r in rs] == [{"index": i, "count": len(rs)} for i in range(1, len(rs) + 1)])
    assert not any("-c" in r["url"].partition("#")[2] for r in rows)
    payload = PROJ.project_record(de[1], require_license=True)[3]
    assert payload["window"] == {"index": 2, "count": n}


def _prior_from(rows, *, extra_anchor=None, drop_domain=False):
    """Prior records shaped like the served payload: the door's own rows, plus one article the
    current XML does not have (`extra_anchor`)."""
    prior = [{"url": r["url"], "domain": r["domain"], "source_type": r["source_type"],
              "title": r["title"], "lang": r["lang"]} for r in rows]
    if extra_anchor:
        base = rows[0]["url"].partition("#")[0]
        prior.append({**prior[0], "url": f"{base}#{extra_anchor}",
                      "title": f"Old law — {extra_anchor}"})
    if drop_domain:
        prior[0]["domain"] = 3
    return prior


def _write_prior(tmp_path, prior) -> str:
    p = tmp_path / "prior.jsonl"
    p.write_text("".join(json.dumps({"id": str(i), "payload": x}) + "\n" for i, x in enumerate(prior)))
    return f"{hashlib.sha256(p.read_bytes()).hexdigest()}={p}"


def _contract(cap):
    c = copy.deepcopy(_row())
    if cap is None:
        c.pop("refresh", None)
    else:
        c["refresh"] = {"removed_share_cap": cap}
    return c


def test_prior_records_write_a_removed_source_pointer_for_an_article_the_xml_no_longer_has(
        monkeypatch, tmp_path):
    _m, rows = _run(monkeypatch, tmp_path / "a", _tape())
    de = [r for r in rows if r["lang"] == "de"]
    opt = _write_prior(tmp_path, _prior_from(de, extra_anchor="art_99"))
    _m, rows2 = _run(monkeypatch, tmp_path / "b", _tape(), options={"prior_records": opt},
                     contract=_contract(0.10))
    removed = [r for r in rows2 if r.get("status") == "removed_from_current_source"]
    assert [r["url"].partition("#")[2] for r in removed] == ["art_99"]
    assert removed[0]["license"] == "public-domain"
    # the one composer: the pointer text already names Fedlex, so the site name is not repeated
    assert removed[0]["dense_text"] == compose_index_text({**removed[0], "site_name": "Fedlex"})
    assert removed[0]["dense_text"].startswith(removed[0]["title"] + ". Fedlex source-state pointer.")
    assert len(rows2) == len(rows) + 1


def test_an_article_still_in_the_raw_xml_is_never_removed(monkeypatch, tmp_path):
    _m, rows = _run(monkeypatch, tmp_path / "a", _tape())
    de = [r for r in rows if r["lang"] == "de"]
    prior = _prior_from(de)
    # a parse loss: the door "forgot" art_1, but the raw XML still holds its eId
    with pytest.raises(RuntimeError, match="parser omission"):
        FX.removed_rows_for_act(prior, de[1:], eli="cc/1948/195_183_155", lang="de",
                                raw_article_anchors={FC.split_anchor(r["url"])[1] for r in de},
                                source_version="x", source_version_url="u", cap=1.0)


def test_the_declared_cap_refuses_a_large_removal(monkeypatch, tmp_path):
    _m, rows = _run(monkeypatch, tmp_path / "a", _tape())
    de = [r for r in rows if r["lang"] == "de"]
    prior = _prior_from(de, extra_anchor="art_99")
    with pytest.raises(RuntimeError, match="above the declared cap"):
        FX.removed_rows_for_act(prior, de, eli="cc/1948/195_183_155", lang="de",
                                raw_article_anchors={FC.split_anchor(r["url"])[1] for r in de},
                                source_version="x", source_version_url="u", cap=0.01)
    with pytest.raises(RuntimeError, match="encoded"):
        FX.removed_rows_for_act(_prior_from(de, drop_domain=True), de, eli="cc/1948/195_183_155",
                                lang="de", raw_article_anchors=set(), source_version="x",
                                source_version_url="u", cap=1.0)


def test_prior_records_without_a_declared_cap_are_refused_before_anything_is_read(
        monkeypatch, tmp_path):
    opt = _write_prior(tmp_path, [])
    with pytest.raises(RF.RunError, match="removed_share_cap"):
        _run(monkeypatch, tmp_path / "b", _tape(), options={"prior_records": opt},
             contract=_contract(None))


def test_the_committed_contract_row_declares_the_cap_and_the_window_field():
    row = _row()
    assert FX.removed_share_cap(row) == 0.10
    assert "window" in row["payload_minimum"]["family_fields"]
    assert "compose_index_text" in row["representation"]["dense_surface"]


def _de_xml(tape: dict) -> dict:
    return next(t for t in tape["tape"] if t["url"].endswith("de-xml.xml"))


def _parse_de(tape: dict) -> list[dict]:
    """`parse_akn` itself (the door's parser) on the German XML of the real tape act."""
    eli = tape["eli"]
    canon = "https://www.fedlex.admin.ch/eli/%s/de" % eli
    return FX.parse_akn(_de_xml(tape)["body"].encode("utf-8"), canon, "de", eli, {},
                        last_seen=TODAY)


def test_r1_two_identical_windows_of_one_article_mint_one_id_and_are_refused():
    """The door's only guard against a silent overwrite (review 1049 REQUIRED 1): the drain re-mints
    I1 and keeps one survivor of an in-pack collision without an error. In the real-tape act the
    first German article gets a 240-word period four times: two of its 240-word windows are equal,
    at one bare url, so they mint one id. The parser must raise and name the act, the language and
    the id. (`run_family` reports a failed unit generically, so the test calls the parser.)"""
    import re
    clean = _parse_de(_tape())
    assert len({r["point_id"] for r in clean}) == len(clean)      # the unmodified act is accepted
    tape = _tape()
    period = " ".join(f"gleichwort{i}" for i in range(240))
    old = "Personen mit Wohnsitz in der Grenzzone gelten als Grenzbewohner im Sinne der Vereinbarung.</p>"
    body = _de_xml(tape)["body"]
    assert old in body
    _de_xml(tape)["body"] = body.replace(old, old + "<p>" + " ".join([period] * 4) + "</p>", 1)
    with pytest.raises(RuntimeError) as err:
        _parse_de(tape)
    msg = str(err.value)
    assert msg.startswith("cc/1948/195_183_155/de: two rows mint id ")      # act and language
    assert re.search(r"two rows mint id [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12} ", msg)
    assert "/de#art_1)" in msg and "refusing a silent overwrite" in msg


def test_the_committed_cap_of_ten_percent_refuses_a_removal_above_it_and_accepts_one_below(
        monkeypatch, tmp_path):
    """Review 1049 RECOMMENDED: the declared 0.10, not a lowered one, through the committed contract."""
    assert FX.removed_share_cap(_row()) == 0.10
    _m, rows = _run(monkeypatch, tmp_path / "a", _tape())
    de = [r for r in rows if r["lang"] == "de"]
    raw = {FC.split_anchor(r["url"])[1] for r in de}
    kw = dict(eli="cc/1948/195_183_155", lang="de", raw_article_anchors=raw,
              source_version="x", source_version_url="u", cap=FX.removed_share_cap(_row()))
    base = _prior_from(de)

    def extra(k):
        return base + [{**base[0], "url": base[0]["url"].partition("#")[0] + f"#art_9{i}"} for i in range(k)]
    over = len(raw) // 10 + 1                                    # strictly more than 10 % of the prior articles
    with pytest.raises(RuntimeError, match="above the declared cap"):
        FX.removed_rows_for_act(extra(over * 2), de, **kw)
    assert len(FX.removed_rows_for_act(extra(1), de, **kw)) == 1     # one article of ~30 is below 10 %
    opt = _write_prior(tmp_path, extra(over * 2))
    with pytest.raises(RF.RunError):        # the same refusal through run_family and the committed row
        _run(monkeypatch, tmp_path / "b", _tape(), options={"prior_records": opt})
