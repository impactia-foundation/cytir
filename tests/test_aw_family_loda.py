"""Exemplar 4 of the source-family interface (task 683, S7 part 2): DILA LEGI/LODA, ported.

`loda_exact` = `extract_legi_loda_full.py` (task 633) as a `SourceFamily`. The proof is
the same shape as the fedlex one: the family's records are EQUAL to what the legacy driver
(`extract_legi_loda_full.run`, reached through its old path, now a shim) writes on the same
input, and the family's terminal table is the legacy summary's `reject_reasons`, unit by unit.

The input is the eight REAL LEGIARTI articles of `fixtures/legi_loda_full/` (copied verbatim from
the the build box DILA LEGI view, task 633): a LOI, an ORDONNANCE, a DECRET and an ARRETE each on both
sides of the body floor, an ABROGE article and an ABROGE_DIFF article. No network; a dump-route
family contacts no host, so no per-host consent is resolved.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import sys

import pytest


@pytest.fixture(autouse=True)
def _loda_token_counter_stand_in(monkeypatch):
    """`loda_exact` refuses a run without the embed tokenizer (task 706 preflight). The sample
    articles here are all short and never ask for a count, so a stand-in is enough; CI has no
    `transformers`."""
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter", lambda: (lambda text: len(text) // 3))

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE), str(HERE.parent / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from loda_stage_support import FAMILY, FIX, RELEASE, TODAY, _family  # noqa: E402,F401  (task 1280: shared with tests/)
#: pinned from this fixture on 2026-09-30: the pack's records.jsonl (sorted walk) and its table
GOLDEN = {"records": 3, "sha256": "70f597288c89995792403a1ab474973e2c7f45067d71039203c50d2c8025783c",
          "sha256_stamped": "b3c4fc06981bb22b0e822ea10adb64b105a6ff64757650e293fcfa8a7678c80c",
          "excluded": {"below_body_floor:ARRETE": 1, "below_body_floor:DECRET": 2,
                       "below_body_floor:ORDONNANCE": 1, "not_in_force:ABROGE": 1}}


@pytest.fixture
def dump(tmp_path):
    """A tiny LEGI view: `<root>/legi/global/code_et_TNC_en_vigueur/TNC_en_vigueur/<n>/LEGIARTI*.xml`."""
    root = tmp_path / "view"
    branch = root / "legi" / "global" / "code_et_TNC_en_vigueur" / "TNC_en_vigueur" / "JORF"
    branch.mkdir(parents=True)
    for i, f in enumerate(sorted(FIX.glob("*.xml"))):
        shutil.copy(f, branch / f"LEGIARTI9{i:011d}.xml")
    # noise the walk must ignore: a codified (non-TNC) article and a non-article file
    other = root / "legi" / "global" / "code_en_vigueur" / "x"
    other.mkdir(parents=True)
    shutil.copy(FIX / "sample_loi_kept.xml", other / "LEGIARTI9999999999999.xml")
    (branch / "README.txt").write_text("not an article")
    return root


def _legacy(dump, tmp_path):
    from cytir.sources.families import loda as X
    summary = X.run(root=str(dump), out_l1=str(tmp_path / "l1.jsonl"),
                    shard_dir=str(tmp_path / "shards"), rows_per_shard=1000,
                    release=RELEASE, log=lambda m: None)
    rows = []
    for shard in sorted((tmp_path / "shards").glob("shard-*.texts.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return summary, rows


def _records(out):
    return [json.loads(line) for line in
            (out / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def test_family_records_equal_the_legacy_drivers_rows(dump, tmp_path):
    summary, legacy_rows = _legacy(dump, tmp_path / "legacy")
    manifest = _family(dump, tmp_path / "family")
    fam = _records(tmp_path / "family")
    key = lambda r: r["point_id"]          # noqa: E731 -- the legacy walk is os.walk order
    assert len({r.get("family_id") for r in fam}) == 1 and fam[0]["family_id"]   # ADR-124 stamp
    fam = [{k: v for k, v in r.items() if k != "family_id"} for r in fam]
    assert sorted(fam, key=key) == sorted(legacy_rows, key=key)
    assert len(fam) == summary["kept"] == 3
    t = manifest["terminals"]
    assert t["units_in"] == summary["scanned"] == 8       # the codified article and README are not walked
    assert t["units_admitted"] == summary["kept"]
    # the terminal table IS the legacy reject table (the legacy one keys on the reason's prefix)
    by_prefix: dict[str, int] = {}
    for reason, n in t["excluded_reasons"].items():
        by_prefix[reason.split(":", 1)[0]] = by_prefix.get(reason.split(":", 1)[0], 0) + n
    assert by_prefix == summary["reject_reasons"]


def test_a_dump_family_contacts_no_host_and_resolves_no_consent(dump, tmp_path):
    manifest = _family(dump, tmp_path / "o")              # the consent callable fails the test
    assert manifest["consent"]["verdicts"] == []
    assert manifest["route"] == "source_owned_bulk_dump"


def test_golden_and_reasons(dump, tmp_path):
    manifest = _family(dump, tmp_path / "o")
    t = manifest["terminals"]
    assert manifest["records"]["count"] == GOLDEN["records"]
    assert t["excluded_reasons"] == GOLDEN["excluded"]
    # ADR-124: the door stamps `family_id`, so the manifest digest is of the stamped bytes (re-pinned
    # 2026-10-30); the unstamped bytes still hash to the original golden, so nothing else moved.
    assert manifest["records"]["sha256"] == GOLDEN["sha256_stamped"]
    raw = (tmp_path / "o" / "pack" / "records.jsonl").read_bytes().decode("utf-8").splitlines()
    unstamped = "".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "family_id"},
                                   ensure_ascii=False) + "\n" for x in raw)
    assert hashlib.sha256(unstamped.encode("utf-8")).hexdigest() == GOLDEN["sha256"]
    rows = _records(tmp_path / "o")
    assert {r["domain"] for r in rows} == {"legifrance.gouv.fr"}
    assert {r["license"] for r in rows} == {"Etalab-2.0"}
    assert all(r["anchor"].startswith("article:LEGIARTI") for r in rows)
    assert {r["nature"] for r in rows} == {"ARRETE", "DECRET", "LOI"}     # the nature cap is lifted


def test_the_seal_holds_each_articles_exact_bytes_and_the_run_is_reproducible(dump, tmp_path):
    a = _family(dump, tmp_path / "a")
    b = _family(dump, tmp_path / "b")
    assert (tmp_path / "a/pack/records.jsonl").read_bytes() == (tmp_path / "b/pack/records.jsonl").read_bytes()
    assert a["records"]["sha256"] == b["records"]["sha256"]
    index = [json.loads(line) for line in (tmp_path / "a/l0/index.jsonl").read_text().splitlines()]
    assert len(index) == 8
    for row in index:
        data = (tmp_path / "a/l0" / row["sha256"][:2] / row["sha256"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert data == next(dump.rglob(pathlib.Path(row["unit"]).name)).read_bytes()


def test_a_wrong_root_is_an_error_not_an_empty_source(tmp_path):
    from cytir.sources import run_family
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="wrong root"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o", today=TODAY,
                              consent=lambda h, d: {}, options={"root": str(empty), "release": RELEASE},
                              log=lambda m: None)
    with pytest.raises(ValueError, match="needs --opt root"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o2", today=TODAY,
                              consent=lambda h, d: {}, options={}, log=lambda m: None)


def test_the_family_is_declared_and_chunks_natively():
    from cytir.sources import registry
    adapter, row, spec = registry.admit(FAMILY)
    assert row["consent"]["basis"] == "official-no-copyright"      # declared by the contract
    assert row["licence"]["instrument"] == "Etalab-2.0"
    assert spec.declared == "native-chunked"


def test_a_mutated_quality_floor_fails_the_equivalence(dump, tmp_path, monkeypatch):
    """Mutation check: the family shares the legacy evaluator, so a change there moves BOTH
    sides equally; what must NOT be shared is the terminal accounting. Raising the statutory
    floor drops the kept LOI, and the family's table says so."""
    from cytir.sources.families import loda as X
    monkeypatch.setattr(X, "MIN_BODY_STATUTORY", 10 ** 6)
    manifest = _family(dump, tmp_path / "o")
    assert manifest["terminals"]["units_admitted"] < GOLDEN["records"]
    assert manifest["terminals"]["excluded_reasons"].get("below_body_floor:LOI", 0) >= 1


def test_consent_hosts_file_units_have_none_and_a_hostless_http_url_is_still_held():
    """Framework fix found by this port: a `file:` unit (a local dump member) contacts no host;
    it was quarantined as `consent_unresolved:` (empty host). A URL with a scheme that does
    contact a host but names none must STILL be held, never admitted."""
    from cytir.sources import run_family
    from cytir.sources.family import RunContext, Unit
    f = Unit(key="a", url="file:///x/a.xml")
    h = Unit(key="b", url="https://")
    g = Unit(key="c", url="https://x.example/p", consent_hosts=("y.example",))
    assert run_family._consent_hosts(f) == ()
    assert run_family._consent_hosts(h) == ("",)
    assert run_family._consent_hosts(g) == ("y.example",)
    ctx = RunContext(out_dir=pathlib.Path("."), today=TODAY, consent=lambda host, d: {"bucket": "ok"})
    _verdicts, blocked = run_family.resolve_unit_consent([f, h], ctx, direct=False)
    assert "a" not in blocked and blocked["b"][1] == "consent_unresolved:"
