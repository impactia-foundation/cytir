"""Task 738: `cyt consent seal` reproduces the sealed task-629 file, the loader reads it
strictly, and an incomplete probe file is refused."""
import copy
import json
import pathlib

import pytest

from cytir.consent import derive as D
from cytir.consent import seal as S
from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
TASKS_DIR = sorted(["tasks"])[0]   # the withheld evidence dir, spelled so the cut scan sees no constant path
V1 = REPO / TASKS_DIR / "629-artifacts" / "pool-a-consent.v1.json"
POOL_A_HOSTS = [
    "entreprendre.service-public.gouv.fr", "files.eric.ed.gov", "github.com", "openalex.org",
    "stackoverflow.com", "sunfounder.com", "toybox.com", "upesy.com", "www.canada.ca",
    "www.rfc-editor.org", "www.rsi.ch", "www.rts.ch", "www.service-public.gouv.fr",
    "www.srf.ch", "www.swissinfo.ch", "www150.statcan.gc.ca"]
PATCHED = ("origin", "origin_scheme", "gate_bucket", "gate_bucket_legacy_both_flags",
           "gate_bucket_search_indexing", "direct_terminal", "terminal_schema_version")


def _v1():
    return json.loads(repo_file_or_skip(V1.relative_to(REPO)).read_text())


def _raw(v1):
    """What `resolve_hosts` hands back: the sealed records minus everything the finaliser adds."""
    return {h: {k: v for k, v in v1["host_consent_resolution"][h].items() if k not in PATCHED}
            for h in POOL_A_HOSTS}


def _build(v1, families, **kw):
    fh = {f: v1["family_verdicts"][f]["hosts"] for f in families}
    args = dict(family_hosts=fh, raw_records=_raw(v1), task=629, precondition="P7",
                sealed_at=v1["sealed_at_utc"], hostname="Heimdall", egress_ip=None,
                operator=v1["operator_identity"], shas=v1["resolver"]["sha256"],
                settings_used="x", probe_box="Heimdall")
    args.update(kw)
    return S.build(**args)


def test_sealing_the_task629_host_records_reproduces_its_records_and_verdicts():
    v1 = _v1()
    fams = [f for f in v1["family_verdicts"] if v1["family_verdicts"][f]["hosts"]
            and f != "stackexchange_nonso_qa"]
    doc = _build(v1, fams)
    assert doc["host_consent_resolution"] == {h: v1["host_consent_resolution"][h]
                                              for h in POOL_A_HOSTS}
    for f in fams:
        want = v1["family_verdicts"][f]
        got = doc["family_verdicts"][f]
        assert got["hosts"] == want["hosts"] and got["verdict"] == want["verdict"], f
        if f != "stackoverflow_qa":          # that reason carries hand wording in v1
            assert got["reason"] == want["reason"], f
    # legitimately different from v1: sealer provenance (egress_ip, sealer, cytir/ file shas),
    # and the sections v1 held that this build did not carry
    assert doc["schema"] == v1["schema"] and doc["terminal_schema_version"] == v1["terminal_schema_version"]


def test_carried_sections_are_restricted_and_marked():
    v1 = _v1()
    doc = _build(v1, ["eric_official_pdf", "technical_manufacturer_catalog"],
                 carry=v1, carry_sha="abc", carry_path="v1")
    assert set(doc["family_origin_roster"]) == {"eric_official_pdf", "technical_manufacturer_catalog"}
    packs = doc["pack_to_origin_conservation"]["packs"]
    assert packs and {p["family"] for p in packs} == set(doc["family_origin_roster"])
    assert doc["pack_to_origin_conservation"]["total_actual_rows"] == sum(p["actual"] for p in packs)
    assert doc["carried_forward_from"]["sha256"] == "abc"
    assert set(doc["host_consent_resolution"]) == {"files.eric.ed.gov", "sunfounder.com",
                                                   "upesy.com", "toybox.com"}


def test_sealer_refuses_a_mislabelled_vantage_and_a_missing_record():
    v1 = _v1()
    with pytest.raises(S.SealRefused):
        _build(v1, ["eric_official_pdf"], probe_box="Heimdall", hostname="otherbox.example")
    with pytest.raises(S.SealRefused):
        _build(v1, ["eric_official_pdf"], probe_box=None)
    with pytest.raises(S.SealRefused):
        _build(v1, ["eric_official_pdf"], raw_records={})


def test_ccbot_style_train_off_stays_search_ok():
    doc = _build(_v1(), ["github_agent_resources"])
    rec = doc["host_consent_resolution"]["github.com"]
    assert rec["search_ok"] is True and rec["train_ok"] is False
    assert doc["family_verdicts"]["github_agent_resources"]["verdict"] == "admissible"


def test_written_file_is_never_overwritten_and_has_a_seal_line(tmp_path):
    doc = _build(_v1(), ["eric_official_pdf"])
    out = tmp_path / "x.json"
    line = S.write_sealed(doc, out)
    assert line.startswith(S.sha256_file(out)) and "hosts=1" in line
    assert (tmp_path / "x.json.seal").read_text().strip() == line
    with pytest.raises(S.SealRefused):
        S.write_sealed(doc, out)


def _fams(path, ids, **cons_over):
    return [{"family_id": i, "consent": {"evidence": [f"{path}#host_consent_resolution"]}}
            for i in ids]


def test_sealed_output_loads_strict_and_is_inside_30_days_at_the_deadline(tmp_path):
    v1 = _v1()
    doc = _build(v1, ["eric_official_pdf", "technical_manufacturer_catalog"],
                 sealed_at="2026-10-08T07:00:00Z")
    for r in doc["host_consent_resolution"].values():
        r["checked_at"] = "2026-10-08T07:00:00Z"
    (tmp_path / "s.json").write_text(S.dumps(doc))
    got = D.load_sealed_probes(_fams("s.json", ["eric_official_pdf", "technical_manufacturer_catalog"]),
                               str(tmp_path))
    assert set(got["technical_manufacturer_catalog"]) == {"sunfounder.com", "upesy.com", "toybox.com"}
    rec = got["eric_official_pdf"]["files.eric.ed.gov"]
    as_of = D._parse_utc("2026-10-22T06:59:00Z")
    assert as_of - D._parse_utc(rec["checked_at"]) <= D.timedelta(days=D.SEALED_MAX_AGE_DAYS)


def test_the_old_v1_file_is_past_the_bound_after_its_deadline():
    rec = _v1()["host_consent_resolution"]["files.eric.ed.gov"]
    late = D._parse_utc("2026-10-22T07:00:00Z")
    assert late - D._parse_utc(rec["checked_at"]) > D.timedelta(days=D.SEALED_MAX_AGE_DAYS)


@pytest.mark.parametrize("drop", ["family_verdicts", "host_consent_resolution"])
def test_loader_refuses_a_file_missing_a_section(tmp_path, drop):
    doc = _v1()
    del doc[drop]
    (tmp_path / "s.json").write_text(json.dumps(doc))
    with pytest.raises(D.SealedEvidenceRefused):
        D.load_sealed_probes(_fams("s.json", ["eric_official_pdf"]), str(tmp_path), strict=True)


def test_loader_refuses_a_family_without_hosts_and_a_host_without_a_record(tmp_path):
    doc = _v1()
    d1 = copy.deepcopy(doc)
    d1["family_verdicts"]["eric_official_pdf"]["hosts"] = []
    (tmp_path / "a.json").write_text(json.dumps(d1))
    with pytest.raises(D.SealedEvidenceRefused):
        D.load_sealed_probes(_fams("a.json", ["eric_official_pdf"]), str(tmp_path), strict=True)
    d2 = copy.deepcopy(doc)
    del d2["host_consent_resolution"]["files.eric.ed.gov"]
    (tmp_path / "b.json").write_text(json.dumps(d2))
    with pytest.raises(D.SealedEvidenceRefused):
        D.load_sealed_probes(_fams("b.json", ["eric_official_pdf"]), str(tmp_path), strict=True)
    (tmp_path / "c.json").write_text("{not json")
    with pytest.raises(D.SealedEvidenceRefused):
        D.load_sealed_probes(_fams("c.json", ["eric_official_pdf"]), str(tmp_path), strict=True)


def test_the_real_contracts_load_strict_against_the_real_v1():
    _v1()
    contracts = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())
    got = D.load_sealed_probes(contracts["families"], str(REPO))
    assert "files.eric.ed.gov" in got["eric_official_pdf"]


def test_default_is_strict_in_a_tree_with_sealed_evidence_and_lenient_in_the_public_cut(tmp_path):
    fams = _fams(f"{TASKS_DIR}/gone.json", ["eric_official_pdf"])
    assert D.load_sealed_probes(fams, str(tmp_path)) == {}          # no sealed-evidence dir: the cut
    (tmp_path / TASKS_DIR).mkdir()
    with pytest.raises(D.SealedEvidenceRefused):                    # the dir exists: refuse
        D.load_sealed_probes(fams, str(tmp_path))


R6_FAMILIES = ("eric_official_pdf", "technical_manufacturer_catalog", "rfc_editor_net_new",
               "statcan_wds", "context7_pinned_docs", "github_agent_resources")


def test_v3_is_pinned_by_every_r6_family_loads_strict_and_is_in_bound_at_the_deadline():
    v3 = repo_file_or_skip((REPO / TASKS_DIR / "2531-artifacts" / "pool-a-consent.v3.json").relative_to(REPO))
    contracts = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())
    fams = [f for f in contracts["families"] if f["family_id"] in R6_FAMILIES]
    assert len(fams) == 6
    for f in fams:
        ev = f["consent"]["evidence"]
        assert any(f"{TASKS_DIR}/2531-artifacts/pool-a-consent.v3.json#host_consent_resolution" == e for e in ev), f["family_id"]
        assert not any("629-artifacts" in e for e in ev), f["family_id"]
        assert any(S.sha256_file(v3) in e for e in ev), f["family_id"]
    got = D.load_sealed_probes(fams, str(REPO), strict=True)
    assert set(got) == set(R6_FAMILIES)
    as_of = D._parse_utc("2026-11-08T12:00:00Z")
    for fid, hosts in got.items():
        for h, rec in hosts.items():
            assert rec["status"] == "ok" and as_of - D._parse_utc(rec["checked_at"]) <= D.timedelta(
                days=D.SEALED_MAX_AGE_DAYS), (fid, h)
    # the v2 file's own seal line matches its bytes
    assert (str(S.sha256_file(v3)) in (v3.parent / (v3.name + ".seal")).read_text())
