"""The exact-URL preserved acquirers on the family Protocol: four rows, two drivers (task 1410, S7).

Rows: `ca_canadaca_captured_exact` and `cochrane_preserved_exact` (Common Crawl range-GET driver),
`ca_canadaca_direct_exact` and `cochrane_direct_exact` (consent-gated direct GET driver).

No network, no live fixture: the CC index resolver, the WARC fetcher/parser, the origin fetcher and the
consent getter are injected, as the drivers' own tests do. Pins: all four adapters are registered and
conformant and say they are PARTIAL; `discover`/`acquire` read and validate the sealed run (a missing
or tampered receipt is refused, a CC miss and a recorded decline are EXCLUDED terminals, an unreadable
policy is not); consent and licence come from the contract row as declared; and the moved drivers
seal the SAME BYTES as the pre-move drivers.

PARITY STANDARD (tasks 1152, 1405): `PINNED_CC` and `PINNED_DIRECT` hold the sha256 of every file the
pre-move drivers sealed for these fixtures, taken from the drivers at origin/main commit 9b23bc508
(`git show 9b23bc508:<driver path>`), run twice to prove them deterministic and once from a second
directory to prove them path-independent. A change to the moved code that alters one byte of a sealed
receipt fails here.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib

import pytest

from cytir.acquire import stage_receipt as SS
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.exact_url import cc_preserved as CCP
from cytir.sources.exact_url import direct_preserved as DIRECT
from cytir.sources.families import exact_url_preserved as FAM
from cytir.sources.family import RunContext, SourceFamily

CC_IDS = ("ca_canadaca_captured_exact", "cochrane_preserved_exact")
DIRECT_IDS = ("ca_canadaca_direct_exact", "cochrane_direct_exact")
ALL_IDS = CC_IDS + DIRECT_IDS

A_URL = "https://www.canada.ca/en/a.html"        # CC capture
B_URL = "https://www.cochrane.org/news/b"        # CC capture
C_URL = "https://www.canada.ca/en/c.html"        # CC miss
BODY_A = b"<html><main>alpha page</main></html>"
BODY_B = b"<html><main>beta page</main></html>"

PINNED_CC: dict[str, str] = {
    "RUN.json": "7587b19b31c5c0ca567855185f924fce11edde277c583461499ea6b33509f4a9",
    "original/decoded-html/4b/4b5cbef898eab6534ee5934ef16c6a647fb61acb108afcb4de3048808391c692.html.gz": "e428d9f01fe3aefc9db2049f53a48aa7daa79a4c6649eb385fde3a15d86c8f2b",
    "original/decoded-html/86/86c785b814f77f476c7c14638e172d8aa47a7f6a00ca2c0563188775c5499d40.html.gz": "e4cabfc9d8408d78a8211ea1e5f30721db1280dd9930851fcbfb19604e8e93b9",
    "original/http-body/4b/4b5cbef898eab6534ee5934ef16c6a647fb61acb108afcb4de3048808391c692.bin.gz": "e428d9f01fe3aefc9db2049f53a48aa7daa79a4c6649eb385fde3a15d86c8f2b",
    "original/http-body/86/86c785b814f77f476c7c14638e172d8aa47a7f6a00ca2c0563188775c5499d40.bin.gz": "e4cabfc9d8408d78a8211ea1e5f30721db1280dd9930851fcbfb19604e8e93b9",
    "original/http-headers/28/280aecdb4af64913558bc5ef92ad523d52fca07f98a48a4ef7e799c8c38dd6c3.bin.gz": "ec891510aa4fcc51353efc6897460b7aea4ea1a53fca55f168efa89f3737b739",
    "original/transport/4a/4a79b6bf73d420e9b3d55624e70f522e1a8483577b5f660272ef1277c8e520ed.json.gz": "ebf27b5a8b6f25a6d79bd406fc4f3ef530236848c860dc6557424fb00cfb668e",
    "original/transport/9e/9e99a514809da6591922d49a922f0eea31c1d6d802abdcd066d01262f281aa2e.json.gz": "4352f41a638cd1c6a22849bf2c3248adfe5223acf015392fce00059590f43a3e",
    "original/warc/b9/b94e91d8bf469a49061df0c06fe3e04c2990fb7d04f17f758d23b2902f364cb4.warc.gz.gz": "9f8d03662d880930d5fd13fb41ea9f496334b43c20312856af5e11ec5c35fe3e",
    "original/warc/df/df5d1360d4559f7f0ce22873348e29702b612fcc5807244423fdffa2d71ec2d9.warc.gz.gz": "2950b70aad7af5b9ea7a99776af71f855f08f7f012a440cb6daf46f15d8da5c9",
    "plan/RECEIPT.json": "10166dddc9e29f8f4edd89bcfd0f0045728aa34d186a8774c30ea3bf789bf5a1",
    "plan/captures.jsonl": "eed1dfd216589c8bd06efe73c680bc83f4fbfec6afaff4e34d782be203a5a6c4",
    "plan/identity.json": "610a83a1ee9b1f577d471ae9fcef7e810421dca6cb18097b20875da30ebcca9e",
    "receipts/7aaf5d1414883de0a26416843f87fca831185499d8ea0efa30d0d8154347a889.json": "0df3bec1fc47a6f2361f1ee9c9cd438be7c40406e9375ba4f7e41175341263cb",
    "receipts/88594e6962905fc9482ecfa42b8e3921199b34dfe5a7234bb6e9fafcaf33c5dd.json": "4fbc251c17c15da395fe2549b3f12dd73448ae87ffaa1e51f727b5a9c8e3657b",
    "receipts/be3b60d01101ef44e97035c8c33b17bad4436eeefc81962dcb98e128afd169b5.json": "5e9ddcbcb482307fea3c5901293b3bb4b16b4ac811010a8ff2a5c230c80b57a1",
    "stages/original/receipt.json": "125d479a2afa43b05a3ab56b7d83711f65440289246c603ad158a5856cd26cc7"
}
PINNED_DIRECT: dict[str, str] = {
    "RUN.json": "359135b9003f5ffd84b2a812796eec497c90214c1809ab0202dc3b8c24a8c1f0",
    "consent/0c/0c550f5226f263bb7ff3913fcc2d421713ceebb8f8f6a549e99ec04cc1f77375.json.gz": "6677909c37001e57b91a3bcd0b4c6a12595a5e664925cb8bbffefdc709f9ffd6",
    "consent/5d/5d301bd799ce62b9fd368035e703ac3098eacb39e3ba047ea2d03005361ed0f5.json.gz": "04419068e7a3749c86d3942900a004ec1d5d5f55e1e8b301cf4bf7e1a7888389",
    "original/http-body/0f/0f4636c78f65d3639ece5a064b5ae753e3408614a14fb18ab4d7540d2c248543.bin.gz": "31fca02454bf886110284d7b6094667474b81a1ceb148e569a2ad3ca13e6924a",
    "original/http-body/ae/aea5260dd5fcc879a4b396f912c64b882149e1ae0f83b13e8a253f75fd823424.bin.gz": "581a43727c2a3aafdc8616964e5e2e56b597318985e794044bb1b31a2a63f7af",
    "original/response-headers/75/754d559efaabefde1bd2a36971890bba9a8dbae3b4b37aab28bbb74a44f94737.json.gz": "656d3cfd58d6acd406404532fce600e7d46f8cbd71b0453ae0bc9f3bdae11da3",
    "original/response-headers/9f/9f3d5e448a39782b82d2fa51e2f7da5394bb9f7ad0e78fa6fa74ba068c517b74.json.gz": "97a8286a88342a6f159af2009136fcdb56ac980d577d4c7813bea913e9297fa0",
    "original/transport/b1/b16490d65ee0a28efc368cc711a0abfb10caad73c170b2b8cc1f9cdffc421354.json.gz": "e6a96122005ca6f88a439e2b98015d7459a91fef7adc4e0d981e9dcdb228902e",
    "original/transport/f0/f0b73a7169ebb275711e0a164f4e59eda11236a774ec06c71605cfd15f4c652e.json.gz": "7ebf366a592e7e919bcb6e7e8b61090f740284c926ad482eff68917bb897c5f5",
    "plan/direct-cc-misses.jsonl": "03c8d375f26624e08282c15346e181bf81cfded56a87af3000c50eaab553abe8",
    "quarantine/records.jsonl": "597e656237491e475f64fef57772c689b307f8856f85a57adb8dfbcc1db8ec75",
    "receipts/2627d493c981d56c1d2ccae6385eb6e6bad0da4f3019a02056d967f42fb0831a.json": "b4fd3dfb90ee1312d73f29cd07884f6ac5e087c22e16c296251288f73d33ea2c",
    "receipts/7aaf5d1414883de0a26416843f87fca831185499d8ea0efa30d0d8154347a889.json": "0b582711785c4051d2f69bd2f52f1ec69e2a15e9f5a909e083910985fcd4f758",
    "receipts/88594e6962905fc9482ecfa42b8e3921199b34dfe5a7234bb6e9fafcaf33c5dd.json": "c3ed33552b732a4cca6a9d1c81c7b020abe2753d7f9ce58a7fb1ce899aadbe0e",
    "stages/original/receipt.json": "0bd8fe69f9b0a614325c13c3e486f02b463858b443165c24f2ae280951d1cd02",
    "stages/quarantine/receipt.json": "4fc52024e8a32b86fad9d49b88e6758934a4671593d55ea4b982a159185651b0"
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _b32(body: bytes) -> str:
    return base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip("=")


def run_cc(cc_mod, base: pathlib.Path) -> pathlib.Path:
    """Plan + acquire three frontier URLs (two captures, one miss) with injected index and fetcher.

    The plan receipt records the frontier and plan paths as given, so the run uses paths RELATIVE to
    `base` (the working directory for the call): the sealed bytes then do not depend on where the
    test directory is."""
    frontier = base / "frontier.jsonl"
    frontier.write_text("".join(json.dumps({"ordinal": i, "url": u}) + "\n"
                                for i, u in enumerate((A_URL, B_URL, C_URL))))
    receipt = base / "frontier-receipt.json"
    receipt.write_text(json.dumps({"schema": "aw.exact-url-frontier-receipt.v1", "result": {
        "frontier": {"sha256": _sha(frontier.read_bytes()), "urls": 3}}}))
    captures = {u: {"url": u, "fetch_time": "2026-01-01", "warc_filename": f"w{i}", "offset": i,
                    "length": 2, "digest": _b32(b)}
                for i, (u, b) in enumerate(((A_URL, BODY_A), (B_URL, BODY_B)), 1)}
    bodies = {b"warc-1": BODY_A, b"warc-2": BODY_B}

    def fetcher(capture):
        return b"warc-%d" % capture["offset"], {"schema": "aw.common-crawl-range-transport.v1",
                                                 "http_status": 206}

    here = pathlib.Path.cwd()
    os.chdir(base)
    try:
        rel = pathlib.Path
        cc_mod.plan(frontier_path=rel("frontier.jsonl"), frontier_receipt_path=rel("frontier-receipt.json"),
                    root=rel("cc"), crawl="CC-MAIN-TEST", resolver=lambda urls: captures)
        cc_mod.acquire(root=rel("cc"), artifact_root=rel("."), rate=4, fetcher=fetcher,
                       parser=lambda raw: (b"HTTP/1.1 200 OK\r\n", bodies[raw], bodies[raw].decode()))
    finally:
        os.chdir(here)
    return base / "cc"


def _open_consent(host: str) -> dict:
    if host.endswith("cochrane.org"):
        return {"host": host, "status": "ok", "search_ok": False, "train_ok": False, "mode": "direct",
                "signals": {"robots": {"wildcard_root": False}}, "checked_at": "2026-08-16T00:00:00Z"}
    return {"host": host, "status": "ok", "search_ok": True, "train_ok": False, "mode": "direct",
            "checked_at": "2026-08-16T00:00:00Z", "_robots_body": "",
            "signals": {"robots": {"fetched": True, "http_status": 404}}}


def cc_miss_fixture(base: pathlib.Path) -> pathlib.Path:
    """A terminal CC run with three misses (built by hand, independent of the CC driver)."""
    root = base / "cc"
    (root / "receipts").mkdir(parents=True)
    urls = (C_URL, B_URL, "https://www.canada.ca/en/d.html")
    paths = []
    for url in urls:
        path = root / "receipts" / (_sha(url.encode()) + ".json")
        path.write_text(json.dumps({"schema": "aw.exact-url-cc-capture-receipt.v1", "version": "v",
                                    "status": "cc_miss", "source_url": url, "artifacts": {}}))
        paths.append(path)
    plan = root / "plan/captures.jsonl"
    plan.parent.mkdir()
    plan.write_text("".join(json.dumps({"url": u, "state": "MISS"}) + "\n" for u in urls))
    (root / "plan/RECEIPT.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-plan-receipt.v1", "status": "complete",
        "plan": {"sha256": _sha(plan.read_bytes()), "misses": len(urls)}}))
    stage = SS.seal(artifact_root=base, artifact_paths=paths,
                    receipt_path=root / "stages/original/receipt.json", records=0, fmt="fixture",
                    logical_sha256="a" * 64, producer_id="fixture", producer_version="v1")
    (root / "RUN.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
        "counts": {"cc_misses": len(urls), "cc_captures": 0}, "stages": {"original": stage}}))
    return root


def _fetch(url: str):
    if url.endswith("d.html"):
        return (b"\x89PNG", {"http_status": 200, "final_url": url, "requested_url": url},
                [["content-type", "image/png"]])
    return (b"<html><main>origin copy</main></html>",
            {"http_status": 200, "final_url": url, "requested_url": url,
             "retrieved_at": "2026-08-16T00:00:01Z"},
            [["content-type", "text/html; charset=utf-8"]])


def run_direct(direct_mod, base: pathlib.Path) -> pathlib.Path:
    """Direct-fetch three CC misses: one complete, one consent decline, one non-textual body."""
    cc = cc_miss_fixture(base)
    root = base / "direct"
    direct_mod.acquire(cc_root=cc, root=root, artifact_root=base, rate=2, fetcher=_fetch,
                       consent_getter=_open_consent)
    return root


def _row(fid: str) -> dict:
    return contract_mod.by_id(contract_mod.load())[fid]


def _ctx(fid: str, tmp_path, contract=None, **opts) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True, options=opts,
                      contract=contract or _row(fid), log=lambda m: None)


# ---- parity: the moved drivers seal the bytes the pre-move drivers sealed -----------------------

def test_cc_driver_seals_the_pinned_bytes(tmp_path):
    root = run_cc(CCP, tmp_path)
    assert tree_digest(root) == PINNED_CC


def test_direct_driver_seals_the_pinned_bytes(tmp_path):
    root = run_direct(DIRECT, tmp_path)
    assert tree_digest(root) == PINNED_DIRECT


def test_frozen_producer_labels_are_unchanged():
    assert CCP.PRODUCER.endswith("/acquire_exact_url_cc_preserved.py")
    assert DIRECT.PRODUCER.endswith("/acquire_exact_url_direct_preserved.py")
    assert DIRECT.VERSION == "exact-url-direct-preserved.v4-consent-gated"


# ---- the adapters --------------------------------------------------------------------------------

@pytest.mark.parametrize("fid", ALL_IDS)
def test_registered_conformant_and_partial(fid):
    assert fid in registry.registered()
    adapter, row, spec = registry.admit(fid)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == fid
    assert getattr(adapter, "PARTIAL", ()) == ()
    declared = _row(fid)["chunk_policy"]["policy_id"]
    assert spec.declared == ("dispatcher" if declared == "unknown" else declared)


@pytest.mark.parametrize("fid,route,direct", [
    ("ca_canadaca_captured_exact", "common_crawl", False),
    ("cochrane_preserved_exact", "common_crawl", False),
    ("ca_canadaca_direct_exact", "direct_origin_crawl", True),
    ("cochrane_direct_exact", "direct_origin_crawl", True)])
def test_routes(fid, route, direct):
    adapter, _row_, _spec = registry.admit(fid)
    assert adapter.route == route and adapter.direct_crawl is direct


@pytest.mark.parametrize("fid", CC_IDS)
def test_cc_discover_and_acquire_read_the_sealed_run(fid, tmp_path):
    root = run_cc(CCP, tmp_path)
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path, root=str(root), artifact_root=str(tmp_path))
    units = list(fam.discover(ctx))
    assert [u.url for u in units] == [A_URL, B_URL, C_URL]
    assert ctx.notes["plan_captures"] == 2 and ctx.notes["plan_misses"] == 1
    (cap,) = fam.acquire(units[0], ctx)
    assert cap.body == BODY_A
    assert cap.meta["fetched_at"] == "2026-01-01" and cap.meta["artifact"] == "decoded_html"
    with pytest.raises(FAM.UnitExcluded) as miss:
        fam.acquire(units[2], ctx)
    assert miss.value.reason == "cc_miss"


@pytest.mark.parametrize("fid", CC_IDS)
def test_cc_acquire_refuses_a_missing_or_tampered_receipt(fid, tmp_path):
    root = run_cc(CCP, tmp_path)
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path, root=str(root), artifact_root=str(tmp_path))
    units = list(fam.discover(ctx))
    receipt = FAM._receipt_path(root, A_URL)
    kept = receipt.read_bytes()
    receipt.unlink()
    with pytest.raises(RuntimeError, match="no sealed terminal receipt"):
        fam.acquire(units[0], ctx)
    receipt.write_bytes(kept)
    # The run holds two bodies; directory order decides which next() picks (CI red 2026-10-30).
    for body in (p for p in (tmp_path / "cc/original/decoded-html").rglob("*") if p.is_file()):
        body.write_bytes(b"tampered")
    with pytest.raises(Exception, match="digest|disagree|invalid|mismatch|differ"):
        fam.acquire(units[0], ctx)


@pytest.mark.parametrize("fid", DIRECT_IDS)
def test_direct_discover_and_acquire_read_the_sealed_run(fid, tmp_path):
    root = run_direct(DIRECT, tmp_path)
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path, root=str(root), cc_root=str(tmp_path / "cc"),
               artifact_root=str(tmp_path))
    units = {u.url: u for u in fam.discover(ctx)}
    assert sorted(units) == sorted((B_URL, C_URL, "https://www.canada.ca/en/d.html"))
    (cap,) = fam.acquire(units[C_URL], ctx)
    assert cap.body == b"<html><main>origin copy</main></html>"
    assert cap.meta["status"] == 200 and cap.meta["content_type"].startswith("text/html")
    with pytest.raises(FAM.UnitExcluded) as declined:                 # consent search_ok false
        fam.acquire(units[B_URL], ctx)
    assert declined.value.reason == "consent_declined"
    with pytest.raises(FAM.UnitExcluded) as binary:                   # image/png body
        fam.acquire(units["https://www.canada.ca/en/d.html"], ctx)
    assert binary.value.reason == "non_textual_empty_or_non_2xx"


@pytest.mark.parametrize("fid", DIRECT_IDS)
def test_direct_unknown_consent_is_quarantined_not_excluded(fid, tmp_path):
    cc = cc_miss_fixture(tmp_path)
    root = tmp_path / "direct"
    DIRECT.acquire(cc_root=cc, root=root, artifact_root=tmp_path, rate=2,
                   fetcher=lambda _u: (_ for _ in ()).throw(AssertionError("fetched")),
                   consent_getter=lambda h: {"host": h, "status": "unknown", "search_ok": True,
                                             "train_ok": True, "mode": "direct", "signals": {},
                                             "reasons": ["policy endpoint unavailable"],
                                             "checked_at": "2026-08-16T00:00:00Z"})
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path, root=str(root), cc_root=str(cc), artifact_root=str(tmp_path))
    unit = list(fam.discover(ctx))[0]
    with pytest.raises(RuntimeError, match="consent_unknown_retryable"):
        fam.acquire(unit, ctx)


@pytest.mark.parametrize("fid", ALL_IDS)
def test_terms_come_from_the_contract_row_as_declared(fid, tmp_path):
    root = run_cc(CCP, tmp_path)
    fam, row, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path, root=str(root), cc_root=str(root), artifact_root=str(tmp_path))
    try:
        fam.discover(ctx)
    except Exception:       # the direct rows need a miss-only CC run; the terms are noted first
        pass
    t = ctx.notes["contract_terms"]
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_resolution"] == row["consent"]["resolution"]
    assert t["licence_basis"] == row["licence"]["basis"]
    assert t["licence_spdx"] == row["licence"]["spdx"]
    with pytest.raises(ValueError, match="contract row"):
        fam.discover(_ctx(fid, tmp_path, {"family_id": fid}, root=str(root)))


@pytest.mark.parametrize("fid", ALL_IDS)
def test_the_live_route_refuses_a_row_that_declares_no_search(fid, tmp_path):
    fam, row, _s = registry.admit(fid)
    declined = json.loads(json.dumps(row))
    declined["consent"]["search_ok"] = False
    ctx = _ctx(fid, tmp_path, declined, root=str(tmp_path), cc_root=str(tmp_path),
               artifact_root=str(tmp_path))
    with pytest.raises(ValueError, match="search_ok false"):
        fam.acquire_run(ctx)


HTML = ("<html lang='en'><head><title> A  page\ntitle </title></head><body><main><h1>Heading</h1>"
        "<p>%s</p></main></body></html>") % ("Alpha beta gamma delta. " * 40)
CANADA = ("ca_canadaca_captured_exact", "ca_canadaca_direct_exact")
COCHRANE = ("cochrane_preserved_exact", "cochrane_direct_exact")


def _cap(url, html=HTML):
    from cytir.sources.family import Captured, Unit
    return Captured(Unit(key="k", url=url), html.encode(), {})


@pytest.mark.parametrize("fid", CANADA)
def test_canada_extract_is_the_sealed_parent(fid, tmp_path):
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path)
    url = "https://www.canada.ca/en/x.html"
    (doc,) = fam.extract(_cap(url), ctx)
    assert doc.title == "A page title" and doc.lang == "en"
    assert doc.text.endswith("\n") and doc.text == doc.source_text
    assert doc.extra["document_id"] == hashlib.sha256(url.encode()).hexdigest()
    assert doc.extra["markdown_sha256"] == hashlib.sha256(doc.text.encode()).hexdigest()


_V2_WITHHELD = (
    "data/chunk-reference-selector.v2.json is withheld from the public cut "
    "(registry: public=false, sealed derivation history; v4 ships)")


def _skip_if_selector_v2_withheld() -> None:
    """Skip ONE test (not the module) when the pinned v2 selector is absent, as in a stranger's cut."""
    if not (pathlib.Path(__file__).resolve().parent.parent / "data" / "chunk-reference-selector.v2.json").is_file():
        pytest.skip(_V2_WITHHELD)


@pytest.mark.parametrize("fid", CANADA)
def test_canada_chunk_follows_the_ratified_selector_per_page(fid, tmp_path, word_only_chunking_allowed):
    _skip_if_selector_v2_withheld()
    from types import SimpleNamespace
    from cytir.sources.family import Doc
    fam, row, _s = registry.admit(fid)
    assert row["chunk_policy"]["sha256"].startswith("73507461") and row["chunk_policy"]["epoch"] == 5
    ctx = _ctx(fid, tmp_path)
    doc = Doc(url="https://www.canada.ca/en/x.html", title="T", lang="en", source_text="",
              extra={"document_id": "d" * 64}, text="# T\n\n## A\n\nalpha beta gamma.\n\n## B\n\ndelta epsilon.\n")
    with pytest.raises(RuntimeError, match="before preflight"):
        fam.chunk(doc, ctx)
    fam.preflight(ctx)
    assert fam._selector.config["selector_id"]
    seen = []

    def stub(family):
        def classify(text, **kw):
            seen.append(kw)
            return SimpleNamespace(policy_family=family)
        return classify

    fam._selector.classify = stub("chunk.reference")
    ref = fam.chunk(doc, ctx)
    fam._selector.classify = stub("chunk.generic")
    gen = fam.chunk(doc, ctx)
    assert ref and gen and seen == [{"has_native_anchor": False}] * 2
    assert all(r["stamps"]["chunk"] == 5 for r in ref)      # reference 192w, epoch 5
    assert all(r["stamps"]["chunk"] == 4 for r in gen)      # unselected generic 768w, epoch 4
    assert [r["chunk_ordinal"] for r in ref] == list(range(len(ref)))
    assert all(r["anchor"] == f"document:{'d' * 64}:chunk-{r['chunk_ordinal']}" for r in gen)


def test_canada_mixed_selector_must_be_the_pinned_artifact(tmp_path):
    _skip_if_selector_v2_withheld()
    fid = CANADA[0]
    fam, _r, _s = registry.admit(fid)
    ctx = _ctx(fid, tmp_path)
    ctx.contract["chunk_policy"] = {**ctx.contract["chunk_policy"], "selected_by": {
        **ctx.contract["chunk_policy"]["selected_by"], "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="pins"):
        fam.preflight(ctx)


def test_canada_heading_only_page_is_a_terminal_exclusion(tmp_path):
    from cytir.sources.family import UnitExcluded
    fam, _r, _s = registry.admit("ca_canadaca_direct_exact")
    html = "<html><head><title>t</title></head><body><main><h1>Only a heading</h1></main></body></html>"
    with pytest.raises(UnitExcluded):
        fam.extract(_cap("https://www.canada.ca/en/y.html", html), _ctx("ca_canadaca_direct_exact", tmp_path))


@pytest.mark.parametrize("fid", COCHRANE)
def test_cochrane_extract_takes_title_and_lang_from_the_sealed_manifest(fid, tmp_path):
    import gzip
    fam, _r, _s = registry.admit(fid)
    url = "https://www.cochrane.org/evidence/CD000001_x"
    man = tmp_path / "source-manifest.jsonl.gz"
    with gzip.open(man, "wt") as f:
        f.write(json.dumps({"url": url, "title": "Current title | Cochrane", "lang": "en"}) + "\n")
    with pytest.raises(ValueError, match="current_manifest"):
        fam.extract(_cap(url), _ctx(fid, tmp_path))
    ctx = _ctx(fid, tmp_path, current_manifest=str(man))
    (doc,) = fam.extract(_cap(url), ctx)
    assert doc.title == "Current title | Cochrane" and doc.lang == "en"
    assert doc.text.startswith("# Current title | Cochrane\n\nSource: [Cochrane evidence record](" + url + ")\n\n")
    assert doc.extra["document_id"] == hashlib.sha256(("cochrane-evidence\0" + url).encode()).hexdigest()


@pytest.mark.parametrize("fid", COCHRANE)
def test_cochrane_chunk_runs_the_declared_policy_and_invents_no_values(fid, tmp_path, word_only_chunking_allowed):
    from cytir.sources.family import Doc
    fam, row, _s = registry.admit(fid)
    assert not getattr(fam, "PARTIAL", ())
    assert row["chunk_policy"]["sha256"].startswith("7dcb54c4") and row["chunk_policy"]["epoch"] == 4
    ctx = _ctx(fid, tmp_path)
    doc = Doc(url="https://www.cochrane.org/evidence/CD1_x", title="T", lang="en", source_text="",
              extra={"document_id": "d" * 64}, text="# T\n\n## A\n\nalpha beta gamma.\n\n## B\n\ndelta epsilon.\n")
    with pytest.raises(RuntimeError, match="before preflight"):
        fam.chunk(doc, ctx)
    fam.preflight(ctx)
    recs = fam.chunk(doc, ctx)
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["anchor"] == f"document:{'d' * 64}:chunk-{r['chunk_ordinal']}" for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)
    assert not any(k in r for r in recs for k in ("tier", "source_type", "trust_tier"))
