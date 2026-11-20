"""The stage-time refusal for an extraction collapse (task 523).

The incident this guards against, in its own numbers: `task472_f2_html`
shipped 21 digest cohorts of >= 25 distinct URLs sharing one body -- 7,985 of
127,074 staged rows, 6.28% -- and every count check passed, because
conservation counts rows and 3,010 copies of the ".gov websites use HTTPS"
banner are 3,010 rows. The tests below replay those exact numbers through the
contract and require a FAIL, then replay the post-fix lane (4,098 of 168,128 =
2.44%) and require a PASS.
"""
import json

from cytir.index import payload_shape_contract as PSC

SET = "task472_f2_html_1024_embed"


def _contract(**limits):
    base = {"min_cohort_urls": 25, "sparse_posting_ceiling": 256,
            "max_collapsed_row_frac": 0.03}
    base.update(limits)
    return {"schema": PSC.SCHEMA, "_path": "<test>",
            "defaults": {"max_missing_serving_signal_frac": 1.0,
                         "max_no_bm25_posting_frac": 1.0,
                         "max_provenance_head_frac": 1.0},
            "extraction_collapse": {SET: base}}


def _receipts(rows, collapsed_rows, cohorts, top=None, *, measured=True,
              partitions=2):
    collapse = {"rows": rows, "min_cohort_urls": 25,
                "sparse_posting_ceiling": 256, "cohorts": cohorts,
                "collapsed_rows": collapsed_rows,
                "collapsed_urls": collapsed_rows,
                "top_cohorts": top or [], "partitions": partitions}
    out = []
    for shard in range(partitions):
        receipt = {"set_id": SET, "shard": f"source=x.shard-{shard:04d}",
                   "staged": rows // partitions, "serving_metadata": {},
                   "staged_without_sparse": 0}
        if measured:
            # Set-level and identical on every receipt of the set, exactly as
            # `backfill_extraction_collapse` writes it.
            receipt["extraction_collapse"] = collapse
        out.append(receipt)
    return out


# --- the incident, replayed -------------------------------------------------

def test_the_shipped_f2_lane_is_refused():
    report = PSC.evaluate(_receipts(127074, 7985, 21, top=[{
        "content_digest": "eeoc", "distinct_urls": 3010, "rows": 3010,
        "sparse_postings": 34,
        "snippet": "** Official websites use .gov ** A **.gov** website belongs"}]),
        _contract())
    assert report["verdict"] == "FAIL"
    (finding,) = [f for f in report["findings"] if "digest cohort" in f]
    assert "7,985 of 127,074" in finding
    assert "3,010 URLs" in finding
    assert "34 postings" in finding
    assert "Official websites use .gov" in finding


def test_the_repaired_f2_lane_passes():
    report = PSC.evaluate(_receipts(168128, 4098, 20), _contract())
    assert report["verdict"] == "PASS", report["findings"]
    (row,) = report["sets"]
    assert row["collapsed_row_frac"] == round(4098 / 168128, 8)
    assert row["max_collapsed_row_frac"] == 0.03


# --- the contract cannot be satisfied by not looking ------------------------

def test_a_declared_but_unmeasured_set_fails():
    report = PSC.evaluate(_receipts(168128, 0, 0, measured=False), _contract())
    assert report["verdict"] == "FAIL"
    assert any("no partition measured it" in f for f in report["findings"])


def test_a_set_that_declares_no_rule_is_not_evaluated():
    contract = _contract()
    contract["extraction_collapse"] = {}
    report = PSC.evaluate(_receipts(168128, 4098, 20), contract)
    assert report["verdict"] == "PASS"
    # The measurement still rides along (it is free once taken); what an
    # undeclared set must NOT get is a verdict against a number nobody chose.
    assert "collapsed_row_frac" not in report["sets"][0]
    assert "max_collapsed_row_frac" not in report["sets"][0]


def test_every_threshold_must_be_declared_never_defaulted():
    contract = _contract()
    del contract["extraction_collapse"][SET]["sparse_posting_ceiling"]
    report = PSC.evaluate(_receipts(168128, 4098, 20), contract)
    assert report["verdict"] == "FAIL"
    assert any("declares no sparse_posting_ceiling" in f
               for f in report["findings"])


def test_the_set_level_measurement_is_read_once_not_summed():
    """Eight partitions of one set must not report eight times its own rows."""
    report = PSC.evaluate(_receipts(168128, 4098, 20, partitions=8), _contract())
    (row,) = report["sets"]
    assert row["extraction_collapse"]["rows"] == 168128


# --- the measurement itself, on a synthetic payload tree --------------------

def test_measure_separates_a_banner_cohort_from_a_shared_document(tmp_path):
    import gzip

    def row(url, digest, postings, snippet):
        return {"id": url, "payload": {"url": url, "content_digest": digest,
                                       "snippet": snippet},
                "bm25": {"indices": list(range(postings)), "values": []}}

    path = tmp_path / "s.payload.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for i in range(30):                       # a collapse: short, many urls
            fh.write(json.dumps(row(f"https://a.test/{i}", "banner", 34,
                                    "Official websites use .gov")) + "\n")
        for i in range(30):                       # a genuine shared document
            fh.write(json.dumps(row(f"https://b.test/{i}", "eurlex", 900,
                                    "Article 65(5)")) + "\n")
        for i in range(30):                       # ordinary distinct pages
            fh.write(json.dumps(row(f"https://c.test/{i}", f"d{i}", 400,
                                    "text")) + "\n")
    out = PSC.measure_extraction_collapse([path], 25, 256)
    assert out["rows"] == 90
    assert out["cohorts"] == 1
    assert out["collapsed_rows"] == 30
    assert out["top_cohorts"][0]["content_digest"] == "banner"
    assert out["top_cohorts"][0]["sparse_postings"] == 34
