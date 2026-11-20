"""ADR-024 amendment 3 (RATIFIED Tim 2026-08-19, PR #61; option (c)).

An UNREADABLE policy endpoint is not an ABSENT one — and not a refusal either.
These tests pin the sub-class split: (a) robots readable+permissive with an
unreadable tdmrep => admit SEARCH ONLY with train_ok=false, but only for hosts
carrying a sealed witness in the register; (b) everything else keeps declining.

The Wave28 livelock these encode: a host answering 5xx on an absent well-known
path can never resolve, where a 404 resolves cleanly as "no reservation".
"""
import os
import tempfile

import pytest

import cytir.consent.resolver as CR


def _reg_yaml(hosts_block: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(hosts_block)
    return path


ENTRY = """\
adr: ADR-024 amendment 3
hosts:
- host: www.example-5xx.test
  search_ok: true
  train_ok: false
  witness: artifact://w/hosts/www.example-5xx.test.json
  witness_sha256: aa
  egress_liveness: artifact://w/EGRESS-LIVENESS.json
  egress_liveness_sha256: bb
"""


def _rec(**over):
    """A held record whose ONLY unresolved leg is tdmrep — the amendment-3 shape."""
    rec = {
        "host": "www.example-5xx.test", "status": "unknown",
        "search_ok": True, "train_ok": True,
        "signals": {"robots": {"fetched": True, "http_status": 200},
                    "tdmrep": {"fetched": False, "error": "HTTP 502"}},
        "reasons": [],
    }
    rec.update(over)
    return rec


@pytest.fixture(autouse=True)
def _clear_cache():
    CR._UNRESOLVED_REGISTER_CACHE.clear()
    yield
    CR._UNRESOLVED_REGISTER_CACHE.clear()


def test_absent_register_is_feature_off():
    assert CR.load_unresolved_register("/nonexistent/register.yaml") == {}


def test_registered_host_with_unreadable_tdmrep_is_search_only():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    assert CR.unresolved_search_ok(_rec(), reg) is True


def test_unregistered_host_is_never_admitted():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    assert CR.unresolved_search_ok(_rec(host="other.test"), reg) is False


def test_robots_unreadable_is_subclass_b_not_admitted():
    """euipo case: 403 on robots too — we cannot establish permission at all."""
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": False, "error": "HTTP 403"},
                        "tdmrep": {"fetched": False, "error": "HTTP 403"}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_robots_status_without_readable_answer_not_admitted():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": True, "http_status": 403},
                        "tdmrep": {"fetched": False}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_readable_tdmrep_is_not_this_class():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                        "tdmrep": {"fetched": True, "http_status": 200}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_robots_decline_is_never_overridden():
    """search_ok=False means robots already declined us — amendment 3 must not flip it."""
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    assert CR.unresolved_search_ok(_rec(search_ok=False), reg) is False


def test_resolved_record_is_out_of_scope():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    assert CR.unresolved_search_ok(_rec(status="ok"), reg) is False


def test_missing_witness_fails_loud():
    bad = ENTRY.replace("  witness_sha256: aa\n", "")
    with pytest.raises(ValueError, match="witness_sha256"):
        CR.load_unresolved_register(_reg_yaml(bad))


def test_missing_egress_proof_fails_loud():
    bad = ENTRY.replace("  egress_liveness_sha256: bb\n", "")
    with pytest.raises(ValueError, match="egress_liveness_sha256"):
        CR.load_unresolved_register(_reg_yaml(bad))


def test_train_ok_must_be_false():
    bad = ENTRY.replace("  train_ok: false", "  train_ok: true")
    with pytest.raises(ValueError, match="train_ok=false"):
        CR.load_unresolved_register(_reg_yaml(bad))


def test_shipped_register_is_valid_and_search_only():
    """The real register must load and stamp every host search-only."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "consent-unresolved-register.yaml")
    reg = CR.load_unresolved_register(path)
    assert reg, "shipped consent-unresolved register must not be empty"
    for host, entry in reg.items():
        assert entry["search_ok"] is True and entry["train_ok"] is False, host
        assert entry["witness_sha256"] and entry["egress_liveness_sha256"], host


def test_shipped_declined_residue_is_well_formed_and_never_admitted():
    """Task 554: the real register's `declined_residue` bucket (sub-class b —
    hosts whose robots.txt itself is unreadable, so the amendment-3 admission
    premise never holds) is auditable data, not an admission path. Every
    entry must carry a witness, and none of them may leak into the
    `hosts`/admission mapping `load_unresolved_register` actually returns."""
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "consent-unresolved-register.yaml")
    doc = yaml.safe_load(open(path)) or {}
    residue = doc.get("declined_residue") or []
    assert residue, "shipped declined_residue bucket must not be empty"
    residue_hosts = {e["host"] for e in residue}
    assert "economie.gouv.fr" in residue_hosts
    reg = CR.load_unresolved_register(path)
    for entry in residue:
        assert entry["search_ok"] is False, entry["host"]
        assert entry["witness"] and entry["witness_sha256"], entry["host"]
        assert entry["rationale"], entry["host"]
        # sub-class (b) is documentary only -- it must never admit a host
        assert entry["host"] not in reg, (
            f"{entry['host']} is in declined_residue AND in the admission "
            "mapping -- sub-class (b) must never grant search_ok")


def test_economie_gouv_fr_evidence_matches_the_euipo_shape():
    """The new row's evidence must actually be the robots-itself-unreadable
    shape (both legs unreadable), not the sub-class-(a) tdmrep-only shape --
    otherwise it belongs under `hosts:` with a witnessed admission, not here."""
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "consent-unresolved-register.yaml")
    doc = yaml.safe_load(open(path)) or {}
    entry = next(e for e in doc.get("declined_residue") or []
                 if e["host"] == "economie.gouv.fr")
    ev = entry["evidence"]
    assert ev["robots_http_status"] == 403
    assert ev["tdmrep_http_status"] == 403
    assert ev.get("reproduced_with_browser_ua") is True


# --- "definitively unreadable" means 5xx ONLY (Tim's ruling, 2026-08-28) -----
# The ratified text of amendment 3 describes the 5xx livelock; the code admitted
# ANY non-answer. Text and code disagreed and the text won (task 466, Q3(b)).

def test_transport_error_is_our_instrument_not_a_publisher_verdict():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    for error in ("ConnectError", "ReadTimeout", "RemoteProtocolError",
                  "ConnectTimeout", "TooManyRedirects"):
        rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                            "tdmrep": {"fetched": False, "error": error}})
        assert CR.unresolved_search_ok(rec, reg) is False, error


def test_budget_exhausted_is_not_definitively_unreadable():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                        "tdmrep": {"fetched": False,
                                   "error": "policy endpoint budget exhausted"}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_an_unrecorded_error_never_promotes():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                        "tdmrep": {"fetched": False}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_every_5xx_still_promotes():
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    for status in (500, 502, 503, 504, 520):
        rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                            "tdmrep": {"fetched": False,
                                       "error": f"HTTP {status}"}})
        assert CR.unresolved_search_ok(rec, reg) is True, status


def test_structured_class_is_preferred_over_the_error_string():
    """Receipts sealed from task 466 on carry the class; older ones do not."""
    reg = CR.load_unresolved_register(_reg_yaml(ENTRY))
    rec = _rec(signals={"robots": {"fetched": True, "http_status": 200},
                        "tdmrep": {"fetched": False, "error": "HTTP 503",
                                   "unreadable_class": "transport_error"}})
    assert CR.unresolved_search_ok(rec, reg) is False


def test_classify_unreadable_names_each_terminal_class():
    assert CR.classify_unreadable("HTTP 503") == "http_5xx"
    assert CR.classify_unreadable("HTTP 403") == "http_4xx"
    assert CR.classify_unreadable("ReadTimeout") == "transport_error"
    assert CR.classify_unreadable("policy endpoint budget exhausted") == "budget_exhausted"
    assert CR.classify_unreadable(None) == "unrecorded"


def test_explicit_refusal_is_visible_on_an_unresolved_record():
    """The euipo shape: 403 on robots, nothing readable on tdmrep."""
    rec = _rec(search_ok=False,
               signals={"robots": {"fetched": True, "http_status": 403},
                        "tdmrep": {"fetched": False, "error": "HTTP 403"}})
    assert CR.explicit_search_refusal(rec) is True


def test_training_only_refusal_is_not_a_search_refusal():
    rec = _rec(train_ok=False,
               signals={"robots": {"fetched": True, "http_status": 200,
                                   "train_tokens_disallowed": ["gptbot"]},
                        "tdmrep": {"fetched": False, "error": "ConnectError"}})
    assert CR.explicit_search_refusal(rec) is False
