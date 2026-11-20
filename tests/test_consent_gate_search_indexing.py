import cytir.consent.gate as consent_gate


def receipt(*, status="ok", search_ok=True, train_ok=False):
    return {"status": status, "search_ok": search_ok, "train_ok": train_ok,
            "signals": {}, "reasons": [], "checked_at": "2026-08-03T00:00:00Z"}


def test_search_indexing_admits_resolved_search_only_receipt():
    value = receipt(search_ok=True, train_ok=False)
    assert consent_gate.gate_bucket(value) == "declined"
    assert consent_gate.gate_bucket(value, search_indexing=True) == "ok_search_only"


def test_search_indexing_still_rejects_search_decline_and_unknown():
    assert consent_gate.gate_bucket(
        receipt(search_ok=False, train_ok=False), search_indexing=True) == "declined"
    assert consent_gate.gate_bucket(
        receipt(status="unknown", search_ok=True), search_indexing=True) == "unknown"


def test_search_indexing_preserves_full_consent_class():
    assert consent_gate.gate_bucket(
        receipt(search_ok=True, train_ok=True), search_indexing=True) == "ok"


def test_search_indexing_declines_non_resolved_error_status():
    assert consent_gate.gate_bucket(
        receipt(status="error", search_ok=True, train_ok=False),
        search_indexing=True) == "declined"
