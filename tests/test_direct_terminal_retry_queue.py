"""Task 523: a rate limit is not a disposition — 429 belongs in a retry queue.

Task 521 measured the cost of the old reading: 1,935 of the 2,202 points the
1024 rebuild lost for ``commission.europa.eu`` carry the terminal reason
``http_429``.  That is OUR rate limit, sealed as though the European Commission
had answered about the document (verification doctrine 6: an error is not an
empty).  The whole sealed class, from
the task 523 exploration script ``inventory_http_terminals.py`` over 526 rollout partitions
and 361,050 terminal rows:

    http_429  2,181 urls /  3,614 points     http_502   111 urls / 168 points
    http_503    139 urls /    157 points     http_504     2 urls /   3 points

``http_outcome`` moves exactly those come-back-later codes to ``retry_queued``
at v7 and leaves every other code — 404, 403, 406, 401, 410, 400, 500, and the
Cloudflare 52x family — terminal, unchanged.  Sealed v5/v6 ledgers re-derive
byte-identically, so no receipt already on disk is rewritten.
"""
import cytir.consent.terminal as DTC


V5, V6, V7, V8 = (DTC.SCHEMA_VERSION_V5, DTC.SCHEMA_VERSION_V6,
                  DTC.SCHEMA_VERSION_V7, DTC.SCHEMA_VERSION_V8)


# --- the class, and only the class ------------------------------------------

def test_429_is_a_retry_queue_outcome_at_v7():
    assert DTC.http_outcome(429, schema_version=V7) == ("retry_queued", "http_429")


def test_the_whole_come_back_later_class_is_requeued():
    for code in (408, 425, 429, 502, 503, 504):
        assert DTC.http_outcome(code, schema_version=V7) == (
            "retry_queued", f"http_{code}"), code


def test_real_dead_ends_stay_terminal():
    # These are the codes the sealed inventory actually holds beside 429; a
    # 404/403/406 is an answer about the URL, and 500 is not a documented
    # come-back-later signal.
    for code in (400, 401, 402, 403, 404, 406, 410, 421, 500, 510,
                 520, 521, 522, 525):
        assert DTC.http_outcome(code, schema_version=V7) == (
            "http_terminal", f"http_{code}"), code


def test_empty_success_body_is_unchanged_at_every_version():
    for version in (V5, V6, V7, V8):
        assert DTC.http_outcome(204, has_body=False, schema_version=version) == (
            "http_terminal", "empty_success_body")


# --- forward-only: sealed evidence re-derives to itself ----------------------

def test_sealed_v5_and_v6_ledgers_still_read_429_as_terminal():
    for version in (V5, V6):
        assert DTC.http_outcome(429, schema_version=version) == (
            "http_terminal", "http_429"), version


def test_429_is_still_a_retry_queue_outcome_at_v8():
    """Task 542 (v8) touches a different fork of ``consent_terminal`` --
    this class must carry forward unchanged, not silently regress the
    moment v8 becomes ``CURRENT_SCHEMA_VERSION`` (see ``http_outcome``'s
    own docstring)."""
    for code in (408, 425, 429, 502, 503, 504):
        assert DTC.http_outcome(code, schema_version=V8) == (
            "retry_queued", f"http_{code}"), code


def test_v8_is_current_and_every_version_is_known():
    assert DTC.CURRENT_SCHEMA_VERSION == V8
    assert DTC.SCHEMA_VERSIONS == (V5, V6, V7, V8)


def test_unknown_schema_version_is_refused_not_defaulted():
    try:
        DTC.http_outcome(429, schema_version="adr092-direct-terminal.v99")
    except DTC.DirectTerminalContractError:
        return
    raise AssertionError("an unknown terminal schema version must fail loud")


def test_retry_queued_is_a_declared_outcome():
    assert "retry_queued" in DTC.OUTCOMES
