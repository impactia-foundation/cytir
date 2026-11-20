"""Task 502 defect 1 / ADR-103: an unreadable robots.txt is HELD, not DECLINED.

``consent_terminal`` used to collapse two different origin-consent shapes into
one "declined" outcome: (a) a robots.txt that was actually READ and says no
(a wildcard/our-token/TDM/noai disallow, or a path-specific Disallow), and
(b) a robots.txt that was NEVER READ at all -- access-controlled (401/403), or
a redirect chain that ended with no Location to follow (task 502 defect 2).
ADR-073's own text calls (b) a RETRY posture ("re-probing from a second
vantage + reason-tagged retry queue"), not a refusal, and the 7,925-row
task-467 anatomy (the task 502 evidence) found (b) was 61% of every
row ever counted "declined". These tests pin the split, and pin that a v5
caller (re-deriving a SEALED task-426 receipt) still gets the old, undivided
reading -- because rewriting what a sealed receipt's own ``status`` says is
forbidden, not because the old reading was right.
"""
import cytir.consent.terminal as DTC


ORIGIN = "https://example.test"
URL = "https://example.test/some/path"


def _receipt(*, status="ok", search_ok=True, train_ok=True,
             robots=None, robots_body=""):
    return {
        "origin": ORIGIN, "origin_scheme": "https", "host": "example.test",
        "mode": "direct", "status": status, "search_ok": search_ok,
        "train_ok": train_ok, "checked_at": "2026-09-03T00:00:00Z",
        "signals": {"robots": robots or {}},
        "_robots_body": robots_body,
    }


# --- host-level search_ok=False: readable vs unreadable ----------------------

def test_access_controlled_401_is_held_unreadable_not_declined():
    receipt = _receipt(search_ok=False, robots={
        "fetched": True, "http_status": 401})
    assert DTC.consent_terminal(receipt, URL) == ("held", "consent_unreadable")


def test_access_controlled_403_is_held_unreadable_not_declined():
    receipt = _receipt(search_ok=False, robots={
        "fetched": True, "http_status": 403})
    assert DTC.consent_terminal(receipt, URL) == ("held", "consent_unreadable")


def test_readable_wildcard_disallow_stays_declined():
    """A body was actually parsed and says no -- a genuine decline."""
    receipt = _receipt(
        search_ok=False, robots={"fetched": True, "http_status": 200},
        robots_body="User-agent: *\nDisallow: /\n")
    assert DTC.consent_terminal(receipt, URL) == (
        "declined", "origin_search_declined")


# --- path-level direct_url_allowed=False: readable vs unreadable ------------

def test_redirect_terminal_no_location_is_held_unreadable():
    """task 502 defect 2 shape: a 3xx terminal with no Location, empty body."""
    receipt = _receipt(robots={"fetched": True, "http_status": 301})
    assert DTC.consent_terminal(receipt, URL) == ("held", "consent_unreadable")


def test_other_empty_body_non_permissive_status_is_held_unreadable():
    for status in (406, 429, 202):
        receipt = _receipt(robots={"fetched": True, "http_status": status})
        assert DTC.consent_terminal(receipt, URL) == (
            "held", "consent_unreadable"), status


def test_readable_path_specific_disallow_stays_declined():
    receipt = _receipt(
        robots={"fetched": True, "http_status": 200},
        robots_body="User-agent: authoritative-web\nDisallow: /some\n")
    assert DTC.consent_terminal(receipt, URL) == (
        "declined", "path_robots_declined")


def test_definitive_no_policy_answers_are_allowed_not_declined():
    """404/410/empty-200 are readable, explicit no-policy answers -- open."""
    for status in (404, 410):
        receipt = _receipt(robots={"fetched": True, "http_status": status})
        assert DTC.consent_terminal(receipt, URL) is None, status


# --- unaffected paths ---------------------------------------------------------

def test_unresolved_status_is_still_held_consent_unresolved():
    receipt = _receipt(status="unknown")
    assert DTC.consent_terminal(receipt, URL) == ("held", "consent_unresolved")


def test_fetching_bound_is_unchanged_by_the_split():
    """The whole point: nothing gets fetched that wasn't before. The split only
    renames the reason a fetch is refused, never the fetch/no-fetch verdict."""
    cases = [
        _receipt(search_ok=False, robots={"fetched": True, "http_status": 401}),
        _receipt(robots={"fetched": True, "http_status": 301}),
    ]
    for receipt in cases:
        outcome = DTC.consent_terminal(receipt, URL)
        assert outcome is not None and outcome[0] in ("held", "declined")


# --- schema-version pinning: v5 receipts stay readable ------------------------

def test_v5_pin_reproduces_the_old_undivided_declined_reading():
    receipt = _receipt(search_ok=False, robots={
        "fetched": True, "http_status": 401})
    assert DTC.consent_terminal(
        receipt, URL, schema_version=DTC.SCHEMA_VERSION_V5) == (
        "declined", "origin_search_declined")

    receipt2 = _receipt(robots={"fetched": True, "http_status": 301})
    assert DTC.consent_terminal(
        receipt2, URL, schema_version=DTC.SCHEMA_VERSION_V5) == (
        "declined", "path_robots_declined")


def test_v6_is_the_default_with_no_schema_version_argument():
    receipt = _receipt(search_ok=False, robots={
        "fetched": True, "http_status": 401})
    assert DTC.consent_terminal(receipt, URL) == DTC.consent_terminal(
        receipt, URL, schema_version=DTC.SCHEMA_VERSION_V6)


def test_unknown_schema_version_is_refused():
    receipt = _receipt()
    try:
        DTC.consent_terminal(receipt, URL, schema_version="nonsense")
        assert False, "expected DirectTerminalContractError"
    except DTC.DirectTerminalContractError:
        pass


# --- consent_resolver.robots_unreadable[_class] -- the shared classifier -----

# --- the actual sealed evidence (task 502 anatomy, live examples) -----------
# The exact ``receipt`` object read back from the sealed task-426 consent
# artifact for ``www.gc-handball.ch`` (read-only fetch from the GPU build box,
# _preservation/task426/r0-adr092-direct-terminal-rollout-0a9a6aa1-p00316-a1/.
# .../consent/38/38da85...json.gz). The sealed TERMINALS.jsonl for this row
# records ``"status":"declined","reason":"path_robots_declined"`` -- exactly
# what SCHEMA_VERSION_V5 reproduces below. This is the 61%-of-declines class
# ADR-103 named: the origin never published a readable robots.txt at all.
GC_HANDBALL_SEALED_RECEIPT = {
    "_robots_body": "", "checked_at": "2026-08-28T16:49:37Z",
    "host": "www.gc-handball.ch", "mode": "direct",
    "origin": "http://www.gc-handball.ch", "origin_scheme": "http",
    "reasons": ["no opt-out signal found"], "search_ok": True,
    "signals": {
        "robots": {"fetched": True, "http_status": 301,
                   "note": "no robots rules published (open)",
                   "transport": "http"},
        "tdmrep": {"fetched": True, "http_status": 301, "transport": "http"},
    },
    "status": "ok", "train_ok": True,
}


def test_sealed_gc_handball_receipt_v5_reproduces_the_sealed_declined_reason():
    """v5 pin must byte-for-byte reproduce what the sealed rollout wrote."""
    outcome = DTC.consent_terminal(
        GC_HANDBALL_SEALED_RECEIPT, "http://www.gc-handball.ch",
        schema_version=DTC.SCHEMA_VERSION_V5)
    assert outcome == ("declined", "path_robots_declined")


def test_sealed_gc_handball_receipt_v6_reclassifies_as_held_unreadable():
    """Same evidence, v6 default: this was never a policy answer, so it holds
    for retry (ADR-073) instead of counting as a refusal (ADR-103)."""
    outcome = DTC.consent_terminal(
        GC_HANDBALL_SEALED_RECEIPT, "http://www.gc-handball.ch")
    assert outcome == ("held", "consent_unreadable")


def test_robots_unreadable_class_matches_every_case_above():
    import cytir.consent.resolver as CR
    assert CR.robots_unreadable_class(
        {"signals": {"robots": {"fetched": True, "http_status": 401}}}
    ) == "access_controlled"
    assert CR.robots_unreadable_class(
        {"signals": {"robots": {"fetched": True, "http_status": 301}}}
    ) == "redirect_no_location"
    assert CR.robots_unreadable_class(
        {"signals": {"robots": {"fetched": True, "http_status": 404}}}
    ) == "readable"
    assert CR.robots_unreadable_class(
        {"signals": {"robots": {"fetched": True, "http_status": 200}},
         "_robots_body": "User-agent: *\nDisallow: /\n"}
    ) == "readable"
    assert CR.robots_unreadable_class({"signals": {}}) == "unfetched"
    assert CR.robots_unreadable({
        "signals": {"robots": {"fetched": True, "http_status": 403}}}) is True
    assert CR.robots_unreadable({
        "signals": {"robots": {"fetched": True, "http_status": 404}}}) is False
