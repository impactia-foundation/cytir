"""The two robots-parser defects Tim's 2026-08-28 ruling ordered fixed (task 466).

Both were measured on sealed receipts, not imagined:

  * ``spark.apache.org`` / ``numpy.org`` / ``rubyonrails.org`` publish a robots.txt
    whose only record is ``Sitemap:``. Under RFC 9309 §2.2.1 a crawler with no
    matching group may access everything — a well-formed allow-all. The retracted
    evidence gate read "no policy directives" and held the host.
  * ``asvhg-foot.com`` publishes a real ``User-agent: *`` group behind a CMS
    banner glued to line 1 (``/** Load from cache **/User-agent: *``). Every
    parser here missed the whole group: ``ua_tokens_seen`` came back empty AND
    the group's own Disallow rules never bound us.

The third case in the ruling's evidence — ``hudoc.echr.coe.int`` serving an HTML
page at /robots.txt — stays observable, and stays a non-decision.
"""
import cytir.consent.resolver as CR


SITEMAP_ONLY = "Sitemap: https://spark.apache.org/sitemap.xml\n"
JUNK_PREFIX = ("/** Load from cache **/User-agent: *\n"
               "Allow: /\n"
               "Disallow: /public/2497/private/*\n"
               "Sitemap : https://asvhg-foot.com/sitemap.xml\n")
HTML_BODY = ('\n<!DOCTYPE html>\n<html>\n<head>\n<title>HUDOC</title>\n'
             '<meta name="description" content="User-agent: * Disallow: /">\n'
             '</head></html>\n')


# --- defect 1: a Sitemap-only robots.txt is a well-formed allow-all ----------

def test_sitemap_only_body_is_a_parsed_record_not_an_empty_one():
    shape = CR.robots_body_shape(SITEMAP_ONLY)
    assert shape["non_group_lines"] == 1
    assert shape["group_lines"] == 0
    assert shape["directive_lines"] == 1
    assert shape["unparsed_lines"] == 0
    assert shape["html"] is False
    assert shape["fields"] == ["sitemap"]


def test_sitemap_only_body_permits_every_agent():
    signals = CR._robots_signals(SITEMAP_ONLY)
    assert signals["wildcard_root"] is True
    assert signals["our_agent_root"] is True
    assert signals["ccbot_root"] is True
    rec = {"search_ok": True, "train_ok": True, "reasons": []}
    CR.apply_robots_flags(rec, signals)
    assert (rec["search_ok"], rec["train_ok"]) == (True, True)


# --- defect 2: a junk prefix must not hide the publisher's group -------------

def test_junk_prefix_no_longer_empties_ua_tokens():
    signals = CR._robots_signals(JUNK_PREFIX)
    assert signals["ua_tokens_seen"] == ["*"]
    assert signals["lines_recovered"] == 1


def test_recovered_group_makes_its_own_rules_bind():
    """The point of the fix: the DISALLOW comes back too, not just the token."""
    body = "/** cached **/User-agent: *\nDisallow: /\n"
    signals = CR._robots_signals(body)
    assert signals["wildcard_root"] is False
    rec = {"search_ok": True, "train_ok": True, "reasons": []}
    CR.apply_robots_flags(rec, signals)
    assert rec["search_ok"] is False


def test_bom_prefix_is_stripped():
    signals = CR._robots_signals("﻿User-agent: *\nDisallow: /\n")
    assert signals["ua_tokens_seen"] == ["*"]
    assert signals["wildcard_root"] is False


def test_recovery_is_bounded_to_a_short_prefix():
    body = ("x" * (CR.JUNK_PREFIX_CAP + 5)) + "User-agent: *\nDisallow: /\n"
    text, stats = CR.sanitize_robots_body(body)
    assert stats["recovered_lines"] == 0
    assert text == body


def test_html_body_is_never_recovered_into_directives():
    """Prose that mentions a directive is not policy the publisher wrote."""
    text, stats = CR.sanitize_robots_body(HTML_BODY)
    assert stats["html"] is True
    assert stats["recovered_lines"] == 0
    signals = CR._robots_signals(HTML_BODY)
    assert signals["wildcard_root"] is True      # unchanged from before the fix
    assert CR.robots_body_shape(HTML_BODY)["html"] is True


# --- TDMRep in HTML is value-SENSITIVE --------------------------------------

def test_tdm_meta_content_1_reserves():
    body = '<html><meta name="tdm-reservation" content="1"></html>'
    assert CR.tdm_reservation_in_html(body)["reserved"] is True


def test_tdm_meta_content_0_is_an_explicit_non_reservation():
    body = "<html><meta name='tdm-reservation' content='0'></html>"
    result = CR.tdm_reservation_in_html(body)
    assert result["present"] is True
    assert result["values"] == ["0"]
    assert result["reserved"] is False


def test_tdm_meta_absent():
    assert CR.tdm_reservation_in_html(SITEMAP_ONLY) == {
        "present": False, "values": [], "reserved": False}


# --- offline re-derivation can only tighten ---------------------------------

def _sealed(body, **over):
    receipt = {
        "host": "example.test", "mode": "cc", "status": "ok",
        "search_ok": True, "train_ok": True,
        "signals": {"robots": {"fetched": True, "http_status": 200},
                    "tdmrep": {"fetched": True, "http_status": 404}},
        "_robots_body": body,
    }
    receipt.update(over)
    return receipt


def test_rederivation_leaves_a_genuinely_permissive_receipt_alone():
    out = CR.rederive_sealed_verdict(_sealed(SITEMAP_ONLY))
    assert out["changed"] is False
    assert (out["search_ok"], out["train_ok"]) == (True, True)


def test_rederivation_tightens_when_the_recovered_group_disallows():
    out = CR.rederive_sealed_verdict(
        _sealed("/** cached **/User-agent: *\nDisallow: /\n"))
    assert out["changed"] is True
    assert out["search_ok"] is False
    assert any("disallows" in reason for reason in out["added_reasons"])


def test_rederivation_never_loosens_a_sealed_decline():
    """A sealed refusal stays a refusal even if today's parser sees allow-all."""
    out = CR.rederive_sealed_verdict(
        _sealed(SITEMAP_ONLY, search_ok=False, train_ok=False))
    assert out["changed"] is False
    assert (out["search_ok"], out["train_ok"]) == (False, False)


def test_rederivation_honours_a_tdm_reservation_hidden_in_an_html_body():
    body = '<!DOCTYPE html><html><meta name="tdm-reservation" content="1"></html>'
    out = CR.rederive_sealed_verdict(_sealed(body))
    assert out["changed"] is True
    assert (out["search_ok"], out["train_ok"]) == (False, False)


def test_rederivation_does_not_decline_on_a_content_zero_meta():
    body = '<!DOCTYPE html><html><meta name="tdm-reservation" content="0"></html>'
    out = CR.rederive_sealed_verdict(_sealed(body))
    assert out["changed"] is False
    assert out["search_ok"] is True


def test_rederivation_reports_a_truncated_body():
    out = CR.rederive_sealed_verdict(_sealed("Sitemap: x\n" + "#" * CR.ROBOTS_BODY_CAP))
    assert out["truncated"] is True


# --- ADR-024 amendment 5: a CCBot disallow is a TRAINING signal --------------
# Tim, 2026-09-03: "A cc-bot decline should mean no_train, not no_search."
# Before this, `direct_crawl=False` — the CLI default, and therefore the mode
# of every sweep in the record — turned a CCBot disallow into a full decline.
# No test asserted either branch, which is how the old reading outlived
# ADR-024 amendment 2 (2026-07-15) by seven weeks.

CCBOT_ONLY = "User-agent: CCBot\nDisallow: /\n"
CCBOT_PLUS_WILDCARD_OPEN = ("User-agent: *\nAllow: /\n\n"
                            "User-agent: CCBot\nDisallow: /\n")
WILDCARD_CLOSED = "User-agent: *\nDisallow: /\n"


def _flags(body, **kwargs):
    rec = {"search_ok": True, "train_ok": True, "reasons": []}
    CR.apply_robots_flags(rec, CR._robots_signals(body), **kwargs)
    return rec


def test_ccbot_disallow_is_train_only_in_cc_mode():
    rec = _flags(CCBOT_ONLY, direct_crawl=False)
    assert (rec["search_ok"], rec["train_ok"]) == (True, False)
    assert any("CCBot" in r and "train_ok=False only" in r
               for r in rec["reasons"])


def test_ccbot_disallow_is_train_only_in_direct_mode():
    """Both modes, one verdict — provenance no longer moves any flag."""
    rec = _flags(CCBOT_ONLY, direct_crawl=True)
    assert (rec["search_ok"], rec["train_ok"]) == (True, False)


def test_ccbot_verdict_does_not_depend_on_provenance_mode():
    cc = _flags(CCBOT_PLUS_WILDCARD_OPEN, direct_crawl=False)
    direct = _flags(CCBOT_PLUS_WILDCARD_OPEN, direct_crawl=True)
    assert (cc["search_ok"], cc["train_ok"]) == (True, False)
    assert (cc["search_ok"], cc["train_ok"]) == (direct["search_ok"],
                                                 direct["train_ok"])


def test_wildcard_disallow_still_declines_both_flags():
    """The only robots signal that declines SEARCH is a `*` disallow."""
    for direct in (False, True):
        rec = _flags(WILDCARD_CLOSED, direct_crawl=direct)
        assert (rec["search_ok"], rec["train_ok"]) == (False, False)


def test_wildcard_disallow_plus_ccbot_still_declines_both_flags():
    rec = _flags(WILDCARD_CLOSED + "\nUser-agent: CCBot\nDisallow: /\n")
    assert (rec["search_ok"], rec["train_ok"]) == (False, False)


def test_our_own_token_disallow_still_declines_both_flags():
    rec = _flags("User-agent: %s\nDisallow: /\n" % CR.ROBOT_AGENT)
    assert (rec["search_ok"], rec["train_ok"]) == (False, False)


def test_ccbot_only_host_is_admitted_for_search_indexing():
    """The flip is worth nothing unless the aw_b1 admission rubric sees it."""
    import cytir.consent.gate as consent_gate
    rec = {"host": "example.test", "status": "ok", "search_ok": True,
           "train_ok": True, "reasons": [], "signals": {},
           "checked_at": "2026-09-03T00:00:00Z"}
    CR.apply_robots_flags(rec, CR._robots_signals(CCBOT_ONLY))
    assert consent_gate.gate_bucket(rec, search_indexing=True) == "ok_search_only"


def test_ccbot_flip_does_not_loosen_a_sealed_decline():
    """``rederive_sealed_verdict`` may only tighten, so a sealed CC-mode
    decline is NOT re-opened as a side effect of this amendment. Re-bucketing
    sealed rows is an explicit, auditable re-derivation (task 504)."""
    out = CR.rederive_sealed_verdict(
        _sealed(CCBOT_ONLY, search_ok=False, train_ok=False))
    assert out["changed"] is False
    assert (out["search_ok"], out["train_ok"]) == (False, False)


# --- task 542: the unclear-consent register lacks the "ccbot" token --------
# ADR-024 amendment 2 (2026-07-15) asks a host whose ONLY vendor-specific
# AI-training block is CCBot to be logged in data/consent-unclear-register.yaml
# with the token. Amendment 5 (Tim, 2026-09-03, PR #86, above) gave CCBot its
# own flag-setting branch in `apply_robots_flags` (the `ccbot_root` check) --
# a branch that never touches `train_tokens_disallowed`, which is the only
# list `unclear_search_ok`/the register writer read. A CCBot-only host landed
# `search_ok=True, train_ok=False` with an EMPTY token list, so
# `unclear_search_ok` read "train_ok flipped by something else entirely" and
# never qualified the host -- decisions.md's amendment-5 write-up named this
# and left it "open for Tim". Resolved here: `unclear_search_tokens` folds
# "ccbot" in from `ccbot_root`, so the register writer and `unclear_search_ok`
# both see it, without changing which flags a CCBot disallow sets.
#
# Real record, not synthetic: fetched live from the GPU build box 2026-09-08
# (`curl -A '<UA>' https://www.numerama.com/robots.txt`, rate-limited,
# per-project probe-vantage doctrine -- never from the operator box). numerama.com is
# already a same-shape historical row in data/consent-unclear-register.yaml
# (added by hand by task 245, `robots_train_tokens_blocked: [ccbot]`,
# rationale "AI-training-crawler block only; `*` open") -- exactly the class
# the automatic gate writer this test exercises was silently missing.
NUMERAMA_ROBOTS_TXT = """User-agent: *

Disallow: /search/
Disallow: /search
Disallow: /wp/wp-admin/
Disallow: /wp/wp-includes/
Disallow: /wp-content/plugins/
Disallow: /numeramaplus/refresh/
Disallow: /wp-content/themes/numerama-next/misc/
Disallow: /cdn-cgi/
Disallow: /plsbl/
Disallow: /wp-content/mu-plugins/humanoid-core/includes/Core/Ajax/
Disallow: /*?show_reader_reports*
Disallow: /*&show_reader_reports*
Disallow: /actualites/api
Disallow: /actualites/feed/
Disallow: /actualites/api/feed/

Allow: /videos/$
Allow: /videos/page/*
Disallow: /videos/*/page/*
Disallow: /videos/*/

User-agent: GPTBot
Allow: /

User-agent: OAI-SearchBot
Allow: /

User-agent: ChatGPT-User
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Googlebot
Allow: /

User-agent: CCBot
Disallow: /

User-agent: SemjiBot
Disallow: /

User-agent: AmazonAdBot
Allow: /

Disallow: /ajax-postviews.php

Sitemap: https://www.numerama.com/sitemap_index.xml
Sitemap: https://www.numerama.com/sitemap-news.xml
"""


def _numerama_receipt():
    rec = {"host": "www.numerama.com", "status": "ok",
           "search_ok": True, "train_ok": True, "reasons": [],
           "checked_at": "2026-09-08T00:00:00Z", "signals": {}}
    sig = CR._robots_signals(NUMERAMA_ROBOTS_TXT)
    CR.apply_robots_flags(rec, sig)
    rec["signals"]["robots"] = sig
    return rec


def test_a_real_ccbot_only_host_carries_no_train_tokens_disallowed():
    """The real defect, reproduced: CCBot's own robots block never lands in
    `train_tokens_disallowed` -- that field is populated from TRAIN_TOKENS
    (gptbot, anthropic-ai, claudebot, claude-web, cohere-ai, bytespider,
    meta-externalagent) intersected with tokens the body actually uses, and
    CCBot is deliberately not in that set (its flag comes from the dedicated
    `ccbot_root` check instead). numerama.com's robots.txt explicitly ALLOWS
    GPTBot and mentions no Claude-family token at all, so this is not a
    fixture artifact -- it is what the real file says."""
    rec = _numerama_receipt()
    assert rec["signals"]["robots"]["ccbot_root"] is False
    assert rec["signals"]["robots"]["train_tokens_disallowed"] == []
    assert (rec["search_ok"], rec["train_ok"]) == (True, False)


def test_unclear_search_tokens_folds_in_ccbot():
    rec = _numerama_receipt()
    assert CR.unclear_search_tokens(rec) == ["ccbot"]


def test_a_real_ccbot_only_host_now_qualifies_as_unclear():
    """Before task 542: False (empty token list). After: True -- and the
    register writer (`consent_gate.py`) now has a non-empty token list to
    log, so amendment 2's own request ("logged in that register... with the
    token") is finally met for this class of host."""
    rec = _numerama_receipt()
    assert CR.unclear_search_ok(rec) is True


def test_a_claude_family_block_alongside_ccbot_still_declines():
    """The Claude-family carve-out must survive the ccbot fold-in: a host
    blocking BOTH ccbot and a Claude-family token is still a decline, not
    unclear -- ccbot never bypasses amendment 2's own exception."""
    rec = _numerama_receipt()
    rec["signals"]["robots"]["train_tokens_disallowed"] = ["claudebot"]
    assert CR.unclear_search_tokens(rec) == ["claudebot", "ccbot"]
    assert CR.unclear_search_ok(rec) is False


def test_the_register_writer_now_logs_the_ccbot_token(tmp_path):
    """End-to-end through the actual register-writing shape used by
    `consent_gate.py`'s CLI `main()` (host -> entry, keyed off
    `unclear_search_tokens`) over a REAL resolved receipt, not a hand-built
    one -- proving the token this test is about actually reaches the entry
    a real gate run would append, not just the classifier function."""
    rec = _numerama_receipt()
    verdict = {"www.numerama.com": "ok_search_only"}
    res = {"www.numerama.com": rec}
    reg_hosts = []
    for h, b in sorted(verdict.items()):
        if b != "ok_search_only" or not CR.unclear_search_ok(res[h]):
            continue
        rb = res[h]["signals"]["robots"]
        entry = {
            "host": h,
            "signals_observed": "; ".join(res[h]["reasons"]),
            "robots_train_tokens_blocked": sorted(CR.unclear_search_tokens(res[h])),
            "wildcard_root_allowed": bool(rb.get("wildcard_root", True)),
        }
        reg_hosts.append(entry)
    assert reg_hosts[0]["robots_train_tokens_blocked"] == ["ccbot"]
