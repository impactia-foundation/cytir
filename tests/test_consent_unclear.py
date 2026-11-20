#!/usr/bin/env python3
"""Offline tests for the ADR-024 amendment (2026-07-13) unclear-consent
class: consent_resolver.unclear_search_ok + bucket(unclear_ok=True).

  python test_consent_unclear.py
"""
import cytir.consent.resolver as cr


def rec(search_ok=True, train_ok=False, toks=(), wildcard=True,
        cu=None, cs=None, status="ok", grant=False):
    return {"host": "x.test", "status": status,
            "search_ok": search_ok, "train_ok": train_ok,
            "reasons": [], "checked_at": "2026-07-13T00:00:00Z",
            "signals": {"robots": {
                "wildcard_root": wildcard,
                "train_tokens_disallowed": list(toks),
                "content_usage": cu or {},
                "content_signal": cs or {},
                "content_signal_search_grant": grant}}}


CASES = [
    # (description, record, unclear_search_ok, bucket(unclear_ok=True))
    ("tbray class: google-extended only",
     rec(toks=["google-extended"]), True, "ok_search_only"),
    ("jvns class: gptbot+google-extended",
     rec(toks=["google-extended", "gptbot"]), True, "ok_search_only"),
    ("claudebot block = decline",
     rec(toks=["claudebot", "gptbot"]), False, "declined"),
    ("anthropic-ai block = decline",
     rec(toks=["anthropic-ai"]), False, "declined"),
    ("claude-web block = decline",
     rec(toks=["claude-web", "google-extended"]), False, "declined"),
    ("netflixtechblog class: incl. claudebot = decline",
     rec(toks=["applebot-extended", "bytespider", "claudebot", "gptbot",
               "meta-externalagent"]), False, "declined"),
    ("non-claude big list = unclear ok",
     rec(toks=["applebot-extended", "bytespider", "gptbot",
               "meta-externalagent"]), True, "ok_search_only"),
    ("wildcard decline (search_ok False) stays declined",
     rec(search_ok=False, toks=["gptbot"]), False, "declined"),
    ("both flags ok unaffected",
     rec(train_ok=True, toks=[]), False, "ok"),
    ("AIPREF ai=n explicit opt-out = decline",
     rec(toks=["gptbot"], cu={"ai": "n"}), False, "declined"),
    ("Content-Signal ai-train=no explicit = decline (unless grant path)",
     rec(toks=["gptbot"], cs={"ai-train": "no"}), False, "declined"),
    ("train_ok=False w/o tokens (e.g. tdm-only would also flip search) ",
     rec(toks=[]), False, "declined"),
    ("unknown host held",
     rec(status="unknown", toks=["gptbot"]), False, "unknown"),
]


def main():
    bad = 0
    for desc, r, want_unclear, want_bucket in CASES:
        gu = cr.unclear_search_ok(r)
        gb = cr.bucket(r, unclear_ok=True)
        gs = cr.bucket(r)      # strict rule unchanged when flag off
        ok = (gu == want_unclear and gb == want_bucket)
        if gs == "ok_search_only":
            ok = False         # strict mode must never admit via unclear
        if not ok:
            bad += 1
            print(f"FAIL {desc}: unclear={gu} (want {want_unclear}) "
                  f"bucket={gb} (want {want_bucket}) strict={gs}")
    # grant path still works independently
    g = rec(toks=["gptbot"], cs={"ai-train": "no"}, grant=True)
    assert cr.bucket(g, accept_search_grant=True) == "ok_search_only"
    assert cr.bucket(g) == "declined"
    print(f"{len(CASES) - bad}/{len(CASES)} unclear-class cases pass"
          + ("" if not bad else "  <-- FAILURES"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
