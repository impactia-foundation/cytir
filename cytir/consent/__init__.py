"""cytir.consent -- the one consent resolver and its guards (task 677, task 643 S5).

    resolver.py    live per-host AIPREF / robots / TDMRep resolution (moved from
                   `cytir/consent/resolver.py`)
    block.py       recorded-decline guard, fails closed (moved from
                   `cytir/consent/block.py`)
    policy.py      the search-index admission rubric (moved from
                   the retired `consent_policy` module)
    embed_gate.py  the same recorded-decline refusal, enforced at the embed stage
    derive.py      the ONE place per-record search_ok/train_ok are computed (ADR-115)
    backfill.py    dry-run reader for that derivation over a twin (never writes)
    arxiv_licence.py  per-paper arXiv licences from the L0 OAI pages (ADR-115 r1)

The crawler identity the resolver sends lives in `cytir.core.user_agent`.
Old `consent_*` script paths are shims that ALIAS the package module
(`sys.modules`), so `import consent_resolver` and `cytir.consent.resolver` are
one module object and a test's monkeypatch reaches the code that runs.
The corpus batch gate is `cytir.consent.gate` (`python -m cytir.consent.gate`).
"""
