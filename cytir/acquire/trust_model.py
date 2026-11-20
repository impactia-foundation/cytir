#!/usr/bin/env python3
# Moved from the retired `trust_model` module (task 683 S7 tranche 4); the old path is an alias shim.
"""Earned-vs-inherited trust for discovery-admitted hosts (task 440).

Split out of discovery_ingest.py so that measurement, tests and any retroactive
re-scoring import the SAME code that ships, instead of a copy of it. This module
deliberately has no dependencies beyond the stdlib: discovery_ingest pulls in
httpx/trafilatura/qdrant, and nothing that merely wants to price a trust tier
should have to.

THE DEFECT THIS EXISTS TO FIX
-----------------------------
`prep_discovery_queue.classify()` grants a candidate host the class of any seed
that links it, and `discovery_ingest` then stamped `CLASS_TRUST[cls]` -- the
same number a host earns by carrying the credential in its own name. So one
inbound link from one university page made heathrow.com's travel blog `edu@0.8`
(task 203, live incident 2026-07-13). The endorsement graph's own strength
measure was computed two lines above the point where it was discarded:

    strength = min(float(cnt), SEED_VOTE_CAP)
    seed_host[h] += strength
    if sc:
        host_classes[h].add(sc)     # <-- a set; strength gone

Task 440 measured the consequence on the graph that built the live population:
90,598 hosts hold a gov/health/edu/ref class purely by inheritance against
12,479 that earned it by self-signal, and 75.4% of the inherited ones rest on a
single endorsing registrable domain.

THE RULE
--------
POLICY.md v2 §4.4 (ratified PR #49, Tim 2026-07-27) already decides this:
"Discovery-sourced admissions ... never inherit a mission-content class default.
A discovery-admitted host receives a mission-tier trust class only by
affirmative classification; absent that, it enters at the vendor-content class
(0.60) or below." An inbound link is not affirmative classification.

So an endorsed class is priced from the vendor-content floor, and corroboration
by INDEPENDENT endorsing registrables raises it only within a sub-mission band,
never to the self-signal value. This is ADR-020 §4's own model (authority =
Σ seed_trust × endorsement_strength, decayed by hop distance) finally applied to
trust rather than only to queue ordering.

Deliberately NOT done here: changing the class. The class is a useful topical
facet, and rewriting it would move `source_type` with it, which is the task-263
/ ADR-054 error class (routing health -> official made globally authoritative
content country-stampable). Only the unearned trust is removed -- "the fix is
removing unearned trust, not adding it" (ADR-019 close, decisions.md:2816).

WHY CORROBORATION IS A FLOOR-RAISER AND NOT A CLASSIFIER
--------------------------------------------------------
Task 440 checked corroboration against held-out chaff and it does NOT separate
good from bad on its own: chordie.com (guitar-chord aggregator) carries 11
independent university endorsements, while git-scm.com and sqlite.org carry 1.
Aggregators are popular, so they win link counts. The endorsement graph can tell
you a host is linked; it cannot tell you the host is a primary authority. The
ceiling is therefore the load-bearing part of this rule and the corroboration
term is a minor modulation inside a band that is already below every mission
tier -- which is exactly why the band's top matters less than its existence.
"""
from __future__ import annotations

import math

INHERITED_FLOOR = 0.60      # POLICY v2 §4.4 vendor-content class
INHERITED_CEILING = 0.70    # strictly below CLASS_TRUST["tech"]=0.75, the lowest
                            # mission tier reachable through the graph fallback
INHERITED_GAIN = 0.04       # per doubling of INDEPENDENT endorsing registrables

SELF_SIGNAL = "self-signal"
ENDORSED = "endorsed"


def inherited_trust(n_endorser_regs: int | None) -> float:
    """Trust for a class held only by endorsement, given how many DISTINCT
    endorsing registrable domains support it.

    Registrables, not links and not seed hosts: one endorser's many links are
    one voice (the reason `ENDORSER_HOST_CAP` exists in prep_discovery_queue --
    one seed once contributed 4,912 URLs to a single target), and one endorser's
    many hosts are one voice too.

    A single endorsement lands exactly on the vendor-content class, which is
    POLICY v2 §4.4's letter. It takes 8 independent endorsers to reach the
    ceiling, and the ceiling is still below every mission tier."""
    n = max(int(n_endorser_regs or 1), 1)
    return round(min(INHERITED_CEILING,
                     INHERITED_FLOOR + INHERITED_GAIN * math.log2(n)), 3)


def resolve_trust(base_trust: float, class_basis: str | None,
                  n_endorser_regs: int | None) -> float:
    """Price a discovery candidate's trust tier.

    `base_trust` is the class's self-signal value (CLASS_TRUST[cls]).
    `class_basis` is SELF_SIGNAL when the hostname itself carries the credential
    and ENDORSED when only the link graph does.

    An unknown basis is treated as ENDORSED: a queue built before task 440
    carries no basis field, and the safe reading of a missing credential is that
    there is none. The result is never RAISED above base_trust -- a class that
    already sits below the inherited band (news 0.6, other 0.55) does not get
    promoted for having been linked."""
    if class_basis == SELF_SIGNAL:
        return base_trust
    return round(min(base_trust, inherited_trust(n_endorser_regs)), 3)
