#!/usr/bin/env python3
# Moved from the retired `payload_carry_contract` script (task 677, task 643 S5); the old path is an alias shim.
"""The payload a re-materialised row must CARRY, enforced where it is composed.

WHY THIS EXISTS (task 584, from task 583 §3)
--------------------------------------------
Task 580 re-materialised 187,222 rows of restored page text into the candidate
and every one of them reached the collection with **no `title`, no
`trust_tier`, no `lang` and no `domain`** -- 0% present, against 100% in the
live collection and 100% in task 579's tranche.  Both retrieval legs were
unchanged or better; the golds fell because the FUSION reads those fields
(`_title_centrality("")` is 0 against a weight of 2.5, and `_trust_tier(None)`
is the documented 0.6 unknown default rather than the register's 1.0).  The
lane then concluded against its own hypothesis on rows that were never
scoreable.

THE MECHANISM, named rather than summarised.  `materialize_565.harvest_payloads`
reads the incumbent payload with `with_payload=True` and is correct.  But
`materialize_565.main()` prefers an existing `--payload-cache` file over
harvesting, and task 580 wrote that cache from
`roster_579.cand_points()`, whose scroll body asks for

    "with_payload": ["url", "source_type", "anchor", "document_id"]

-- a four-field PROJECTION written for roster resolution, where it is right.
Reused as the source of the carry, it left `base = {k: payload[k] for k in
CARRY if payload.get(k) is not None}` with `source_type` and nothing else.  The
materialiser ran green, the conservation table balanced, the ids were right,
the row counts were right, and the meaning of every row was gone
(corpus-doctrine 1, 2).

So the contract moves here, out of any one lane's script, and it refuses:

  * `carry_base()` is the ONE composition of a carried payload, and it raises
    on a payload that cannot satisfy the family's required fields.  There is no
    partial carry: a row that cannot state its own title, language, domain,
    source type and trust tier is a build defect, not a row.
  * `assert_payload_cache()` refuses the exact artefact that broke -- a cache
    whose entries are a projection -- BEFORE a materialiser consumes it.
  * `assert_scoreable()` refuses a whole staged population against the same
    floors `gate_payload_contract.py` applies at the apply, so the refusal
    happens before the GPU rather than after it.  `stage_pack_b1.py` calls it.
  * `assert_full_payload_request()` refuses the field-list read itself, at the
    seam where a projection can be substituted for a payload.

None of these has a `--force`.  Waiving a contract is a legislative act that
belongs to whoever owns the standard, never a step inside the run that needs
excusing (verification-doctrine 12).

    python3 payload_carry_contract.py --cache /path/to/payloads.jsonl.gz
    python3 payload_carry_contract.py --records '/path/to/*.records.jsonl.gz'
"""
from __future__ import annotations

import argparse
import glob as _glob
import gzip
import json
import os
import pathlib
import sys


from cytir.core import projection as PROJ  # noqa: E402

#: Page-level payload fields carried from the superseded row.  Imported from
#: the projection rather than retyped so the two cannot drift -- the same
#: reasoning, and the same tuple, `materialize_565.CARRY` has used since task
#: 565.  This module is now its one home in `pipeline/`.
CARRY = tuple(PROJ.BASE_PASSTHROUGH) + tuple(PROJ.CONSENT_KEYS) + ("license",)

#: NOT carried, on purpose (materialize_565's reasons, kept verbatim in
#: intent): `anchor` is the A33 identity trap -- a PAGE-level anchor handed to
#: `classify_tier` collapses every chunk of the page onto one point.  `snippet`
#: is recomputed by `project_record` from the row's own text; carrying the
#: superseded row's snippet pins a sentence from a different passage onto every
#: new one.
NOT_CARRIED = ("anchor", "snippet")

#: The fields the SCORER reads off the payload and a re-cut row therefore may
#: not lose.  This is the subset of CARRY whose absence changes what a row
#: MEANS to `retrieval.search()` rather than merely what it reports:
#:
#:   title       `_title_centrality` (weight 2.5) and `_exact` both read it
#:   lang        tokenisation for title-centrality and German de-compounding
#:   domain      the academic / parasite / machine-url / hub tables key on it
#:   source_type intent gating, entity-card handling, official preference
#:   trust_tier  absent => `_trust_tier`'s 0.6 unknown default, not the
#:               register's value (-1.2 fused on a tier-1.0 host)
REQUIRED_CARRY = ("title", "lang", "domain", "source_type", "trust_tier")

#: Presence floors for a STAGED population, per field.  Same numbers as the
#: apply-time gate, and deliberately the same object: two enforcement surfaces
#: that can disagree about the standard are one surface and one decoration.
#: `country` is not universal in the corpus (94.2% on a uniform sample of
#: aw_e8) so its floor is set where a real regression is still visible without
#: failing honest international rows; `published` is family-specific and is not
#: floored corpus-wide (see `FAMILY_REQUIRED`).
CONTRACT = {
    "url": 1.0,
    "source_type": 1.0,
    "title": 0.99,
    "trust_tier": 0.99,
    "lang": 0.99,
    "domain": 0.99,
    "snippet": 0.95,
    "country": 0.80,
}

#: The floors for a MATERIALISED RECORD, before `project_record` runs.  It is
#: the CARRIED subset of CONTRACT and nothing else: `snippet` is recomputed by
#: the projection from the row's own text and is absent by design at this
#: stage (it is in NOT_CARRIED), and `country` is re-derived there too.  Using
#: the serving CONTRACT on a pre-projection record would fail every honest
#: pack on two fields the record is not supposed to have yet -- a gate that
#: refuses correct work is as useless as one that passes everything, and it
#: teaches its operator to reach for a waiver.
RECORD_CONTRACT = {f: CONTRACT[f] for f in
                   ("url", "source_type", "title", "trust_tier", "lang",
                    "domain")}

#: Fields a NAMED source family must additionally carry.  A family whose
#: records are dated and whose serving path applies a recency penalty cannot
#: lose its date silently: `pen.recency` then reads an undated row as neutral.
#: Keyed on `source_type`; a family not named here is governed by CONTRACT
#: alone, and adding a key is a declaration, not a default.
FAMILY_REQUIRED = {
    "news": ("published",),
}


class PayloadCarryError(RuntimeError):
    """A payload that cannot be carried, or a population that cannot be scored."""


# ------------------------------------------------------------------ compose
def missing_required(payload: dict,
                     required: tuple = REQUIRED_CARRY) -> tuple:
    """The required fields this payload cannot supply, in declaration order."""
    payload = payload or {}
    out = [f for f in required if payload.get(f) is None]
    fam = FAMILY_REQUIRED.get((payload.get("source_type") or "").strip().lower())
    if fam:
        out += [f for f in fam if payload.get(f) is None]
    return tuple(out)


def carry_base(payload: dict, *, url: str = "",
               required: tuple = REQUIRED_CARRY) -> dict:
    """The carried half of a re-materialised row, or raise.

    This is the only place `{k: payload[k] for k in CARRY}` is written.  It
    raises rather than returning a thin dict, because a partial carry is
    indistinguishable downstream from a page that genuinely has no title --
    which is precisely how 187,222 rows reached a gate that could not see it.
    """
    miss = missing_required(payload, required)
    if miss:
        have = sorted((payload or {}).keys())
        raise PayloadCarryError(
            f"payload for {url or (payload or {}).get('url') or '<no url>'} "
            f"cannot carry {list(miss)}; it holds {have}. A payload with only "
            f"a handful of keys is usually a PROJECTION -- a scroll or "
            f"retrieve that asked for a field list instead of "
            f"`with_payload=True` (task 580). Re-read the source row in full; "
            f"do not materialise a row that cannot state what it is.")
    base = {k: payload[k] for k in CARRY if payload.get(k) is not None}
    for k in NOT_CARRIED:
        base.pop(k, None)
    return base


def assert_full_payload_request(with_payload, *, where: str = "") -> None:
    """Refuse a field-list read where a full payload is required.

    The seam itself, guarded at the seam.  `with_payload=True` is the only
    acceptable value when the result will be CARRIED; a list is a projection
    and a projection silently becomes the row.
    """
    if with_payload is not True:
        raise PayloadCarryError(
            f"{where or 'payload read'}: with_payload={with_payload!r} is a "
            f"PROJECTION. A payload that will be carried onto a new row must "
            f"be read with with_payload=True (task 580 carried a four-field "
            f"projection onto 187,222 rows).")


# ------------------------------------------------------------------ populations
#: Fields whose presence is set by a RATIFIED POLICY rather than being universal,
#: with the policy that decides it.  `country` is stamped by
#: `aw_country.country_for_record` (Tim's 2026-07-17 content-conditioned
#: jurisdiction policy) only on location-bound content, INTL domains, reference,
#: and registered entity cards; every other general row -- docs, news, tech --
#: is deliberately UNSTAMPED ("neither boosted nor demoted").
#:
#: So for `country` a row CONFORMS when it carries one, or when the policy stamps
#: none for it.  Counting bare presence instead made the 0.80 floor -- calibrated
#: on the whole mixed corpus -- refuse every single-family pack of an unstamped
#: class: task 629's B01 (445 `docs` rows, sunfounder/upesy/toybox) read 0.0%
#: against a collection whose own `docs` rows are 1,520,614 of 1,522,465
#: (99.88%) country-less, measured on the aw_e8 twin 2026-09-23.  A floor that
#: refuses what the policy prescribes is not a stricter floor; it is a wrong one.
#:
#: The regression the floor exists to catch is still caught: a row the policy
#: WOULD stamp (a `.ch` court, an entity card with a register id) that arrives
#: without a country does not conform.
POLICY_CONDITIONED = {
    "country": "present, or absent where aw_country.country_for_record "
               "stamps none",
}


def _conforms(field: str, pl: dict) -> bool:
    """Present, or -- for a policy-conditioned field -- correctly absent."""
    if pl.get(field) is not None:
        return True
    if field != "country":
        return False
    from cytir.core import country as CT  # noqa: PLC0415
    return CT.country_for_record(pl.get("domain"), pl.get("source_type"),
                                 pl.get("register_id"), pl.get("lei")) is None


def census(payloads, contract: dict = None) -> tuple:
    """(n, {field: n_conforming}) over an iterable of payload dicts.

    Conforming is plain presence for every field except the ones named in
    `POLICY_CONDITIONED`, where it is presence-or-policy-absence.
    """
    contract = CONTRACT if contract is None else contract
    n, have = 0, {f: 0 for f in contract}
    for pl in payloads:
        pl = pl or {}
        n += 1
        for f in contract:
            if _conforms(f, pl):
                have[f] += 1
    return n, have


def verdict(n: int, have: dict, *, source: str = "",
            contract: dict = None) -> dict:
    """The same verdict shape `gate_payload_contract.py` seals."""
    contract = CONTRACT if contract is None else contract
    rows, failures = [], []
    for f, floor in sorted(contract.items()):
        rate = (have.get(f, 0) / n) if n else 0.0
        ok = n > 0 and rate >= floor
        row = {"field": f, "present": round(rate, 4), "floor": floor, "ok": ok}
        if f in POLICY_CONDITIONED:
            row["counts"] = POLICY_CONDITIONED[f]
        rows.append(row)
        if not ok:
            failures.append(f"{f} {rate:.1%} < {floor:.0%}")
    return {"schema": "aw.payload-carry.v1", "source": source,
            "rows_checked": n, "fields": rows, "failures": failures,
            "verdict": "PASS" if (n and not failures) else "FAIL"}


def assert_scoreable(payloads, *, label: str = "",
                     contract: dict = None) -> dict:
    """Refuse a population the fusion could not score.  Returns the verdict.

    An EMPTY population is a failure, not a pass: reporting green over ground
    you could not read is the defect this module exists to stop
    (verification-doctrine 6).
    """
    n, have = census(payloads, contract)
    v = verdict(n, have, source=label, contract=contract)
    if v["verdict"] != "PASS":
        raise PayloadCarryError(
            f"payload carry FAILED on {label or 'this population'}: "
            + ("the population holds 0 rows -- an empty read is an error, not "
               "a pass" if not n else "; ".join(v["failures"]))
            + ". A row with no `title` scores title-centrality 0 against a "
              "weight of 2.5, and a row with no `trust_tier` is scored at the "
              "0.6 unknown default instead of the register's value.")
    return v


# ------------------------------------------------------------------ artefacts
def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8")


def iter_cache(path: str):
    """`{url, payload}` lines of a materialiser payload cache."""
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                yield d.get("url", ""), (d.get("payload") or {})


def assert_payload_cache(path: str, *, required: tuple = REQUIRED_CARRY,
                         max_report: int = 5) -> dict:
    """Refuse a payload cache whose entries are a projection.

    Task 580's cache is the case: `{"url": ..., "payload": {"url": ...,
    "source_type": ...}}` on 5,000 of 5,000 sampled roster-B pages.  A
    materialiser that prefers a cache over harvesting will carry exactly that.
    """
    if not os.path.exists(path):
        raise PayloadCarryError(
            f"payload cache {path} does not exist -- an absent input is a "
            f"failure, not a skip")
    n = 0
    bad: list = []
    for url, pl in iter_cache(path):
        n += 1
        miss = missing_required(pl, required)
        if miss:
            if len(bad) < max_report:
                bad.append({"url": url, "missing": list(miss),
                            "has": sorted(pl.keys())})
            else:
                bad.append(None)
    if not n:
        raise PayloadCarryError(
            f"payload cache {path} holds 0 entries -- an empty cache would "
            f"make the materialiser harvest silently, or carry nothing")
    n_bad = len(bad)
    if n_bad:
        shown = [b for b in bad if b]
        raise PayloadCarryError(
            f"payload cache {path}: {n_bad:,} of {n:,} entries cannot carry "
            f"{list(required)}. First {len(shown)}: "
            f"{json.dumps(shown, ensure_ascii=False)}. This is the shape task "
            f"580 wrote: a scroll that asked for a field list instead of "
            f"`with_payload=True`.")
    return {"schema": "aw.payload-carry-cache.v1", "cache": path,
            "entries": n, "verdict": "PASS"}


def iter_records(patterns) -> "object":
    """Payload-shaped dicts from materialised record shards."""
    paths: list = []
    for p in (patterns if isinstance(patterns, (list, tuple)) else [patterns]):
        paths.extend(sorted(_glob.glob(p)) or ([p] if os.path.exists(p) else []))
    if not paths:
        raise PayloadCarryError(
            f"no record file matched {patterns!r} -- an error is not an empty")
    for path in paths:
        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", help="a materialiser --payload-cache file")
    ap.add_argument("--records", action="append",
                    help="materialised record shard(s); glob allowed")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    if not a.cache and not a.records:
        print("[payload-carry] FAIL: give --cache or --records",
              file=sys.stderr)
        return 2
    out: dict = {"schema": "aw.payload-carry-report.v1"}
    try:
        if a.cache:
            out["cache"] = assert_payload_cache(a.cache)
        if a.records:
            recs = list(iter_records(a.records))
            # RECORD_CONTRACT, not CONTRACT: these are pre-projection rows.
            out["records"] = assert_scoreable(recs, label=str(a.records),
                                              contract=RECORD_CONTRACT)
            out["records"]["n_records"] = len(recs)
    except PayloadCarryError as exc:
        print(f"[payload-carry] FAIL: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"[payload-carry] FAIL: cannot read the input "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
