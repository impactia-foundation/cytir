#!/usr/bin/env python3
"""The chunk-policy sidecar registry: status and family, keyed by sha256.

ADR-090 §6 ordered this mechanism and it was never built (verified absent
2026-09-08, task 541). A policy artifact's sha256 IS its identity -- pinned by
`build_config.json`, by every MANIFEST and by the shipped corpus -- and
`chunk_policy.load_chunk_policy` validates an EXACT key set, so neither a
`status` nor a `family` field can live in the artifact: adding one would change
the hash and make the file unloadable. They live here, in a sidecar consulted at
SELECTION time only.

WHAT THIS REFUSES
-----------------
`assert_selectable(policy)` refuses a policy whose registry status is not
`current`. A superseded artifact stays byte-identical and loadable forever (I2
rule 2, ADR-039) -- historical replay must keep working -- but a NEW
materialization may not pick one up. Replay passes `replay=True` and says so in
its manifest. `family_id=` admits a `proposed` policy for the one family an ADR
named in `admitted_for_families` (ADR-113: ERIC's 1024w arm, ERIC only) --
narrower than flipping `status` to `current`, which would change the shared
default for every other family in the same `policy_family`.

`family_of(policy)` is the one home for the question "which chunker family does
this policy serve?" (`specs/epochs.md` §5: 'if a third chunker family appears,
split the family formally'). Before this file, that question had no
machine-readable answer at all, which is why a 3.73x median granularity change
swept 82 hosts with nothing in the repo able to see it (task 535 §2.4/§6e).

FAIL LOUD (engineering-doctrine 3): a policy that is not in the registry raises.
It does not default to `current`, and it does not default to `chunk.generic`.
An unregistered policy is exactly the state this module exists to detect.
"""
from __future__ import annotations

from cytir.core.paths import data_path

import json
import pathlib
from typing import Any

SCHEMA = "aw.chunk-policy-registry.v1"
DEFAULT_PATH = "data/chunk-policy-registry.v1.json"
REPO = pathlib.Path(__file__).resolve().parents[2]   # maintainer checks only (verify(repo=...)); readers use data_path

#: The only status a NEW materialization may select. `proposed` is deliberately
#: NOT selectable: an unratified policy that could be picked up by a lane is an
#: in-band waiver (verification-doctrine 12).
SELECTABLE = frozenset({"current"})

#: What a MEASUREMENT arm may select (task 569's width sweep). An experiment
#: that materializes a candidate width has to cut pages under a policy nobody
#: has ratified -- that is the whole point of measuring it -- and the three ways
#: to express that were: lie (`replay=True`, which this is not), flip the arm to
#: `current` before it has been measured (an in-band waiver, verification-
#: doctrine 12), or say so explicitly. This is saying so.
#:
#: It is NOT a softer `SELECTABLE`: it still refuses `superseded` and
#: `special-purpose` and an unregistered policy, so the only thing it adds is
#: `proposed` -- a policy that exists, is declared, has an epoch and a family,
#: and has not been ruled on. A caller passing `experiment=True` is asserting it
#: writes to no serving or candidate collection; the registry cannot check that,
#: which is why the flag is explicit and named rather than a default.
EXPERIMENT_SELECTABLE = frozenset({"current", "proposed"})


class ChunkPolicyRegistryError(Exception):
    """A policy is unregistered, or its registry entry contradicts the file."""


def load(path: str | pathlib.Path | None = None) -> dict:
    resolved = pathlib.Path(path or data_path(pathlib.Path(DEFAULT_PATH).name))
    try:
        doc = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkPolicyRegistryError(
            f"cannot load chunk-policy registry {resolved}: {exc}") from exc
    if doc.get("schema") != SCHEMA:
        raise ChunkPolicyRegistryError(
            f"{resolved}: schema {doc.get('schema')!r} is not {SCHEMA!r}")
    for key in ("policy_families", "policies", "epoch_counter"):
        if not isinstance(doc.get(key), dict):
            raise ChunkPolicyRegistryError(f"{resolved}: missing {key!r} block")
    return doc


def entry(sha256: str, doc: dict | None = None) -> dict[str, Any]:
    """The registry entry for one policy digest. Raises when unregistered."""
    doc = doc if doc is not None else load()
    got = doc["policies"].get(sha256)
    if got is None:
        raise ChunkPolicyRegistryError(
            f"chunk policy sha256 {sha256} is not in {DEFAULT_PATH}. An "
            "unregistered policy has no family and no status; declare it "
            "(ADR PR) rather than defaulting it.")
    return got


def family_of(sha256: str, doc: dict | None = None) -> str:
    doc = doc if doc is not None else load()
    fam = entry(sha256, doc).get("policy_family")
    known = sorted(doc["policy_families"])
    if fam not in doc["policy_families"]:
        raise ChunkPolicyRegistryError(
            f"policy {sha256} declares policy_family {fam!r}, which is not "
            f"one of {known}")
    return fam


def status_of(sha256: str, doc: dict | None = None) -> str:
    return entry(sha256, doc).get("status") or "unknown"


def epoch_word_width(epoch: int, doc: dict | None = None) -> int | None:
    """The chunk width in WORDS that `rule_epochs.chunk == epoch` rows were cut at, read from
    the registered policy artifacts (`window_words` for a sliding window, `sizing.soft_target`
    for a structured packer) -- never hard-coded by a caller. `None` when the epoch has no
    word-width policy or its policies declare more than one width (the epoch int is shared
    across families, so an ambiguous epoch is unknown, not guessed). Task 984."""
    doc = doc if doc is not None else load()
    widths: set[int] = set()
    for ent in doc["policies"].values():
        if ent.get("epoch") != epoch or not ent.get("file"):
            continue
        try:
            art = json.loads(data_path(pathlib.Path(ent["file"]).name).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        sizing = art.get("sizing") or {}
        w = art.get("window_words")
        if w is None and sizing.get("unit") == "words":
            w = sizing.get("soft_target")
        if isinstance(w, int) and not isinstance(w, bool) and w > 0:
            widths.add(w)
    return widths.pop() if len(widths) == 1 else None


def assert_selectable(policy, *, replay: bool = False,
                      experiment: bool = False,
                      family_id: str | None = None,
                      doc: dict | None = None) -> str:
    """Refuse a policy a NEW materialization may not select.

    `policy` is a `chunk_policy.ChunkPolicy` / `LegacyChunkPolicy` (anything
    with `.sha256` and `.policy_id`). Returns the family on success.

    `replay=True` reproduces rows that already exist and ignores status.
    `experiment=True` additionally admits a `proposed` policy for a
    MEASUREMENT arm that writes to no collection (see EXPERIMENT_SELECTABLE);
    it still refuses `superseded`, `special-purpose` and unregistered.

    `family_id`, if given, additionally admits a policy whose status is
    EXACTLY `proposed` (never `superseded`/`special-purpose`/anything else)
    and whose registry entry names that exact family in
    `admitted_for_families` -- ADR-113 (2026-09-25): ERIC's 1024w
    `chunk.generic` arm is ratified selectable for `eric_official_pdf`'s own
    real extraction ONLY, without flipping `chunk.generic`'s shared default
    (still epoch 4) for every other family that uses it. This is narrower
    than `experiment=True`: it admits a real production materialization, but
    only for the family(ies) an ADR actually named, never as a side effect of
    the policy merely existing. The `status == "proposed"` guard matters: an
    arm that is later superseded (a wider width wins a held-out eval, say)
    must NOT keep being selectable through a stale `admitted_for_families`
    entry nobody remembered to strip -- that would defeat this module's whole
    reason to exist (see WHAT THIS REFUSES above).
    """
    doc = doc if doc is not None else load()
    sha = getattr(policy, "sha256", None) or str(policy)
    ent = entry(sha, doc)
    fam = family_of(sha, doc)
    if replay:
        return fam
    allowed = EXPERIMENT_SELECTABLE if experiment else SELECTABLE
    status = ent.get("status")
    if status not in allowed:
        if (status == "proposed" and family_id
                and family_id in (ent.get("admitted_for_families") or [])):
            return fam
        successor = ent.get("successor_sha256")
        adr = ent.get("superseded_by_adr")
        raise ChunkPolicyRegistryError(
            f"chunk policy {ent.get('policy_id')} (sha {sha[:12]}...) has "
            f"registry status {status!r} and may not be selected for a new "
            f"materialization"
            + (" (experiment=True admits `proposed`, not this)"
               if experiment else "")
            + (f"; successor {successor[:12]}..." if successor else "")
            + (f"; {adr}" if adr else "")
            + ". Pass replay=True only to reproduce rows that already exist, "
              "experiment=True only for a measurement arm that writes to no "
              "collection.")
    return fam


def selector_entry(sha256: str, doc: dict | None = None) -> dict[str, Any]:
    """The registry entry for one SELECTOR artifact. Raises when unregistered.

    A selector is the other half of a chunk policy: the width knobs say how a
    page is cut, the selector says which pages. ADR-090 amendment 1 changed the
    second half without touching the first, which is exactly the kind of rule
    change that had no machine-readable home before this block existed.
    """
    doc = doc if doc is not None else load()
    got = ((doc.get("selectors") or {}).get("artifacts") or {}).get(sha256)
    if got is None:
        raise ChunkPolicyRegistryError(
            f"selector sha256 {sha256} is not registered in {DEFAULT_PATH}. "
            "An unregistered selector has no status and no family; declare it "
            "(ADR PR) rather than defaulting it.")
    return got


def assert_selector_selectable(config_path: str | pathlib.Path, *,
                               replay: bool = False,
                               experiment: bool = False,
                               doc: dict | None = None) -> dict[str, Any]:
    """Refuse a selector artifact a NEW materialization may not select.

    Same shape and same reason as `assert_selectable` for policies: a
    superseded selector stays byte-identical and reproducible forever, and a
    new run may not pick it up by accident. Returns the registry entry.

    `experiment=True` admits a `proposed` SELECTOR for a measurement arm that
    writes to no collection, exactly as `assert_selectable` does for a proposed
    policy and for the same reason (EXPERIMENT_SELECTABLE): task 575 has to
    classify pages under a P3 rule nobody has ratified -- that is the point of
    measuring it -- and the alternatives were to lie with `replay=True` or to
    flip the arm to `current` before it had been measured, which is an in-band
    waiver (verification-doctrine 12). It still refuses `superseded` and an
    unregistered selector.
    """
    import hashlib
    path = pathlib.Path(config_path)
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ChunkPolicyRegistryError(
            f"cannot read selector {path}: {exc}") from exc
    doc = doc if doc is not None else load()
    ent = selector_entry(sha, doc)
    if replay:
        return ent
    allowed = EXPERIMENT_SELECTABLE if experiment else SELECTABLE
    status = ent.get("status")
    if status not in allowed:
        successor = ent.get("successor_sha256")
        adr = ent.get("superseded_by_adr")
        raise ChunkPolicyRegistryError(
            f"selector {ent.get('selector_id')} ({path.name}, sha "
            f"{sha[:12]}...) has registry status {status!r} and may not be "
            "selected for a new materialization"
            + (" (experiment=True admits `proposed`, not this)"
               if experiment else "")
            + (f"; successor {successor[:12]}..." if successor else "")
            + (f"; {adr}" if adr else "")
            + ". Pass replay=True only to reproduce rows that already exist, "
              "experiment=True only for a measurement arm that writes to no "
              "collection.")
    return ent


def is_withheld(ent: dict) -> bool:
    """True for a row flagged `public: false` WITH a reason (task 1934)."""
    return ent.get("public", True) is False and bool(ent.get("withheld_reason"))


def withheld(doc: dict | None = None,
             repo: pathlib.Path = REPO) -> list[str]:
    """Selector files the registry declares withheld AND that are absent here."""
    doc = doc if doc is not None else load()
    arts = (doc.get("selectors") or {}).get("artifacts") or {}
    return sorted(e["file"] for e in arts.values()
                  if is_withheld(e) and not (repo / e.get("file", "")).exists())


def verify_against_files(doc: dict | None = None,
                         repo: pathlib.Path = REPO) -> list[str]:
    """Every disagreement between the registry and the artifacts on disk.

    Empty means the two agree. This is the check that keeps the sidecar from
    becoming prose: a registry that names a file whose bytes have moved, or an
    epoch that contradicts the artifact's own, is a defect.
    """
    doc = doc if doc is not None else load()
    problems: list[str] = []
    import hashlib

    on_disk: dict[str, pathlib.Path] = {}
    for path in sorted((repo / "data").glob("chunk-policy-*.json")):
        if path.name == pathlib.Path(DEFAULT_PATH).name:
            continue
        on_disk[hashlib.sha256(path.read_bytes()).hexdigest()] = path

    for sha, ent in doc["policies"].items():
        path = repo / ent.get("file", "")
        if not path.exists():
            problems.append(f"{sha[:12]}: file {ent.get('file')!r} does not exist")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            problems.append(
                f"{ent.get('file')}: sha256 on disk {got} != registry key {sha}")
            continue
        body = json.loads(path.read_text())
        if ent.get("epoch") != body.get("epoch"):
            problems.append(
                f"{ent.get('file')}: registry epoch {ent.get('epoch')} != "
                f"artifact epoch {body.get('epoch')}")
        if ent.get("policy_id") != body.get("policy_id"):
            problems.append(
                f"{ent.get('file')}: registry policy_id {ent.get('policy_id')!r}"
                f" != artifact policy_id {body.get('policy_id')!r}")
        if ent.get("policy_family") not in doc["policy_families"]:
            problems.append(
                f"{ent.get('file')}: policy_family "
                f"{ent.get('policy_family')!r} is undeclared")

    # Absence is measured, never inferred (corpus-doctrine 8): the roster is a
    # glob of data/, not a list somebody typed.
    for sha, path in on_disk.items():
        if sha not in doc["policies"]:
            problems.append(
                f"{path.relative_to(repo)}: on disk, sha {sha[:12]}..., and NOT "
                "in the registry")

    # The same two directions for the SELECTOR artifacts. Without this arm the
    # selectors block is prose: a widened rule could ship under a stale sha and
    # nothing would notice.
    selectors = (doc.get("selectors") or {}).get("artifacts") or {}
    sel_on_disk: dict[str, pathlib.Path] = {}
    for path in sorted((repo / "data").glob("chunk-reference-selector*.json")):
        sel_on_disk[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    for sha, ent in selectors.items():
        path = repo / ent.get("file", "")
        if ent.get("public", True) is False and not ent.get("withheld_reason"):
            problems.append(
                f"selector {sha[:12]}: public=false without a withheld_reason")
        if not path.exists() and is_withheld(ent):
            continue  # withheld from the public cut by design; see withheld()
        if not path.exists():
            problems.append(
                f"selector {sha[:12]}: file {ent.get('file')!r} does not exist")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            problems.append(
                f"{ent.get('file')}: sha256 on disk {got} != registry "
                f"selector key {sha}")
            continue
        body = json.loads(path.read_text())
        declared = body.get("selection_mode", "narrow-partitioned-and-merged")
        if ent.get("selection_mode") != declared:
            problems.append(
                f"{ent.get('file')}: registry selection_mode "
                f"{ent.get('selection_mode')!r} != artifact "
                f"{declared!r}")
        if ent.get("policy_family") not in doc["policy_families"]:
            problems.append(
                f"{ent.get('file')}: policy_family "
                f"{ent.get('policy_family')!r} is undeclared")
    for sha, path in sel_on_disk.items():
        if sha not in selectors:
            problems.append(
                f"{path.relative_to(repo)}: on disk, sha {sha[:12]}..., and "
                "NOT in the registry's selectors block")
    return problems


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="check the registry against every data/chunk-policy-*"
                         " file; exit 1 on any disagreement")
    a = ap.parse_args()
    doc = load(a.registry)
    if a.verify:
        problems = verify_against_files(doc)
        for p in problems:
            print(f"FAIL  {p}")
        if problems:
            print(f"\n{len(problems)} disagreement(s) between "
                  f"{a.registry or DEFAULT_PATH} and data/chunk-policy-*.json")
            return 1
        print(f"OK    {len(doc['policies'])} policies, "
              f"{len(doc['policy_families'])} families, registry agrees "
              "with disk")
        held = withheld(doc)
        if held:
            print(f"      {len(held)} withheld (public: false, absent here): "
                  + ", ".join(held))
        return 0
    for sha, ent in sorted(doc["policies"].items(),
                           key=lambda kv: kv[1]["policy_family"]):
        print(f"{ent['policy_family']:<18} {ent.get('status','?'):<14} "
              f"epoch {ent.get('epoch')}  {ent.get('policy_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
