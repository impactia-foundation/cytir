#!/usr/bin/env python3
"""Can this family be staged at all -- and does its declaration contradict a ruling?

The payload-shape contract asks whether the rows a family produced LOOK right.
This asks the question one step earlier: **has the family declared what it is**,
before a single byte is planned. Task 529.

The class this catches is the one task 526/527 found by accident, in Tim's
"Marc Maret" test: the 1024 rebuild changed WHAT TEXT is embedded for at least
two families -- fedlex lost the law name/abbreviation/SR from its dense surface,
entity cards lost their sparse surface entirely -- and every count check, every
guard and the full-stack gate all passed, because BM25 was carrying the query.
Nothing in the pipeline had ever been told what those families were SUPPOSED to
embed, so nothing could notice that they stopped.

The same hole exists, measured, for eight other properties of a family:

  * the chunk policy id  -- 2 of 86 sealed families declare one (task 529 §2);
  * the consent terminal schema version -- task 521: 28,477 of 129,120 lost
    points sealed `declined` under a v5 reading of `robots_unreadable`, when
    18 of the 20 hosts were admissible on re-probe;
  * the licence instrument -- never inferred from hosting (2026-08-20);
  * the title shape -- task 516: fedlex shipped 22,913 of 22,928 rows with a
    title that carried no handle the query uses;
  * the point-id minting rule -- ADR-101 Ruling 2, the sealed id wins;
  * the ADR-039 payload minimum;
  * the eval classes the family must not regress;
  * the ADR-101 guards that apply to it.

WHAT THIS REFUSES
-----------------
`assembly_binder plan` and `assembly_binder stage` call `assert_declared()`
before they read a row. A set_id with no declaration is a hard refusal: not a
warning, not a default. That is the whole point -- the next family we onboard
must not be able to reach the index by being unremarkable.

A declaration that CONTRADICTS a ratified ADR is also a refusal, and the
message names the ADR. The constraint table lives in `RATIFIED` below, one
entry per rule, each with its cite; adding a constraint is an ADR PR, never an
edit here alone (memory: legislative-prs).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not derive a declaration from the evidence. A check that selects on
what it reports is a tautology (`assembly_guards.py`: "The plan is not derived
from the ledger"); the family declares, the artifacts on disk attest, and this
module refuses when they disagree.

It does not grant waivers. A family that shipped before this contract existed
carries `status: "backfilled"` and may hold `unknown` values -- that
grandfathering is DATA in `data/source-family-contracts.v1.json`, ratified by
Tim's merge of the PR, and `--strict` refuses it too. There is no code path
that lets a caller skip a rule (memory: gate-waivers-escalate-to-tim).

Usage
-----
    python3 -m cytir.sources.contract --check
    python3 -m cytir.sources.contract --check --strict
    python3 -m cytir.sources.contract --assert-sets a,b,c
    python3 -m cytir.sources.contract --coverage      # roster vs doc
"""
from __future__ import annotations

from cytir.core.paths import data_path

import argparse
import hashlib
import json
import pathlib
import re
import sys

SCHEMA = "aw.source-family-contract.v2"
#: v1 -> v2 (task 1016, Q13): v2 adds sibling `<key>_modules` lists (dotted `aw.*` ids) next to
#: path-valued `minted_by` / `declared_by` / `instrument` / `*_source` text. The text itself is
#: unchanged and stays what readers compare and stamp; v1 files load as before for one release.
SCHEMA_V1 = "aw.source-family-contract.v1"
SCHEMAS = (SCHEMA, SCHEMA_V1)
DEFAULT_PATH = "data/source-family-contracts.v1.json"
SCHEMA_PATH = "data/source-family-contract.schema.v1.json"

UNKNOWN = "unknown"

#: `chunk_policy.policy_id: native` (Tim 2026-10-05, ruling 26 item 20c): the chunk IS the
#: source's own article. It is an explicit value, not a file lookup: there is no
#: `data/native.json`, because there is no generic policy to pin. It carries no sha256 and no
#: epoch, and the registry cross-check does not apply (there is no policy to place).
from cytir.sources.family import CHUNK_NATIVE as NATIVE_CHUNK_POLICY  # noqa: E402 -- one home for the word
#: `native-chunked` (Tim 2026-10-07, ruling 27 T6 a; POLICY.md §4.1): the same, and an over-window
#: article may be split at paragraph marks. Same no-file/no-sha256/no-epoch rule as `native`.
from cytir.sources.family import CHUNK_NATIVE_CHUNKED as NATIVE_CHUNKED_CHUNK_POLICY  # noqa: E402
from cytir.sources.family import NATIVE_POLICIES  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

#: `producer_ids[*].status`: `frozen-label` = the old path persists in sealed receipts or served
#: rows (a FROZEN_RECEIPT_LABELS tuple exempts it from the cut); `retired-path` = the path was
#: retired and no artefact carries it (task 1396 ruling).
PRODUCER_ID_STATUSES = frozenset({"frozen-label", "retired-path"})
PRODUCER_ID_KEYS = ("status", "meaning", "code_home", "declared", "date")


class FamilyContractError(Exception):
    """A family is undeclared, or its declaration contradicts a ruling."""


class RosterWithheld(FamilyContractError):
    """The roster file is absent from a public cut (task 2229): not a failure of the cut."""


#: The marker file a public cut export carries (`cytir.ops.public_cut.MANIFEST_NAME`; named here
#: because `cytir.sources` does not import `cytir.ops`). Only a tree that carries it may lack the roster.
CUT_MARKER = "PUBLIC-CUT-MANIFEST.json"


# --------------------------------------------------------------------------
# The ratified constraint table. Every entry is a rule a declaration may not
# contradict, with the ruling that made it one. Editing this table without the
# corresponding ADR is the defect it exists to prevent.
# --------------------------------------------------------------------------
RATIFIED: dict[str, str] = {
    "consent.resolution": (
        "CLAUDE.md durable constraint + ADR-024: consent is re-resolved live, "
        "per source, and is NEVER inherited from Common Crawl."),
    "consent.terminal_schema_version": (
        "ADR-092 / ADR-103 (the retired `direct_terminal_contract.py` script): the terminal "
        "taxonomy is versioned and forward-only. A NEW family acquires at the "
        "current version; an older version is legal only to re-derive already-"
        "sealed evidence. Task 521 measured what the v5 reading cost: 28,477 of "
        "129,120 lost points sealed `declined` on robots 403/ConnectError."),
    "licence.inferred_from_hosting": (
        "2026-08-20 ruling (memory: licence-basis-from-the-instrument): never "
        "infer a legal basis from where a document is hosted; consent and "
        "licence are independent gates."),
    "licence.basis": (
        "Same ruling. A licence rests on an instrument, a statute, or an "
        "explicit grant -- never on the host."),
    "payload_minimum.fields": (
        "ADR-039: the compact redistributable dump carries the minimum record "
        "INCLUDING per-record `license`."),
    "payload_minimum.master_only": (
        "ADR-039: owner/attribution and license_version are master-only, "
        "captured at ingest and STRIPPED from the dump. A family that ships "
        "them in the dump contradicts the ADR."),
    "identity.rebinds_at_assembly": (
        "ADR-101 Ruling 2 (the assembly binder): the sealed point id "
        "wins and the binder does not re-mint. Re-minting produces a different "
        "identity space and invalidates the entire guard chain."),
    "representation.sparse_surface": (
        "Task 526/527: an absent sparse surface is a representation change, "
        "not an implementation detail -- entity cards lost their 82-char "
        "name+UID+people-token surface in the rebuild and nothing noticed "
        "because full-stack scoring was masked by BM25 on other families."),
    "representation.ruling": (
        "Tim 2026-09-06 (plan.md, the regression-free flip contract): a family "
        "whose surfaces differ from the incumbent with NO named ruling is "
        "UNRULED and may not ride a flip."),
    "guards": (
        "ADR-101 Ruling 2 (the retired `assembly_guards.py` script): a replacement owes "
        "G1, G3, G5 and G7. The burden is not reduced for any set."),
    "gate_classes": (
        "The ratified 629-query instrument's class taxonomy "
        "(data/eval_tasks.yaml)."),
    "title_shape.why": (
        "Task 516 §3: the title contract is opt-in because five families were "
        "measured to have no production title pattern. Opting out is a "
        "judgement and must carry its reason."),
    "chunk_policy.policy_id": (
        "`data/chunk-policy-*.json` is the one home for a chunk policy "
        "(engineering-doctrine 1). A policy id that resolves to no file is a "
        "policy nobody can reproduce."),
    "chunk_policy.policy_family": (
        "ADR-090 §3/§4 split `chunk` into `chunk.generic` and `chunk.law` and "
        "scoped 'standardize on 1152' to the generic path alone; "
        "specs/epochs.md §5 says a third chunker family is SPLIT FORMALLY, not "
        "absorbed. `data/chunk-policy-registry.v1.json` is the machine-readable "
        "home of that taxonomy, and a family that names a policy must agree "
        "with it. Task 535 measured why: a 3.73x granularity change swept 82 "
        "hosts and nothing in the repo could see it, because no artifact stated "
        "which family a policy served."),
    "chunk_policy.selectable": (
        "ADR-090 §6 + `cytir.chunk.registry`: a policy whose "
        "registry status is not `current` may not be selected by a NEW "
        "materialization. `proposed` is refused for the same reason "
        "`superseded` is -- a lane that could pick up an unratified policy is "
        "an in-band waiver (verification-doctrine 12)."),
    "unknowns": (
        "Verification-doctrine 6, `an error is not an empty`: a value the "
        "evidence does not carry is named `unknown` and LISTED. A declaration "
        "may not hide an unknown by omitting it from the list."),
}

#: ADR-039's compact-dump minimum. `cytir.core.license` and CLAUDE.md's
#: durable constraints state the same list; this is the machine-readable copy
#: the contract checks against.
ADR039_MINIMUM = frozenset({
    "url", "anchor", "title", "snippet", "embedding", "sparse_terms",
    "lang", "trust_tier", "last_seen", "digest", "license",
})

#: ADR-039 again: captured at ingest, stripped from the redistributable dump.
ADR039_MASTER_ONLY = frozenset({"owner", "attribution", "license_version", "family_id"})

#: The class taxonomy of the pinned 629-query instrument.
GATE_CLASSES = ("civic", "coding", "entity", "legal", "reference",
                "research", "standards")

#: The terminal schema versions live in ONE place --
#: the retired `direct_terminal_contract.py` script -- and this module READS them from
#: there rather than keeping a copy (engineering-doctrine 1 / 6: a second
#: implementation converts one bug into two).
#:
#: It reads them by parsing the source rather than importing it, deliberately.
#: `direct_terminal_contract` imports `consent_resolver`, which refuses at
#: import time on a checkout with no configured crawler identity -- correct
#: behaviour for a fetcher, fatal for a contract check that must run on a cold
#: clone with no operator.json. An `except ImportError: use_a_copy()` here
#: would be a silent fallback to a value that can drift, which is the failure
#: mode this whole module exists to prevent.
_TERMINAL_MODULE = "terminal.py"

GUARDS = tuple(f"G{i}" for i in range(1, 9))
#: assembly_guards.py's own statement of the replacement burden.
REPLACEMENT_GUARDS = frozenset({"G1", "G3", "G5", "G7"})



def _terminal_versions(path: pathlib.Path | None = None
                       ) -> tuple[tuple[str, ...], str]:
    """`(SCHEMA_VERSIONS, CURRENT_SCHEMA_VERSION)`, read from the one home.

    Parses `direct_terminal_contract.py`'s module-level assignments with `ast`
    -- no import, so no crawler-identity side effect -- and refuses if either
    name is missing or is not a plain string / tuple-of-string-names. There is
    no fallback value: a contract that cannot read the version taxonomy must
    stop, not guess.
    """
    import ast

    path = path or (pathlib.Path(__file__).resolve().parents[1] / "consent"
                    / _TERMINAL_MODULE)
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise FamilyContractError(
            f"cannot read the terminal schema versions from {path}: {exc}. "
            "They are not duplicated here on purpose; fix the read.") from exc

    consts: dict[str, str] = {}
    versions: tuple[str, ...] | None = None
    current: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name, value = target.id, node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            consts[name] = value.value
            if name == "CURRENT_SCHEMA_VERSION":
                current = value.value
        elif isinstance(value, ast.Name) and name == "CURRENT_SCHEMA_VERSION":
            current = consts.get(value.id)
        elif isinstance(value, ast.Tuple) and name == "SCHEMA_VERSIONS":
            got = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    got.append(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in consts:
                    got.append(consts[elt.id])
                else:
                    got = []
                    break
            versions = tuple(got) or None

    if not versions or not current:
        raise FamilyContractError(
            f"{path} no longer defines SCHEMA_VERSIONS and "
            "CURRENT_SCHEMA_VERSION as module-level string constants. "
            "source_family_contract reads them from there rather than "
            "keeping a copy; adjust the reader, do not copy the values.")
    if current not in versions:
        raise FamilyContractError(
            f"{path}: CURRENT_SCHEMA_VERSION {current!r} is not in "
            f"SCHEMA_VERSIONS {versions!r}")
    return versions, current


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def duplicate_keys(text: str) -> list[str]:
    """Every key that appears twice in one JSON object of `text`. A parser keeps the last copy
    and silently shadows the first (task 2227)."""
    found: list[str] = []

    def hook(pairs):
        seen: set[str] = set()
        for k, _ in pairs:
            if k in seen:
                found.append(k)
            seen.add(k)
        return dict(pairs)

    json.loads(text, object_pairs_hook=hook)
    return found


def load(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else data_path(pathlib.Path(DEFAULT_PATH).name)
    text = p.read_text()
    dups = duplicate_keys(text)
    if dups:
        raise FamilyContractError(
            f"{p}: duplicate key in a JSON object (the first copy is shadowed): "
            + ", ".join(sorted(set(dups))))
    doc = json.loads(text)
    if doc.get("schema") not in SCHEMAS:
        raise FamilyContractError(f"{p}: not {SCHEMA} (or the v1 shim {SCHEMA_V1})")
    doc["_path"] = str(p)
    return doc


# -- real names by alias (task 2855, RULING 78) ---------------------------
# A row may carry `canonical_id` (the real name) and `legacy_ids` (every id it was ever keyed
# by, always including its own `family_id`). Sealed data stays keyed by `family_id`; only the
# aw_e11 rebuild stamp and reports read the canonical name. Pattern = what the real-named
# families follow (census 2789): lowercase snake case, no task/scratch/wave/stage marker.
TASK_NAMED_FAMILY = re.compile(r"^(scratch_|task\d+_)|_\d{3}[a-z]?$|_w\d+$|wave\d")
CANONICAL_ID_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def family_canonical_id(family_id: str, doc: dict | None = None) -> str:
    """The real name of `family_id`: its `canonical_id`, else `family_id` itself. A legacy id
    that is not a row's own `family_id` also resolves. Unknown ids come back unchanged."""
    m = _alias_maps(doc)
    return m["to_canonical"].get(family_id, family_id)


def legacy_family_id(canonical_id: str, doc: dict | None = None) -> str:
    """Inverse of `family_canonical_id`: the `family_id` a sealed file is keyed by."""
    m = _alias_maps(doc)
    return m["to_family"].get(canonical_id, canonical_id)


_ALIAS_CACHE: dict[str, dict] = {}


def _alias_maps(doc: dict | None) -> dict:
    if doc is None:
        key = str(data_path(pathlib.Path(DEFAULT_PATH).name))
        if key not in _ALIAS_CACHE:
            _ALIAS_CACHE[key] = _build_alias_maps(load())
        return _ALIAS_CACHE[key]
    return _build_alias_maps(doc)


def _build_alias_maps(doc: dict) -> dict:
    to_c: dict[str, str] = {}
    to_f: dict[str, str] = {}
    for fam in doc.get("families") or []:
        fid = fam.get("family_id")
        cid = fam.get("canonical_id")
        if not cid:
            continue
        to_f[cid] = fid
        to_c[fid] = cid
        for lid in fam.get("legacy_ids") or []:
            to_c[lid] = cid
    return {"to_canonical": to_c, "to_family": to_f}


def alias_problems(doc: dict) -> list[str]:
    """Cross-row checks of `canonical_id` / `legacy_ids`."""
    p: list[str] = []
    fams = doc.get("families") or []
    own = {f.get("family_id") for f in fams}
    legacy_owner: dict[str, str] = {}
    for f in fams:
        for lid in f.get("legacy_ids") or []:
            legacy_owner.setdefault(lid, f.get("family_id"))
    seen: dict[str, str] = {}
    for f in fams:
        fid = f.get("family_id")
        cid = f.get("canonical_id")
        legacy = f.get("legacy_ids")
        if cid is None:
            if legacy is not None:
                p.append(f"{fid}: legacy_ids without canonical_id")
            continue
        if not isinstance(cid, str) or not CANONICAL_ID_RE.match(cid):
            p.append(f"{fid}: canonical_id {cid!r} is not lowercase snake case")
            continue
        if TASK_NAMED_FAMILY.search(cid):
            p.append(f"{fid}: canonical_id {cid!r} is task/scratch-named")
        if not isinstance(legacy, list) or fid not in legacy:
            p.append(f"{fid}: canonical_id set but legacy_ids does not contain its own family_id")
        if cid in seen:
            p.append(f"{fid}: canonical_id {cid!r} already used by {seen[cid]}")
        seen[cid] = fid
        if cid in own and cid != fid:
            p.append(f"{fid}: canonical_id {cid!r} equals another row's family_id")
        if cid in legacy_owner and legacy_owner[cid] != fid:
            p.append(f"{fid}: canonical_id {cid!r} is a legacy id of {legacy_owner[cid]}")
        for lid in legacy or []:
            if lid != fid and lid in own:
                p.append(f"{fid}: legacy id {lid!r} is another row's family_id")
            if legacy_owner.get(lid) != fid:
                p.append(f"{fid}: legacy id {lid!r} also claimed by {legacy_owner.get(lid)}")
    return p


def by_id(doc: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fam in doc.get("families") or []:
        fid = fam.get("family_id")
        if not fid:
            raise FamilyContractError("a family declaration has no family_id")
        if fid in out:
            raise FamilyContractError(
                f"family {fid!r} is declared twice; a family has one "
                "declaration or the validator is checking a coin flip")
        out[fid] = fam
    return out


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _data_file(repo: pathlib.Path, name: str) -> pathlib.Path:
    """`<repo>/data/<name>`; for the default repo (this package's own tree) `data_path()`, so an
    installed package finds the shipped copy in `cytir/_data` (task 735)."""
    return data_path(name) if repo is REPO else repo / "data" / name


def roster(doc: dict, repo: pathlib.Path = REPO) -> set[str]:
    """The family ids the roster source names, with its digest re-checked.

    Derived from the manifest, never a hand-list (memory:
    census-roster-from-manifest). A drifted digest is a refusal, not a warning:
    a roster that moved under the declaration is exactly how a family goes
    unnoticed.
    """
    src = doc.get("roster_source") or {}
    uri = src.get("uri")
    if not uri:
        raise FamilyContractError("roster_source.uri is required")
    p = repo / uri
    if not p.exists():
        if (repo / CUT_MARKER).exists():
            raise RosterWithheld(uri)
        raise FamilyContractError(f"roster source {uri} does not exist")
    got = sha256_file(p)
    want = src.get("sha256")
    if want and got != want:
        raise FamilyContractError(
            f"roster source {uri} sha256 {got} != declared {want}. The roster "
            "moved under the declaration; re-derive the declarations rather "
            "than re-stamping the digest.")
    manifests = json.loads(p.read_text())
    sets = manifests.get("sets") or []
    ids = {s["set_id"] for s in sets if isinstance(s, dict) and s.get("set_id")}
    declared_n = src.get("families")
    if declared_n is not None and int(declared_n) != len(ids):
        raise FamilyContractError(
            f"roster_source.families={declared_n} but {uri} names {len(ids)}")
    return ids


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _walk_unknowns(node, path: str, out: list[str]) -> None:
    """Collect every dotted path in `node` whose value is the sentinel.

    `evidence` and `unknowns` are skipped: they are bookkeeping about the
    declaration, not part of it, and a citation that happens to read "unknown"
    must not become a declared unknown.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("unknowns", "evidence"):
                continue
            _walk_unknowns(v, f"{path}.{k}", out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_unknowns(v, f"{path}[{i}]", out)
    elif node == UNKNOWN:
        out.append(path)


def actual_unknowns(fam: dict) -> list[str]:
    out: list[str] = []
    for key, val in fam.items():
        if key in ("unknowns", "evidence", "family_id", "status", "notes"):
            continue
        _walk_unknowns(val, key, out)
    return sorted(out)


def _check_chunk_policy_registry(fid: str, cp: dict, policy_file: pathlib.Path,
                                 status: str,
                                 repo: pathlib.Path) -> list[str]:
    """The family's chunk policy, checked against the ratified registry.

    Two questions the declaration alone cannot answer, both of which cost this
    project a 3.73x granularity change nobody could see (task 535):

      * WHICH FAMILY does this policy serve? `chunk.generic` / `chunk.law` /
        `chunk.news` / `chunk.reference` is a ruling (ADR-090 §3/§4), and its
        machine-readable home is `data/chunk-policy-registry.v1.json`.
      * MAY A NEW FAMILY STILL SELECT IT? A `superseded` or `proposed` policy
        stays loadable forever so historical rows replay, and a NEW
        materialization may not pick one up (ADR-090 §6).

    Imported lazily and reported as a problem rather than raised: the registry
    is a data file this validator checks against, and a checkout without it
    must fail this family loudly rather than crash the whole validation into an
    unread traceback.
    """
    import hashlib
    problems: list[str] = []
    try:
        from cytir.chunk import registry as CREG
        doc = CREG.load(_data_file(repo, pathlib.Path(CREG.DEFAULT_PATH).name))
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        return [f"{fid}: cannot read the chunk-policy registry: {exc}  "
                f"[{RATIFIED['chunk_policy.policy_family']}]"]

    sha = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    try:
        ent = CREG.entry(sha, doc)
    except CREG.ChunkPolicyRegistryError as exc:
        return [f"{fid}: {exc}  [{RATIFIED['chunk_policy.policy_family']}]"]

    declared_family = cp.get("policy_family")
    if declared_family not in (None, UNKNOWN):
        if declared_family != ent.get("policy_family"):
            problems.append(
                f"{fid}: chunk_policy.policy_family={declared_family!r} but the "
                f"registry puts {policy_file.name} in "
                f"{ent.get('policy_family')!r}  "
                f"[{RATIFIED['chunk_policy.policy_family']}]")

    # ADR-113: `admitted_for_families` only excuses a `proposed` policy --
    # never a `superseded`/`special-purpose` one, or a stale scope entry
    # nobody stripped when the arm was superseded would keep silently
    # passing (chunk_policy_registry.assert_selectable has the same guard).
    admitted_for = ent.get("admitted_for_families") or []
    scoped_ok = ent.get("status") == "proposed" and fid in admitted_for
    if (status == "declared" and ent.get("status") not in CREG.SELECTABLE
            and not scoped_ok):
        problems.append(
            f"{fid}: status 'declared' selects chunk policy "
            f"{ent.get('policy_id')} whose registry status is "
            f"{ent.get('status')!r}; only {sorted(CREG.SELECTABLE)} may be "
            f"selected by a new family  "
            f"[{RATIFIED['chunk_policy.selectable']}]")
    return problems


def _code_home_ok(home: str, repo: pathlib.Path) -> bool:
    """A producer_ids code_home is valid when it names an existing file under aw/.

    A code_home under tasks/<NNN>-scripts/ is a FROZEN LABEL (task 1787, memo 3 E):
    the file must exist where the repo has a tasks/ directory (the private tree),
    and is accepted without a file check where tasks/ is absent (a stranger's copy
    of the public cut, which ships no tasks/). The cytir/ check never relaxes."""
    if home.startswith("cytir/"):
        return (repo / home).is_file()
    if re.match(r"^frozen:task-\d+[a-z]?:", home):    # the public cut's relabel of the case below
        return True
    if re.match(r"^tasks/\d+-scripts/", home):
        return (repo / home).is_file() or not (repo / "tasks").is_dir()
    return False


def validate_family(fam: dict, *, strict: bool = False,
                    repo: pathlib.Path = REPO) -> list[str]:
    """Every problem with one declaration. Empty means it may stage."""
    p: list[str] = []
    fid = fam.get("family_id", "?")
    versions, current = _terminal_versions()

    def bad(field: str, msg: str) -> None:
        cite = RATIFIED.get(field)
        p.append(f"{fid}: {msg}" + (f"  [{cite}]" if cite else ""))

    # -- shape ------------------------------------------------------------
    required = ("family_id", "status", "population", "mode", "representation",
                "chunk_policy", "consent", "licence", "title_shape",
                "identity", "payload_minimum", "gate_classes", "guards",
                "contracts", "unknowns")
    for key in required:
        if key not in fam:
            p.append(f"{fid}: missing required block {key!r}")
    if p:
        return p  # nothing below can be trusted against a half-shaped record

    status = fam["status"]
    if status not in ("declared", "backfilled", "retired"):
        p.append(f"{fid}: status {status!r} is not declared/backfilled/retired")

    # -- the unknowns list is not optional bookkeeping --------------------
    got_unknowns = sorted(fam.get("unknowns") or [])
    real = actual_unknowns(fam)
    if got_unknowns != real:
        bad("unknowns",
            f"`unknowns` says {got_unknowns} but the declaration actually "
            f"holds the sentinel at {real}")
    if status == "declared" and real:
        p.append(
            f"{fid}: status 'declared' with {len(real)} unknown value(s) "
            f"{real}. A complete declaration has none; a family reconstructed "
            "from sealed evidence is 'backfilled'.")
    if strict and real:
        p.append(f"{fid}: --strict and {len(real)} unknown value(s): {real}")

    # -- consent ----------------------------------------------------------
    consent = fam["consent"]
    res = consent.get("resolution")
    if res not in ("live-per-origin", "explicit-grant",
                   "not-applicable-synthesized", UNKNOWN):
        bad("consent.resolution",
            f"consent.resolution={res!r} is not a legal basis")
    tsv = consent.get("terminal_schema_version")
    if tsv != UNKNOWN and tsv not in versions:
        bad("consent.terminal_schema_version",
            f"consent.terminal_schema_version={tsv!r} is not one of {versions}")
    if status == "declared" and tsv != current:
        bad("consent.terminal_schema_version",
            f"a NEW family acquires at {current!r}, not {tsv!r}")

    # -- licence ----------------------------------------------------------
    lic = fam["licence"]
    if lic.get("inferred_from_hosting") is not False:
        bad("licence.inferred_from_hosting",
            f"licence.inferred_from_hosting="
            f"{lic.get('inferred_from_hosting')!r}; the only legal value is "
            "false")
    if lic.get("basis") not in ("instrument", "statute", "explicit-grant",
                               UNKNOWN):
        bad("licence.basis", f"licence.basis={lic.get('basis')!r}")

    # -- payload minimum (ADR-039) ---------------------------------------
    pm = fam["payload_minimum"]
    fields = set(pm.get("fields") or [])
    missing = ADR039_MINIMUM - fields
    if missing and fields:
        bad("payload_minimum.fields",
            f"payload minimum omits {sorted(missing)}")
    leaked = fields & ADR039_MASTER_ONLY
    if leaked:
        bad("payload_minimum.master_only",
            f"the dump minimum lists master-only field(s) {sorted(leaked)}")

    # -- identity (ADR-101 Ruling 2) --------------------------------------
    ident = fam["identity"]
    if ident.get("rebinds_at_assembly") not in (False, None):
        bad("identity.rebinds_at_assembly",
            "identity.rebinds_at_assembly is not false")
    if "point_id_rule" not in ident:
        p.append(f"{fid}: identity.point_id_rule is required")

    # -- representation (task 526/527) ------------------------------------
    rep = fam["representation"]
    for surf in ("dense_surface", "sparse_surface"):
        if surf not in rep:
            bad(f"representation.{surf}",
                f"representation.{surf} is not declared at all")
    ruling = rep.get("ruling")
    if ruling not in ("IDENTICAL", "RULED", "UNRULED", UNKNOWN):
        bad("representation.ruling", f"representation.ruling={ruling!r}")
    if ruling == "RULED" and not rep.get("ruling_cite"):
        bad("representation.ruling",
            "representation.ruling is RULED with no ruling_cite; a ruling "
            "nobody can find is not a ruling")
    if rep.get("sparse_surface") == "none" and not rep.get("ruling_cite"):
        bad("representation.sparse_surface",
            "sparse_surface 'none' with no ruling_cite -- this is the "
            "entity-card defect stated as a declaration")

    # -- chunk policy ------------------------------------------------------
    cp = fam["chunk_policy"]
    pid = cp.get("policy_id")
    if pid in NATIVE_POLICIES:
        for k in ("sha256", "epoch"):
            if cp.get(k) not in (None, UNKNOWN):
                bad("chunk_policy.policy_id",
                    f"chunk_policy.policy_id is {pid!r} but {k}={cp.get(k)!r}: "
                    f"a native chunk has no policy file to pin; {k} must be null")
    elif pid and pid != UNKNOWN:
        pf = _data_file(repo, f"{pid}.json")
        if not pf.exists():
            bad("chunk_policy.policy_id",
                f"chunk_policy.policy_id={pid!r} resolves to no "
                f"data/{pid}.json")
        else:
            want = cp.get("sha256")
            if want and want != UNKNOWN:
                got = sha256_file(pf)
                if got != want:
                    bad("chunk_policy.policy_id",
                        f"data/{pid}.json sha256 {got} != declared {want}")
            body = json.loads(pf.read_text())
            ep = cp.get("epoch")
            if ep not in (None, UNKNOWN) and body.get("epoch") != ep:
                bad("chunk_policy.policy_id",
                    f"declared epoch {ep} != {pid}'s own epoch "
                    f"{body.get('epoch')}")
            p.extend(_check_chunk_policy_registry(fid, cp, pf, status, repo))

    # -- title shape -------------------------------------------------------
    ts = fam["title_shape"]
    if ts.get("mode") == "opted_out" and not ts.get("why"):
        bad("title_shape.why", "title_shape opted_out with no `why`")
    if ts.get("mode") == "declared" and not ts.get("pattern"):
        p.append(f"{fid}: title_shape mode 'declared' with no pattern")

    # -- gate classes ------------------------------------------------------
    for c in fam.get("gate_classes") or []:
        if c not in GATE_CLASSES and c != UNKNOWN:
            bad("gate_classes", f"gate class {c!r} is not one of {GATE_CLASSES}")

    # -- guards ------------------------------------------------------------
    guards = set(fam.get("guards") or [])
    unknown_guards = guards - set(GUARDS)
    if unknown_guards:
        bad("guards", f"unknown guard(s) {sorted(unknown_guards)}")
    if fam["mode"] == "replacement":
        owed = REPLACEMENT_GUARDS - guards
        if owed:
            bad("guards",
                f"mode 'replacement' but guards omit {sorted(owed)}")

    # -- producer ids: the vocabulary is closed, the code home exists ----------
    for label, entry in sorted((fam.get("producer_ids") or {}).items()):
        miss = [k for k in PRODUCER_ID_KEYS if not (entry or {}).get(k)]
        if miss:
            p.append(f"{fid}: producer_ids[{label!r}] lacks {miss}")
            continue
        if entry["status"] not in PRODUCER_ID_STATUSES:
            p.append(f"{fid}: producer_ids[{label!r}] status {entry['status']!r} is not one of "
                     f"{sorted(PRODUCER_ID_STATUSES)}")
        tokens = entry["code_home"].split()     # a trailing note after the path is allowed
        home = tokens[0]
        if home == "none":                       # no code home: a note must say why (task 1557)
            if len(tokens) < 2:
                p.append(f"{fid}: producer_ids[{label!r}] code_home 'none' needs a note "
                         "(why there is no code home)")
            continue
        if not _code_home_ok(home, repo):
            p.append(f"{fid}: producer_ids[{label!r}] code_home {home!r} is not an existing "
                     "file under cytir/ or tasks/<NNN>-scripts/")

    return p


def validate(doc: dict, *, strict: bool = False,
             repo: pathlib.Path = REPO) -> dict:
    """Validate the whole document. Returns a report; never raises on a
    per-family problem (the caller decides), but DOES raise on a document that
    cannot be reasoned about at all."""
    fams = by_id(doc)
    problems: list[str] = []
    for fam in doc.get("families") or []:
        problems.extend(validate_family(fam, strict=strict, repo=repo))
    problems.extend(alias_problems(doc))

    coverage = None
    withheld = None
    try:
        names = roster(doc, repo)
    except RosterWithheld as exc:
        withheld = str(exc)
    except FamilyContractError as exc:
        problems.append(f"roster: {exc}")
    else:
        undeclared = sorted(names - set(fams))
        extra = sorted(set(fams) - names)
        coverage = {"roster": len(names), "declared": len(fams),
                    "undeclared": undeclared, "not_in_roster": extra}
        if undeclared:
            problems.append(
                f"{len(undeclared)} family/families in the roster have NO "
                f"declaration: {undeclared[:10]}"
                + (" ..." if len(undeclared) > 10 else ""))

    by_status: dict[str, int] = {}
    for fam in fams.values():
        by_status[fam.get("status", "?")] = by_status.get(
            fam.get("status", "?"), 0) + 1

    return {
        "verdict": "PASS" if not problems else "FAIL",
        "families": len(fams),
        "by_status": by_status,
        "unknown_families": sorted(
            f["family_id"] for f in fams.values() if f.get("unknowns")),
        "coverage": coverage,
        "roster_withheld": withheld,
        "problems": problems,
    }


# --------------------------------------------------------------------------
# the refusal the binder calls
# --------------------------------------------------------------------------
def assert_declared(set_ids, doc: dict | None = None, *, strict: bool = False,
                    repo: pathlib.Path = REPO, stage: str = "plan") -> None:
    """Refuse `assembly_binder <stage>` for any set with no valid declaration.

    Called at the TOP of `cmd_plan` and `cmd_stage`, before a row is read:
    a cheap check at the front of an expensive process beats an assertion at
    hour six (verification doctrine, project interpretation).
    """
    doc = doc if doc is not None else load()
    fams = by_id(doc)
    wanted = sorted(set(set_ids))
    missing = [s for s in wanted if s not in fams]
    if missing:
        raise FamilyContractError(
            f"assembly_binder {stage}: {len(missing)} set(s) have no "
            f"source-family declaration: {missing}. Add one to "
            f"{doc.get('_path', DEFAULT_PATH)} (schema: {SCHEMA_PATH}) and "
            "have Tim ratify the PR. A family the pipeline has never been "
            "told about cannot be checked for having stopped being itself "
            "-- that is the task 526/527 defect.")
    problems: list[str] = []
    for s in wanted:
        problems.extend(validate_family(fams[s], strict=strict, repo=repo))
    if problems:
        raise FamilyContractError(
            f"assembly_binder {stage}: {len(problems)} source-family contract "
            "problem(s):\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def format_report(report: dict, log=print) -> None:
    log(f"source-family contract: {report['verdict']} "
        f"({report['families']} families, {report['by_status']})")
    cov = report.get("coverage")
    if cov:
        log(f"  roster {cov['roster']} · declared {cov['declared']} · "
            f"undeclared {len(cov['undeclared'])} · "
            f"not-in-roster {len(cov['not_in_roster'])}")
    if report["unknown_families"]:
        log(f"  {len(report['unknown_families'])} family/families carry "
            f"`unknown` values (backfilled from sealed evidence)")
    if report.get("roster_withheld"):
        log(f"  roster: withheld-in-cut ({report['roster_withheld']})")
    for prob in report["problems"]:
        log(f"  FAIL {prob}")


def pre_amendment_sha256(text: str) -> tuple[str, str]:
    """(recorded, computed) for the LAST amendment row's `pre_amendment_file_sha256`.

    The amendments array is the top-level list whose objects carry `old_file_sha256`. The
    computed value is sha256 of the file text with that last object, and the comma and
    whitespace before it, removed (string-aware brace walk: the file is hand-edited text,
    not json.dumps output). Earlier rows are historic and not re-checked. Fails closed:
    raises FamilyContractError on malformed input.
    """
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise FamilyContractError(f"contracts file is not JSON: {exc}") from exc
    key = None
    if isinstance(doc, dict):
        key = next((k for k, v in doc.items() if isinstance(v, list) and v
                    and isinstance(v[0], dict) and "old_file_sha256" in v[0]), None)
    if key is None:
        raise FamilyContractError(
            "no amendments array (top-level list of objects with old_file_sha256)")
    last = doc[key][-1]
    recorded = last.get("pre_amendment_file_sha256") if isinstance(last, dict) else None
    if not isinstance(recorded, str) or not recorded:
        raise FamilyContractError("last amendment lacks pre_amendment_file_sha256")
    try:
        arr = text.index("[", text.index('"%s"' % key))
    except ValueError as exc:
        raise FamilyContractError("amendments array not locatable in text") from exc
    depth, starts, end, in_str, esc = 0, [], None, False, False
    for i in range(arr, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
            if depth == 2 and c == "{":
                starts.append(i)
        elif c in "]}":
            depth -= 1
            if depth == 1 and c == "}":
                end = i
            if depth == 0:
                break
    if not starts or end is None:
        raise FamilyContractError("last amendment object not locatable in text")
    head = text[:starts[-1]]
    stripped = head.rstrip()
    cut = len(stripped) - 1 if stripped.endswith(",") else starts[-1]
    computed = hashlib.sha256((text[:cut] + text[end + 1:]).encode()).hexdigest()
    return recorded, computed


# --------------------------------------------------------------------------
# amend: the one way to edit the sealed contracts file (task 2496)
# --------------------------------------------------------------------------
_KEY_SPLIT = re.compile(r"(?<!\\)\.")
_FAMILY_KEY_INDENT = "   "   # the file's own style: families at 2 spaces, their keys at 3, +1 per level


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _object_members(text: str, start: int) -> list[tuple[str, int, int, int]]:
    """(key, key_start, value_start, value_end) for each member of the JSON object whose `{` is
    at `start`. String-aware: the JSON decoder finds every span."""
    dec = json.JSONDecoder()
    i = _skip_ws(text, start + 1)
    out = []
    while text[i] != "}":
        key, j = dec.raw_decode(text, i)
        k = _skip_ws(text, j)
        if text[k] != ":":
            raise FamilyContractError("contracts text: expected ':' after a key")
        vs = _skip_ws(text, k + 1)
        _, ve = dec.raw_decode(text, vs)
        out.append((key, i, vs, ve))
        i = _skip_ws(text, ve)
        if text[i] == ",":
            i = _skip_ws(text, i + 1)
    return out


def _array_elements(text: str, start: int) -> list[tuple[int, int]]:
    dec = json.JSONDecoder()
    i = _skip_ws(text, start + 1)
    out = []
    while text[i] != "]":
        _, e = dec.raw_decode(text, i)
        out.append((i, e))
        i = _skip_ws(text, e)
        if text[i] == ",":
            i = _skip_ws(text, i + 1)
    return out


def _member(text: str, obj_start: int, name: str):
    return next((m for m in _object_members(text, obj_start) if m[0] == name), None)


def _family_span(text: str, family_id: str) -> tuple[int, int]:
    fams = _member(text, 0, "families")
    if fams is None:
        raise FamilyContractError("contracts text has no families array")
    for s, e in _array_elements(text, fams[2]):
        m = _member(text, s, "family_id")
        if m is not None and json.loads(text[m[2]:m[3]]) == family_id:
            return s, e
    raise FamilyContractError(f"unknown family {family_id!r}: no such family_id in the contracts file")


def _dump_value(value, *, ensure_ascii: bool) -> str:
    body = json.dumps(value, indent=1, ensure_ascii=ensure_ascii)
    return body.replace("\n", "\n" + _FAMILY_KEY_INDENT)


def set_in_text(text: str, family_id: str, dotted: str, value) -> str:
    """Insert or replace one dotted key of one family, textually. Only that top-level key's value
    is re-serialised (file style: indent 1 per level); every other byte is unchanged. A key that
    itself contains a dot is written `a\\.b`."""
    path = [seg.replace("\\.", ".") for seg in _KEY_SPLIT.split(dotted)]
    if not path or not all(path):
        raise FamilyContractError(f"bad key {dotted!r}")
    ascii_only = not any(ord(c) > 127 for c in text)
    fs, _ = _family_span(text, family_id)
    top = path[0]
    existing = _member(text, fs, top)
    if len(path) == 1:
        new_val = value
    else:
        new_val = json.loads(text[existing[2]:existing[3]]) if existing else {}
        node = new_val
        for seg in path[1:-1]:
            nxt = node.get(seg) if isinstance(node, dict) else None
            if nxt is None:
                nxt = node[seg] = {}
            if not isinstance(nxt, dict):
                raise FamilyContractError(f"{dotted!r}: {seg!r} holds a non-object value")
            node = nxt
        if not isinstance(node, dict):
            raise FamilyContractError(f"{dotted!r}: parent is not an object")
        node[path[-1]] = value
    dumped = _dump_value(new_val, ensure_ascii=ascii_only)
    if existing is not None:
        return text[:existing[2]] + dumped + text[existing[3]:]
    fid = _member(text, fs, "family_id")
    return text[:fid[3]] + ",\n" + _FAMILY_KEY_INDENT + json.dumps(top) + ": " + dumped + text[fid[3]:]


def append_amendment_row(text: str, row_fn) -> str:
    """Append one amendments row in the file's one-line style. `row_fn(edited_text)` returns the
    row dict, so the row can carry the sha256 of the text it is appended to."""
    arr = None
    for _k, _ks, vs, _ve in _object_members(text, 0):
        if text[vs] == "[":
            els = _array_elements(text, vs)
            if els and text[els[0][0]] == "{" and _member(text, els[0][0], "old_file_sha256"):
                arr = els
    if not arr:
        raise FamilyContractError("no amendments array (top-level list of objects with old_file_sha256)")
    row = row_fn(text)
    ascii_only = not any(ord(c) > 127 for c in text)
    last_end = arr[-1][1]
    return text[:last_end] + ",\n    " + json.dumps(row, ensure_ascii=ascii_only) + text[last_end:]


def check_text(text: str) -> list[str]:
    """The problems `--check` would report for this text (empty = it passes). In-process."""
    try:
        dups = duplicate_keys(text)
        if dups:
            return ["duplicate key in a JSON object: " + ", ".join(sorted(set(dups)))]
        doc = json.loads(text)
        if doc.get("schema") not in SCHEMAS:
            return [f"not {SCHEMA}"]
        recorded, computed = pre_amendment_sha256(text)
        if recorded != computed:
            return [f"pre-amendment sha MISMATCH recorded={recorded} computed={computed}"]
        return list(validate(doc)["problems"])
    except (FamilyContractError, ValueError) as exc:
        return [str(exc)]


def _vcs_dirty(path: pathlib.Path) -> str | None:
    """Porcelain status of `path`; None when version control cannot say (not a work tree)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(path.resolve().parent), "status", "--porcelain", "--", path.name],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def amend(path: pathlib.Path, family_id: str, sets: list[tuple[str, object]], reason: str,
          task: str, *, today: str | None = None, check=None) -> tuple[str, str]:
    """(old_text, new_text) of the amended file; raises FamilyContractError on any refusal.
    Writes nothing: the caller writes new_text."""
    import datetime
    raw = path.read_bytes()
    old = raw.decode("utf-8")
    if not sets:
        raise FamilyContractError("nothing to do: give at least one --set")
    if not reason.strip():
        raise FamilyContractError("--reason is required")
    text = old
    for dotted, value in sets:
        text = set_in_text(text, family_id, dotted, value)
    day = today or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    old_sha = hashlib.sha256(raw).hexdigest()

    def row(edited: str) -> dict:
        return {"date": day, "task": f"task {task}", "change": reason.strip(),
                "families": [family_id], "keys": [d for d, _ in sets],
                "old_file_sha256": old_sha,
                "pre_amendment_file_sha256": hashlib.sha256(edited.encode("utf-8")).hexdigest(),
                "note": "old_file_sha256 is the file at the committed revision before this task; "
                        "pre_amendment_file_sha256 is the file with the edits before this amendments row "
                        f"was appended (cytir.sources.contract amend, task {task})"}

    new = append_amendment_row(text, row)
    problems = (check or check_text)(new)
    if problems:
        raise FamilyContractError("the amended file would fail --check, nothing written: "
                                  + "; ".join(problems[:5]))
    return old, new


def amend_main(argv: list[str]) -> int:
    import difflib
    ap = argparse.ArgumentParser(prog="python3 -m cytir.sources.contract amend",
                                 description="edit a sealed family contract and append the amendments row")
    ap.add_argument("--contracts", default=None, help=f"default {DEFAULT_PATH}")
    ap.add_argument("--family", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=JSON",
                    help="dotted key (a dot inside a key is written \\.) = JSON value; repeatable")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--task", required=True, help="task number, e.g. 2496")
    ap.add_argument("--dry-run", action="store_true", help="print the unified diff, write nothing")
    args = ap.parse_args(argv)
    path = pathlib.Path(args.contracts) if args.contracts else data_path(pathlib.Path(DEFAULT_PATH).name)
    try:
        sets = []
        for item in args.sets:
            key, sep, raw = item.partition("=")
            if not sep or not key:
                raise FamilyContractError(f"--set {item!r}: expected KEY=JSON")
            try:
                sets.append((key, json.loads(raw)))
            except ValueError as exc:
                raise FamilyContractError(f"--set {key}: value is not valid JSON ({exc})") from exc
        dirty = _vcs_dirty(path)
        if dirty:
            raise FamilyContractError(
                f"{path.name} has uncommitted changes ({dirty}); old_file_sha256 must name a "
                "committed revision. Commit or discard them first.")
        old, new = amend(path, args.family, sets, args.reason, args.task.removeprefix("task").strip())
    except FamilyContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                   f"a/{path.name}", f"b/{path.name}"))
        return 0
    path.write_bytes(new.encode("utf-8"))
    print(f"amended {path.name}: {args.family} {[k for k, _ in sets]}; run --check")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "amend":
        return amend_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contracts", default=None,
                    help=f"default {DEFAULT_PATH}")
    ap.add_argument("--check", action="store_true",
                    help="validate the whole document")
    ap.add_argument("--coverage", action="store_true",
                    help="print roster coverage only")
    ap.add_argument("--assert-sets", default=None,
                    help="comma-separated set_ids; exit nonzero if any is "
                         "undeclared or invalid")
    ap.add_argument("--strict", action="store_true",
                    help="an `unknown` value is a FAIL, even for a "
                         "backfilled family")
    args = ap.parse_args()

    try:
        doc = load(args.contracts)
    except FamilyContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    if args.assert_sets:
        try:
            assert_declared([s for s in args.assert_sets.split(",") if s],
                            doc, strict=args.strict)
        except FamilyContractError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
        print("OK: every named set has a valid declaration.")
        return 0

    if args.check:
        path = pathlib.Path(args.contracts) if args.contracts else data_path(
            pathlib.Path(DEFAULT_PATH).name)
        try:
            recorded, computed = pre_amendment_sha256(path.read_text(encoding="utf-8"))
        except FamilyContractError as exc:
            print(f"pre-amendment sha: REFUSED: {exc}", file=sys.stderr)
            return 1
        if recorded != computed:
            print(f"pre-amendment sha: MISMATCH recorded={recorded} computed={computed}",
                  file=sys.stderr)
            return 1
        print("pre-amendment sha: OK")

    report = validate(doc, strict=args.strict)
    if args.coverage:
        print(json.dumps(report["coverage"], indent=1))
        return 0 if report["verdict"] == "PASS" else 1
    format_report(report)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
