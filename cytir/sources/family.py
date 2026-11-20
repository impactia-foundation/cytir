"""The source-family interface (task 683, task 643 section 6 step S7).

A source family is **a declaration plus a plugin** (architecture-doctrine #3; corpus-doctrine
#7). The declaration is a row of `data/source-family-contracts.v1.json` (consent basis,
licence instrument, representation, chunk policy, identity). The plugin is ONE class that
implements `SourceFamily` below. Nothing else is needed to add a country, a topic or a site:
no `build_*`, `authorize_*` or `run_*` script, no driver of its own.

WHAT THE ADAPTER OWNS AND WHAT THE FRAMEWORK OWNS
-------------------------------------------------
    stage       who          what
    ---------   ----------   ------------------------------------------------------------
    discover    adapter      list the units of work (a sitemap URL, an ELI+language, a CDX
                             record). Cheap, no bytes of content.
    consent     FRAMEWORK    every host the adapter says it will contact is resolved by the
                             ONE resolver (`cytir.consent`, via `cytir.sources.runner`) BEFORE
                             acquire. A decline is a terminal; an unreadable policy is HELD
                             (quarantine), never recorded as the publisher's decision
                             (corpus-doctrine #3).
    acquire     adapter      fetch one unit. Returns `Captured`: the bytes and their
                             transport metadata. The framework SEALS them (content-addressed
                             L0, `cytir.sources.run_family.seal`) before `extract` sees them
                             (corpus-doctrine #6: capture the irreversible at first contact).
    extract     adapter      bytes -> `Doc` (text, title, language, anchor...).
    chunk       adapter      `Doc` -> record dicts, using the chunker the family's DECLARED
                             chunk policy names (`RunContext.chunk_spec`, resolved against
                             the policy registry). A family with no declared policy must say
                             it chunks natively (`CHUNK_NATIVE`) and strict mode refuses it.
    embed       FRAMEWORK    the precondition only (`run_family.assert_embeddable`: recorded
                             declines + the declared consent basis). Vectors: `cyt embed
                             shards`; the index: `cyt index build` (slice 8 step 2).
    compose     FRAMEWORK    one pack: `records.jsonl` + `manifest.json` carrying the
                             contract's digest, the consent verdicts and the terminals.

Every unit reaches exactly ONE terminal (`ADMITTED`, `EXCLUDED` with a reason, or
`QUARANTINED` with a reason; corpus-doctrine #2). The pack manifest prints
`in = admitted + excluded + quarantined` and the run refuses when it does not balance.

The Protocol is deliberately small. A family that needs a bespoke dialect (an API, a dump
format) writes that dialect INSIDE `discover`/`acquire`/`extract`; the framework standardises
everything around the dialect, exactly as `cytir.sources.runner` did for extractor routes.

PARTIAL ADAPTERS (task 1124)
----------------------------
An adapter that does not implement every live stage or dialect says so with ONE class attribute,
`PARTIAL: tuple[str, ...]`, naming what is missing: a stage (`"chunk"`) or a dialect
(`"portal:vd"`). Absent = a full adapter. The refactor scoreboard counts a family "on the Protocol"
as full only without the marker, and lists the partial ones with their missing parts. A stage that
raises a named stop (`StopChunk`, any `*Stop*`) or `NotImplementedError` outright without the
marker fails `tests/test_aw_refactor_scoreboard.py`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Iterable, Iterator, Literal, Mapping, Protocol,
                    runtime_checkable)

#: A family that chunks by its own source structure (a law article, a table row) and so
#: names no policy of the registry. Legitimate, but it must be SAID: the framework records
#: it in the pack manifest, and `--strict` refuses it (an undeclared policy is a refusal).
CHUNK_NATIVE = "native"
#: Like `CHUNK_NATIVE` (the chunk is the source's own article), but an article over the embedder
#: window may be split into several rows: at paragraph marks, into the fewest balanced chunks
#: under the window, the header on every chunk, never cut (POLICY.md §4.1, Tim 2026-10-07, ruling
#: 27). `native` stays for a family whose article is ALWAYS one row; the stage door refuses a
#: `native` pack that holds several rows for one article (`cytir.sources.family_stage`).
CHUNK_NATIVE_CHUNKED = "native-chunked"
#: Both are explicit declared values with no policy file, sha256 or epoch.
NATIVE_POLICIES = (CHUNK_NATIVE, CHUNK_NATIVE_CHUNKED)
#: A family that chunks through the central dispatcher (`cytir.core.common.chunk_markdown_channels`)
#: under whatever the pinned build profile / `AW_CHUNK*` selection resolves to, and whose
#: contract declares no policy of its own. The ACTUAL selection is recorded in the manifest.
CHUNK_DISPATCHER = "dispatcher"
#: A family that chunks under the policy ITS CONTRACT declares (read from the contract, never
#: repeated in the adapter). The contract must declare one; the dispatcher's selection must match.
CHUNK_DECLARED = "declared"

# terminals (corpus-doctrine #2)
ADMITTED = "admitted"
EXCLUDED = "excluded"
QUARANTINED = "quarantined"
TERMINALS = (ADMITTED, EXCLUDED, QUARANTINED)


@dataclass(frozen=True)
class Unit:
    """One unit of work a family discovered.

    `key` is stable across runs (the ledger and resume key on it). `url` is the address the
    adapter will fetch; its host is what consent is resolved for unless `consent_hosts` says
    otherwise. `meta` is the adapter's own bookkeeping, JSON-serialisable.
    """

    key: str
    url: str
    meta: Mapping[str, Any] = field(default_factory=dict)
    consent_hosts: tuple[str, ...] = ()


@dataclass
class Captured:
    """The bytes of one acquired unit and how they arrived: sealed before extract."""

    unit: Unit
    body: bytes
    meta: dict = field(default_factory=dict)      # status, content_type, fetched_at, source_url...


@dataclass
class Doc:
    """What `extract` returns: text plus the handles a record needs. Family-specific keys
    go in `extra`; `chunk` is the only reader of them."""

    url: str
    text: str
    title: str = ""
    lang: str = ""
    anchor: str = ""
    extra: dict = field(default_factory=dict)
    #: The WHOLE extracted text before `chunk` cut it into rows. When set, the framework's
    #: stage-time guard (`cytir.sources.staged_text_guard`) checks that the rows together still hold
    #: it, and quarantines the unit when a cap (not a declared transformation) removed text
    #: (task 706). `None` = the adapter did not give it: the doc is counted as `unchecked`.
    source_text: str | None = None
    #: INSTEAD of `source_text`, for a doc made of many articles (a law): one `SourcePart` per
    #: article, and `part_of(record)` names the part a staged row belongs to. The guard then
    #: compares part by part (task 1045). The doc still counts ONCE in `docs_checked`.
    source_parts: list | None = None
    part_of: Callable[[dict], str] | None = field(default=None, compare=False, repr=False)


class UnitExcluded(Exception):
    """Raised by an adapter's `acquire`/`extract`/`chunk` when the ORIGIN says the unit is not
    there to take (a 404/410, an empty document): the unit is EXCLUDED with this reason. Our own
    failures (a timeout, a 5xx, a parse error) are NOT this: they raise normally and the unit is
    QUARANTINED, because an instrument failure is never a fact about the source."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Terminal:
    unit_key: str
    terminal: str
    reason: str = ""
    records: int = 0

    def __post_init__(self):
        if self.terminal not in TERMINALS:
            raise ValueError(f"unknown terminal {self.terminal!r}; one of {TERMINALS}")
        if self.terminal != ADMITTED and not self.reason:
            raise ValueError("an excluded or quarantined unit needs a reason "
                             "(corpus-doctrine #2: no row stops being mentioned)")


class Ledger:
    """Every discovered unit, and the one terminal it reached."""

    def __init__(self) -> None:
        self._t: dict[str, Terminal] = {}
        self.discovered = 0

    def discover(self, n: int = 1) -> None:
        self.discovered += n

    def close(self, unit_key: str, terminal: str, reason: str = "", records: int = 0) -> None:
        if unit_key in self._t:
            raise RuntimeError(f"unit {unit_key!r} already reached a terminal "
                               f"({self._t[unit_key].terminal}); a unit has exactly one")
        self._t[unit_key] = Terminal(unit_key, terminal, reason, records)

    def terminals(self) -> list[Terminal]:
        return list(self._t.values())

    def summary(self) -> dict:
        reasons: dict[str, dict[str, int]] = {EXCLUDED: {}, QUARANTINED: {}}
        counts = {ADMITTED: 0, EXCLUDED: 0, QUARANTINED: 0}
        records = 0
        for t in self._t.values():
            counts[t.terminal] += 1
            records += t.records
            if t.terminal != ADMITTED:
                reasons[t.terminal][t.reason] = reasons[t.terminal].get(t.reason, 0) + 1
        return {"units_in": self.discovered, **{f"units_{k}": v for k, v in counts.items()},
                "excluded_reasons": dict(sorted(reasons[EXCLUDED].items())),
                "quarantined_reasons": dict(sorted(reasons[QUARANTINED].items())),
                "records": records}

    def assert_balanced(self) -> None:
        s = self.summary()
        got = s["units_admitted"] + s["units_excluded"] + s["units_quarantined"]
        if got != s["units_in"]:
            raise RuntimeError(
                f"conservation failed: in={s['units_in']} but admitted+excluded+quarantined="
                f"{got}; {s['units_in'] - got} unit(s) stopped being mentioned (ADR-086)")


@dataclass
class ChunkSpec:
    """The family's declared chunk policy, resolved against the registry."""

    declared: str                 # the contract's policy_id, or CHUNK_NATIVE
    sha256: str | None = None
    epoch: Any = None
    policy_family: str | None = None
    registry_status: str | None = None

    def as_manifest(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class HttpSpec:
    """How the runner builds a family's HTTP client (task 1556): data on the family, no hook.

    A family that fetches over HTTP declares `http_spec = HttpSpec(...)` as a class attribute
    (an OPTIONAL member, probed with `getattr` like `preflight`). `run_family` builds the client
    from it when `ctx.http` is None, hands it to every stage through `ctx.http`, and closes it
    when the run ends. An injected `ctx.http` (tests, a tape replay) is used untouched. A family
    without a spec gets no client from the runner (it is dump-only or owns a special client).

    `user_agent_kind` picks the identity string, both fail closed on an unconfigured operator:
    `operator` is `cytir.core.user_agent.user_agent(user_agent_component)` (None = no component),
    `pack_fetch` is `pack_fetch_user_agent()`. `headers` are added after the User-Agent.
    `kind="curl"` builds the curl-subprocess client (a host that blocks python TLS clients);
    it takes the timeout only and sends the pack-fetch UA itself.
    """

    kind: Literal["httpx", "curl"] = "httpx"
    timeout: Any = 45.0                     # seconds, or an `httpx.Timeout`
    user_agent_kind: Literal["operator", "pack_fetch"] = "operator"
    user_agent_component: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    follow_redirects: bool = True


@dataclass
class RunContext:
    """What the framework hands every stage of one run.

    `http` is an `httpx.Client` (tests pass one over `httpx.MockTransport`, so a sealed tape
    replays with no network). `consent` is `(host, direct_crawl) -> verdict dict` and defaults
    to the ONE resolver; a verdict's `bucket` is `ok` / `ok_search_only` to proceed.
    """

    out_dir: Path
    today: str
    limit: int = 0                                  # 0 = no limit; N = stop after N units
    http: Any = None
    consent: Callable[[str, bool], dict] | None = None
    skip_consent: bool = False                      # offline replay of already-consented L0
    options: dict = field(default_factory=dict)     # family-specific, from `--opt KEY=VALUE`
    chunk_spec: ChunkSpec | None = None
    contract: dict | None = None                    # the family's row of the contracts file
    notes: dict = field(default_factory=dict)       # discover-time counts etc.; in the manifest
    scratch: dict = field(default_factory=dict)     # adapter-private per-run state; NEVER written to the manifest
    log: Callable[[str], None] = print


@runtime_checkable
class SourceFamily(Protocol):
    """One source family. Implement this class, register it, and declare it in data.

    OPTIONAL stages are NOT members of this Protocol (a member would make the
    `runtime_checkable` `isinstance` refuse every adapter without it). They follow the
    `preflight` convention: the runner probes `getattr(adapter, name, None)` and calls it only
    when present.

    `l1(self, ctx: RunContext) -> dict` (optional, ADR-105 am.1): persist the family's L1 inputs
    (the dump / SPARQL / bulk artefacts a stranger needs to rebuild the family) and return a
    manifest dict. Contract: idempotent; writes only under `ctx.out_dir`; `run_family` calls it
    once, after `preflight` and before `discover` (never on a dry run), and records the returned
    dict under the pack manifest key `l1`. A family without it produces a byte-identical manifest.

    `http_spec: HttpSpec` (optional class attribute, task 1556): how `run_family` builds the
    family's HTTP client. The runner owns the client's life; an adapter uses `ctx.http` and never
    builds a fallback.
    """

    #: the key of the family's row in `data/source-family-contracts.v1.json`
    family_id: str
    #: the `cytir.acquire` route label (`data/acquisition-policy.v1.json`), for the manifest
    route: str
    #: True when we contact the origin ourselves (direct-mode consent); False for Common Crawl
    direct_crawl: bool
    #: the contract policy id this family chunks under, or `CHUNK_NATIVE`
    chunk_policy_id: str

    def discover(self, ctx: RunContext) -> Iterable[Unit]: ...

    def acquire(self, unit: Unit, ctx: RunContext) -> Iterable[Captured]: ...

    def extract(self, captured: Captured, ctx: RunContext) -> Iterable[Doc]: ...

    def chunk(self, doc: Doc, ctx: RunContext) -> Iterable[dict]: ...


def iter_dicts(rows: Iterable[Mapping]) -> Iterator[dict]:
    """Records as plain dicts (a helper for adapters that build them from dataclasses)."""
    for r in rows:
        yield dict(r)
