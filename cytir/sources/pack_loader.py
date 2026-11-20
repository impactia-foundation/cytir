#!/usr/bin/env python3
"""Task 434 — the BINDING between `sources/**/*.yaml` and the ingest engines.

WHY THIS EXISTS, precisely: the survey in task 434 found that the
engines were never the missing piece.

  * `pack_fetch.py` (task 195 / ADR-026) is already the shared fetch layer --
    robots, L0 retention, outlinks, resume, heartbeat -- and 5 of 86 ingesters
    call it.
  * `cytir/sources/cov/engine.py` (task 351) is already a config-driven multi-route
    ingester reading a routes YAML.

What did not exist was anything that reads a *pack*. Verified 2026-08-20: the
only code that loads `sources/**/*.yaml` is the pack validator (now `cytir.sources.schema`), and
NO code anywhere reads the `adapter:` field. So this module is a binding, not
an engine. It compiles a pack into plans and hands them to machinery that
already works. That sentence is the whole design.

THE LEVEL MATTERS. Acquisition shape belongs to a ROUTE, not a pack.
`sources/ch/ch-media.yaml` reaches srf.ch and rsi.ch by CC-columnar and rts.ch
by sitemap -- one pack, two shapes. A one-shape-per-pack loader would have been
wrong, and only the survey caught it. A pack-level `acquire:` is a DEFAULT that
each route may override.

ONE TRUTH ONE HOME: this module imports identity, canonicalization, tokenization
and chunking; it defines none of them. `ROUTE_PARAMS` is imported from
`cytir.sources.schema` rather than restated. If you find yourself adding
a `uuid5` here, stop -- that is `aw_b1_identity.mint_point_id`.
"""
from __future__ import annotations

from cytir.core.paths import data_path

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from cytir.core.paths import SourcesNotFound, sources_dir
from cytir.sources import schema as _V

try:
    import yaml
except ImportError:  # pragma: no cover
    print("FATAL: pyyaml not installed (pip install pyyaml)", file=sys.stderr)
    raise

# cytir/sources/pack_loader.py -> repo root (moved from pipeline/, task 677: one level deeper).
REPO_ROOT = Path(__file__).resolve().parents[2]


def sources_root() -> Path:
    """Where the pack declarations are (`cytir.core.paths.sources_dir`): resolved at USE, never at
    import, so a family that reads no pack imports this module from a bare `pip install .`."""
    try:
        return sources_dir()
    except SourcesNotFound as exc:
        raise PackError(str(exc)) from exc


ROUTE_PARAMS = _V.ROUTE_PARAMS
ValidationError = _V.ValidationError


PRESETS_PATH = data_path("url-select-presets.v1.json")


def load_presets() -> dict:
    """Named URL-selection presets — ONE home for generic apparatus filtering.

    Loaded from data, not restated in code or in each pack's YAML.
    """
    if not PRESETS_PATH.exists():
        return {"presets": {}, "defaults": {}}
    return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))


class PackError(ValueError):
    """A pack that cannot be turned into a runnable plan."""


# --- the closed predicate vocabulary ----------------------------------------
# Measured over the 97 predicate lambdas in the five run_pack packs (task 434):
# 43 were a constant lang; the rest reduce to exactly these four forms. This is
# why the sitemap/crawl shapes are expressible as data at all.

def compile_select(select: Optional[dict]) -> Callable[[str], bool]:
    """Compile a `select:` block into the include/allow predicate pack_fetch wants."""
    presets = load_presets()
    if not select:
        select = {}
    deny_preset = select.get("deny_preset")
    preset_re = None
    if deny_preset:
        entry = presets.get("presets", {}).get(deny_preset)
        if entry is None:
            # Fail loud: a typo'd preset name must not silently mean "no filter".
            raise PackError(
                f"unknown deny_preset {deny_preset!r}; known: "
                f"{sorted(presets.get('presets', {}))}")
        preset_re = re.compile(entry["deny_regex"])
    min_segs = select.get("min_path_segments")
    if min_segs is None:
        min_segs = (presets.get("defaults", {})
                    .get("min_path_segments", {}).get("value", 0)) if deny_preset else 0
    if not select:
        return lambda u: True
    path_re = re.compile(select["path_regex"]) if select.get("path_regex") else None
    deny_re = (re.compile(select["path_deny_regex"])
               if select.get("path_deny_regex") else None)
    prefixes = tuple(select.get("path_prefix") or ())
    seg1 = set(select.get("segment1_in") or ())

    def pred(url: str) -> bool:
        path = urlsplit(url).path or "/"
        if preset_re and preset_re.search(path):
            return False
        if min_segs and len([p for p in path.split("/") if p]) < min_segs:
            return False
        if deny_re and deny_re.search(path):
            return False
        if path_re and not path_re.search(path):
            return False
        if prefixes and not path.startswith(prefixes):
            return False
        if seg1:
            parts = [p for p in path.split("/") if p]
            if not parts or parts[0] not in seg1:
                return False
        return True

    return pred


def compile_lang(lang: Optional[dict]) -> Callable[[str], Optional[str]]:
    """Compile a `lang:` block into the lang(url) -> str callable pack_fetch wants."""
    if not lang:
        return lambda u: None
    if "const" in lang:
        const = lang["const"]
        return lambda u: const
    mapping = lang.get("by_path_segment") or {}
    default = lang.get("default")

    def pred(url: str) -> Optional[str]:
        for part in (p for p in (urlsplit(url).path or "").split("/") if p):
            if part in mapping:
                return mapping[part]
        return default

    return pred


@dataclasses.dataclass(frozen=True)
class AcquisitionPlan:
    """One executable unit of acquisition: a single route of a single pack."""
    pack_id: str
    pack_path: str
    route_id: str            # the host, or the pack id for a single-route pack
    route: str               # acquisition-policy.v1 route_precedence value
    params: dict
    select: Optional[dict]
    lang: Optional[dict]
    delay_s: Optional[float]
    cap: Optional[float]
    extractor: Optional[str]
    engine: str
    config_ref: Optional[dict]
    reason_receipt: Optional[dict]
    no_fallback: bool
    route_license: Optional[str]
    route_trust: Optional[float]
    # governance / payload fields carried straight through from the pack
    adapter: str
    country: str
    layer: str
    trust_tier: float
    tier: Optional[str]
    license: dict
    consent_basis: str
    consent_train_ok_override: Optional[bool]
    preservation_profiles: list
    freshness: str
    status: str
    host: Optional[str]

    def effective_hosts(self) -> list[str]:
        """Hosts this plan will actually contact.

        `host:` is authoritative when a route declares one. When it does not
        (a dump/api route addressed by URL rather than by host), the hosts are
        derived from the route params -- otherwise a hostless plan would slip
        past the live consent gate, which is the one thing that must never be
        optional. architecture.md §5 step 2: consent is re-resolved live at
        every build, per source.
        """
        seen: list[str] = []
        if self.host:
            seen.append(self.host)
        for key in ("url", "url_template", "endpoint", "index_endpoint"):
            val = self.params.get(key)
            if not isinstance(val, str):
                continue
            netloc = urlsplit(val).netloc
            if netloc and netloc not in seen:
                seen.append(netloc)
        for val in (self.params.get("urls") or []) + (self.params.get("seeds") or []):
            netloc = urlsplit(val).netloc
            if netloc and netloc not in seen:
                seen.append(netloc)
        # Secondary origins are part of the acquisition boundary, not an
        # extractor implementation detail. They must receive the same preflight
        # as the primary host before any cross-host fetch can occur.
        for val in self.params.get("ok_hosts") or []:
            netloc = urlsplit(val).netloc if "://" in val else val
            netloc = netloc.lower().strip().strip(".")
            if netloc and netloc not in seen:
                seen.append(netloc)
        return seen

    # Placeholder values some packs use in `license.default` to mean "the real
    # licence varies per host, see notes". They are NOT licences and must never
    # reach a record: ADR-039 requires a real per-record `license` in the
    # compact redistributable dump.
    LICENSE_PLACEHOLDERS = frozenset({"per-route", "per-host", "varies", "mixed"})

    def effective_license(self) -> Optional[str]:
        """The licence to stamp on this route's records, or raise.

        Fails loud rather than stamping a placeholder. Found the hard way: the
        education pack declares `license.default: per-route` and carries the
        real mapping (us-gov-public-domain / unesco-cc-by-sa-igo) only in prose
        `notes:`. Stamping "per-route" as a licence would be silently wrong in
        the one field ADR-039 exists to protect, and it would look fine.
        """
        route_lic = (self.route_license or "").strip()
        if route_lic:
            return route_lic
        default = (self.license.get("default") or "").strip()
        if not default:
            return None
        if default.lower() in self.LICENSE_PLACEHOLDERS:
            raise PackError(
                f"{self.pack_id}/{self.route_id}: pack license.default is the "
                f"placeholder {default!r}, and this route declares no `license:` "
                f"of its own. A per-record licence is required (ADR-039); "
                f"refusing to stamp a placeholder. Add `license:` to the route.")
        return default

    def effective_trust(self) -> float:
        """Trust for THIS route: the route override if present, else the pack.

        A pack is often a family of hosts at different authority levels. Using
        the pack number for all of them is not a rounding error -- it changes
        ranking. See ADR-098.
        """
        return self.trust_tier if self.route_trust is None else float(self.route_trust)

    def predicate(self) -> Callable[[str], bool]:
        return compile_select(self.select)

    def lang_fn(self) -> Callable[[str], Optional[str]]:
        return compile_lang(self.lang)

    def to_host_cfg(self, mode: str) -> dict:
        """Project this plan into the HOSTS-dict shape `pack_fetch.run_pack` takes.

        This is the actual bridge. pack_fetch's host cfg is currently hand-written
        Python per pack (9 HOSTS dicts across the repo); everything below is
        derived from YAML instead. No new fetch logic -- run_pack does the work.
        """
        cfg = {
            "host": self.host,
            "mode": mode,
            "include": self.predicate(),
            "allow": self.predicate(),
            "lang": self.lang_fn(),
            "source_type": "official",
            "trust": self.effective_trust(),
        }
        if self.tier:
            cfg["tier"] = self.tier
        if self.delay_s is not None:
            cfg["delay"] = self.delay_s
        if self.cap is not None:
            cfg["cap"] = int(self.cap)
        lic = self.effective_license()
        if lic:
            cfg["license"] = lic
        if mode == "sitemap":
            cfg["sitemaps"] = list(self.params.get("urls") or [])
            if self.params.get("expand"):
                cfg["expand"] = True
        elif mode == "bfs":
            cfg["seeds"] = list(self.params.get("seeds") or [])
            hub_path_regex = self.params.get("hub_path_regex")
            if hub_path_regex:
                hub_re = re.compile(hub_path_regex)
                cfg["hub"] = lambda url: bool(
                    hub_re.search(urlsplit(url).path or "/"))
            if self.params.get("hub_depth") is not None:
                cfg["hub_depth"] = self.params["hub_depth"]
        if self.params.get("ok_hosts"):
            cfg["ok_hosts"] = set(self.params["ok_hosts"])
        return cfg


def load_pack(path: str | Path, *, validate: bool = True) -> dict:
    """Load ONE pack YAML, schema-validated by default.

    Fails loud: an invalid pack raises rather than degrading to a partial run.
    A pack that does not validate is not a pack.
    """
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if doc is None:
        raise PackError(f"{path}: empty file")
    if validate:
        schema = _V.load_schema()
        try:
            _V.validate_pack(doc, schema, str(path))
        except ValidationError as e:
            raise PackError(f"{path}: schema invalid: {e}") from e
        try:
            errors, warnings = _V.semantic_checks(doc, str(path))
        except SourcesNotFound as exc:      # a pack's relative pointers need the packs' tree
            raise PackError(f"{path}: {exc}") from exc
        for w in warnings:
            print(f"WARN  {w}", file=sys.stderr)
        if errors:
            raise PackError(f"{path}: " + "; ".join(errors))
    doc["_path"] = str(path)
    return doc


def find_pack(pack_id: str) -> Path:
    """Resolve a pack by its `id:` field, so callers name packs not paths."""
    root = sources_root()
    for p in sorted(root.rglob("*.yaml")):
        if "_schema" in p.parts:
            continue
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and doc.get("id") == pack_id:
            return p
    raise PackError(f"no pack with id {pack_id!r} under {root}")


def plans(doc: dict) -> list[AcquisitionPlan]:
    """Compile a loaded pack into one AcquisitionPlan per acquirable route.

    Route-level `acquire:` wins over the pack-level default -- this is the
    route-not-pack rule, in code. Routes whose status is not runnable
    (declined / not-viable / deferred) are skipped: a pack records history as
    well as intent, and a deferred route is a decision, not a work item.
    """
    pack_acquire = doc.get("acquire")
    routes = doc.get("routes") or []
    common = dict(
        pack_id=doc["id"],
        pack_path=doc.get("_path", ""),
        adapter=doc["adapter"],
        country=doc["country"],
        layer=doc["layer"],
        # NOTE: pack-level default; a route may override it (see route_trust).
        trust_tier=doc["trust_tier"],
        tier=doc.get("tier"),
        license=doc.get("license") or {},
        consent_basis=(doc.get("consent") or {}).get("basis", ""),
        consent_train_ok_override=(doc.get("consent") or {}).get("train_ok_override"),
        preservation_profiles=list(doc.get("preservation_profiles") or []),
        freshness=doc["freshness"],
    )

    RUNNABLE = {"ingested", "staged", "planned"}
    out: list[AcquisitionPlan] = []

    if not routes:
        if not pack_acquire:
            return []
        out.append(AcquisitionPlan(
            route_id=doc["id"], host=None, status=doc["status"],
            route=pack_acquire["route"],
            params=pack_acquire.get("params") or {},
            select=pack_acquire.get("select"), lang=pack_acquire.get("lang"),
            delay_s=pack_acquire.get("delay_s"), cap=pack_acquire.get("cap"),
            extractor=pack_acquire.get("extractor") or doc.get("extractor"),
            engine=pack_acquire.get("engine", "default"),
            config_ref=pack_acquire.get("config_ref"),
            reason_receipt=pack_acquire.get("reason_receipt"),
            no_fallback=bool(pack_acquire.get("no_fallback")),
            route_license=None, route_trust=None, **common))
        return out

    for r in routes:
        acq = r.get("acquire") or pack_acquire
        if not acq:
            continue
        if r.get("status") not in RUNNABLE:
            continue
        out.append(AcquisitionPlan(
            route_id=r.get("host") or doc["id"], host=r.get("host"),
            status=r.get("status"),
            route=acq["route"], params=acq.get("params") or {},
            select=acq.get("select"), lang=acq.get("lang"),
            delay_s=acq.get("delay_s"), cap=acq.get("cap"),
            extractor=acq.get("extractor") or doc.get("extractor"),
            engine=acq.get("engine", "default"),
            config_ref=acq.get("config_ref"),
            reason_receipt=acq.get("reason_receipt"),
            no_fallback=bool(acq.get("no_fallback")
                             or (pack_acquire or {}).get("no_fallback")),
            route_license=r.get("license"),
            route_trust=r.get("trust_tier"),
            **common))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Inspect what a source pack compiles to (task 434).")
    ap.add_argument("pack", help="path to a pack YAML, or its `id:`")
    ap.add_argument("--json", action="store_true", help="emit plans as JSON")
    args = ap.parse_args(argv)

    path = Path(args.pack)
    if not path.exists():
        path = find_pack(args.pack)
    doc = load_pack(path)
    ps = plans(doc)

    if args.json:
        print(json.dumps([dataclasses.asdict(p) for p in ps], indent=2))
        return 0

    print(f"pack   {doc['id']}  ({path})")
    print(f"adapter={doc['adapter']}  country={doc['country']}  "
          f"layer={doc['layer']}  status={doc['status']}")
    if not ps:
        print("\nNO RUNNABLE PLANS — pack declares no `acquire:` block yet.")
        print("It is still documentation. Add an `acquire:` block to make it runnable.")
        return 0
    print(f"\n{len(ps)} plan(s):")
    for p in ps:
        print(f"  - {p.route_id:28s} route={p.route:30s} "
              f"params={sorted(p.params)} extractor={p.extractor or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
