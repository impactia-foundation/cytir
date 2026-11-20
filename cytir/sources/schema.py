"""Validate `sources/**/*.yaml` pack files against the pack schema (task 311; moved into the package by task 735).

Minimal, dependency-free JSON-Schema-subset validator. No `jsonschema` dependency, because the
schema is small and the validator must run on a clean `pip install .`. It implements exactly the
draft-07 features `pack.schema.json` uses: type, enum, pattern, format=date, minimum/maximum,
required, additionalProperties=false, properties, items, minItems, $ref into #/definitions.

WHERE THINGS COME FROM (task 735). The package never imports from `sources/`:
  * the schema is package data (`cytir/_data/pack.schema.json`, a byte-for-byte copy of
    `sources/_schema/pack.schema.json`; tests/test_aw_sources_install.py holds the two equal);
  * the policy and preservation registries come from `cytir.core.paths.data_path()`;
  * the pack YAML files come from `cytir.core.paths.sources_dir()` (`AW_SOURCES_DIR`, else the checkout's
    `sources/`), and a pack's relative pointers (`extractor`, `config_ref`) resolve against that
    directory's parent, which is the repo root in a clone.
`sources/_schema/validate.py` is a thin shim over this module.

Usage:
    python3 -m cytir.sources.schema                 # validate every pack under sources/
    python3 -m cytir.sources.schema sources/ch/*.yaml
Exit 0 = all valid, 1 = at least one invalid.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("FATAL: pyyaml not installed (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)

from cytir.core.paths import data_path, sources_dir

SCHEMA_PATH = data_path("pack.schema.json")
POLICY_PATH = data_path("acquisition-policy.v1.json")
PRESERVATION_PROFILES_PATH = data_path("source-preservation-profiles.v1.json")


def tree_root() -> Path:
    """The directory a pack's relative pointers (`extractor`, `config_ref.path`) resolve against:
    the parent of the packs directory (the repo root in a clone). Refuses when no packs are found."""
    return sources_dir().parent

# --- the route -> params contract (task 434) ---------------------------------
# ONE HOME. cytir.sources.pack_loader imports ROUTE_PARAMS from this module rather
# than restating it; a second copy would be a forked contract (doctrine 1).
# `required` is what the runner cannot proceed without; `allowed` bounds the rest
# so a typo fails loudly instead of being silently ignored.
ROUTE_PARAMS = {
    "source_owned_sitemap_fetch": {
        "required": {"urls"},
        "allowed": {"urls", "expand", "ok_hosts"},
    },
    "common_crawl": {
        "required": set(),
        "allowed": {"path_prefixes", "crawl", "index_mode", "host_scope"},
    },
    "source_owned_bulk_dump": {
        # `url` for a single fixed artifact; `url_template` for a per-item
        # structured dump whose members are addressed by id (fedlex: one Akoma
        # Ntoso XML per ELI+date). `index_endpoint` records the machine route
        # that ENUMERATES those ids -- for fedlex a SPARQL endpoint. That
        # two-step shape is not an anomaly: the survey found cytir.extract.swisstopo
        # doing the same thing (a STAC api call whose only job is resolving a
        # bulk_dump URL), which is why it is modelled rather than flattened.
        "required": {"format"},
        "one_of": [{"url", "url_template"}],
        # `transport` because a bulk dump is not always an HTTP GET. The Italian
        # Senate publishes 1.71 GB of Akoma Ntoso via a plain `git clone`
        # (UK/US/IT gov lane, 2026-08-20) -- a dump by every meaningful
        # definition whose transport happens to be git. Flattening that to
        # "download a URL" would have forced that lane back into a bespoke
        # script, which is the exact failure this tool exists to end.
        "allowed": {"url", "url_template", "format", "member_glob",
                    "checksum_url", "index_endpoint", "ids",
                    "transport", "ref", "subpath"},
    },
    "documented_source_api": {
        "required": {"endpoint"},
        "allowed": {"endpoint", "dialect", "paginate", "query", "headers"},
    },
    "source_owned_incremental_feed": {
        "required": {"url"},
        "allowed": {"url", "kind", "since"},
    },
    "direct_origin_crawl": {
        # Either explicit seeds, or a generated URL pattern. The generated form
        # is not new: cytir/sources/cov/engine.py's `urllist` mode already did it for
        # NORMLEX instrument codes. Gazzetta Ufficiale (IT gov lane) is
        # date-based path enumeration with ELI/RDFa metadata -- closer to a
        # crawl than a sitemap, and expressible only with a pattern.
        "required": set(),
        "one_of": [{"seeds", "url_pattern"}],
        "allowed": {"seeds", "hub_path_regex", "hub_depth", "ok_hosts",
                    "url_pattern", "date_from", "date_to", "date_format"},
    },
}


class ValidationError(Exception):
    def __init__(self, path: str, msg: str):
        super().__init__(f"{path}: {msg}")
        self.path = path
        self.msg = msg


def _check_type(value, schema_type, path):
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    pytypes = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    for t in types:
        expect = pytypes[t]
        if t == "number" and isinstance(value, bool):
            continue  # bool is not a number for our purposes
        if isinstance(value, expect):
            return
    raise ValidationError(path, f"expected type {types}, got {type(value).__name__} ({value!r})")


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate(value, schema, path, root=None):
    # Local $ref support (task 434): the `acquire` block is defined once in
    # $defs and referenced from both the pack level and each routes[] entry.
    # Inlining it twice would fork the contract it exists to enforce.
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValidationError(path, f"unsupported non-local $ref {ref!r}")
        target = root if root is not None else {}
        for part in ref[2:].split("/"):
            target = target[part]
        schema = target

    if "type" in schema:
        _check_type(value, schema["type"], path)

    if value is None:
        return

    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(path, f"{value!r} not in enum {schema['enum']}")

    if "pattern" in schema and isinstance(value, str):
        if not re.match(schema["pattern"], value):
            raise ValidationError(path, f"{value!r} does not match pattern {schema['pattern']!r}")

    if schema.get("format") == "date" and isinstance(value, str):
        if not DATE_RE.match(value):
            raise ValidationError(path, f"{value!r} is not YYYY-MM-DD")
        try:
            y, m, d = (int(x) for x in value.split("-"))
            date(y, m, d)
        except ValueError as e:
            raise ValidationError(path, f"{value!r} is not a real date: {e}")

    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        raise ValidationError(path, f"{value} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        raise ValidationError(path, f"{value} > maximum {schema['maximum']}")

    if isinstance(value, dict) and schema.get("type") == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                raise ValidationError(path, f"missing required property '{req}'")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise ValidationError(path, f"unexpected propert{'y' if len(extra)==1 else 'ies'}: {sorted(extra)}")
        for k, v in value.items():
            if k in props:
                _validate(v, props[k], f"{path}.{k}", root)

    if isinstance(value, list) and schema.get("type") == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationError(path, f"array has {len(value)} items, need >= {schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]", root)


def validate_pack(doc, schema, label):
    _validate(doc, schema, label, schema)


def load_policy_routes():
    """Route enum + ranks, read from data/acquisition-policy.v1.json.

    The schema restates the enum for editor tooling, but this file is the
    authority. Reading it here means a policy re-rank cannot silently disagree
    with the packs -- config is data, and it has one home.
    """
    if not POLICY_PATH.exists():
        return None
    policy = json.loads(POLICY_PATH.read_text())
    return {r["route"]: r["rank"] for r in policy.get("route_precedence", [])}


def load_preservation_profiles():
    """Preservation-profile names, read from data/source-preservation-profiles.v1.json.

    The registry is the authority (doctrine 1). The schema file restates the
    enum for editor tooling, but validation must not gate on that hand-copied
    list -- a profile added to the registry must be declarable immediately,
    not weeks after someone remembers to edit the schema. Task 437.
    """
    if not PRESERVATION_PROFILES_PATH.exists():
        return None
    registry = json.loads(PRESERVATION_PROFILES_PATH.read_text())
    return sorted(registry.get("profiles", {}))


def load_schema():
    """Load pack.schema.json with the preservation-profile enum bound to the registry.

    Every consumer that validates packs (this module's main() and
    cytir.sources.pack_loader.load_pack) must call this rather than read the schema
    file directly, so the registry has exactly one home. The schema file's own
    enum stays for editor tooling; the schema-vs-registry test keeps that copy
    from silently drifting. Task 437.
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    profiles = load_preservation_profiles()
    if profiles is not None:
        schema["properties"]["preservation_profiles"]["items"]["enum"] = profiles
    return schema


CRAWL_ROUTES = {"direct_origin_crawl"}


def _check_no_fallback(doc, label, errors):
    """Enforce a pack's commitment not to crawl.

    Requested as a HARD requirement by the company-register lane (2026-08-20):
    the Canadian pack commits to never crawling the HTML detail pages, and that
    commitment must be "expressible and enforceable, not merely conventional".

    A promise a tool cannot check is a comment. So: if ANY acquire block in the
    pack sets no_fallback, then NO route in that pack may be a crawl route.
    Checked statically at validation, so the contradiction cannot reach a run.
    """
    blocks = []
    if isinstance(doc.get("acquire"), dict):
        blocks.append(("acquire", doc["acquire"]))
    for i, r in enumerate(doc.get("routes") or []):
        if isinstance(r, dict) and isinstance(r.get("acquire"), dict):
            blocks.append((f"routes[{i}].acquire", r["acquire"]))

    if not any(b.get("no_fallback") for _, b in blocks):
        return
    offenders = [where for where, b in blocks if b.get("route") in CRAWL_ROUTES]
    if offenders:
        errors.append(
            f"{label}: pack declares no_fallback, but {offenders} use a crawl "
            f"route ({sorted(CRAWL_ROUTES)}). no_fallback is a commitment not to "
            f"crawl; a pack cannot hold both.")


def _check_acquire(acq, label, ranks, errors, warnings):
    route = acq.get("route")
    if ranks is not None and route not in ranks:
        errors.append(
            f"{label}.route: {route!r} is not a route in "
            f"{POLICY_PATH.name} (have: {sorted(ranks)})")
        return
    if "rank" in acq and ranks is not None and acq["rank"] != ranks[route]:
        errors.append(
            f"{label}.rank: declared {acq['rank']}, but {POLICY_PATH.name} "
            f"ranks {route!r} at {ranks[route]}")

    spec = ROUTE_PARAMS.get(route)
    params = acq.get("params", {})
    if spec:
        missing = spec["required"] - set(params)
        if missing:
            errors.append(
                f"{label}.params: route {route!r} requires {sorted(missing)}")
        for group in spec.get("one_of", []):
            present = group & set(params)
            if len(present) != 1:
                errors.append(
                    f"{label}.params: route {route!r} needs exactly one of "
                    f"{sorted(group)} (found {sorted(present) or 'none'})")
        unknown = set(params) - spec["allowed"]
        if unknown:
            errors.append(
                f"{label}.params: unknown key(s) {sorted(unknown)} for route "
                f"{route!r}; allowed: {sorted(spec['allowed'])}")

    # acquisition-policy.v1 direct_fetch.required_reason_fields.
    # WARN-ONLY for now (orchestrator ruling 2026-08-20): 16 existing ingesters
    # owe a receipt. A validator that fails everything on day one is a validator
    # someone disables. Flip to error once the offenders are fixed or waived.
    if route == "direct_origin_crawl" and "reason_receipt" not in acq:
        warnings.append(
            f"{label}: route=direct_origin_crawl (policy rank 6) without a "
            f"reason_receipt. acquisition-policy.v1 requires "
            f"higher_priority_routes_checked + reason_selected_route_is_needed. "
            f"WARN-ONLY today; this becomes an error.")

    config_ref = acq.get("config_ref")
    if config_ref:
        root = tree_root()
        path = (root / config_ref["path"]).resolve()
        try:
            path.relative_to(root.resolve())
            inside_repo = True
        except ValueError:
            inside_repo = False
        if not inside_repo or not path.is_file():
            errors.append(
                f"{label}.config_ref.path: {config_ref['path']!r} does not "
                f"resolve to a file under the repository")
        if acq.get("engine", "default") != "extractor":
            errors.append(
                f"{label}.config_ref: extractor-owned config requires "
                f"engine=extractor; the default engine does not consume it")


def semantic_checks(doc, label):
    """Cross-file checks the JSON-Schema subset cannot express.

    Returns (errors, warnings).
    """
    errors, warnings = [], []
    ranks = load_policy_routes()
    if ranks is None:
        warnings.append(f"{label}: {POLICY_PATH} not found; route checks skipped")

    if isinstance(doc.get("acquire"), dict):
        _check_acquire(doc["acquire"], f"{label}.acquire", ranks, errors, warnings)

    for i, route in enumerate(doc.get("routes") or []):
        if isinstance(route, dict) and isinstance(route.get("acquire"), dict):
            _check_acquire(route["acquire"], f"{label}.routes[{i}].acquire",
                           ranks, errors, warnings)

    _check_no_fallback(doc, label, errors)

    # extractor pointers must resolve -- a dangling one is an unrunnable pack
    for field_label, value in [(f"{label}.extractor", doc.get("extractor"))]:
        if value and not (tree_root() / value).exists():
            errors.append(f"{field_label}: {value!r} does not exist")
    return errors, warnings


def main(argv):
    schema = load_schema()

    if argv:
        paths = [Path(p) for p in argv]
    else:
        sources_root = sources_dir()
        paths = sorted(
            p for p in sources_root.rglob("*.yaml") if "_schema" not in p.parts
        )

    if not paths:
        print("No pack YAML files found.")
        return 0

    n_ok = 0
    errors = []
    warnings = []
    for p in paths:
        try:
            doc = yaml.safe_load(p.read_text())
        except yaml.YAMLError as e:
            errors.append(f"{p}: YAML parse error: {e}")
            continue
        if doc is None:
            errors.append(f"{p}: empty file")
            continue
        try:
            validate_pack(doc, schema, str(p))
        except ValidationError as e:
            errors.append(str(e))
            continue
        sem_errors, sem_warnings = semantic_checks(doc, str(p))
        warnings.extend(sem_warnings)
        if sem_errors:
            errors.extend(sem_errors)
            continue
        n_ok += 1
        print(f"OK    {p}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"WARN  {w}")

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"FAIL  {e}", file=sys.stderr)
        print(f"\n{n_ok} ok, {len(errors)} failed, {len(paths)} total")
        return 1

    print(f"\n{n_ok} ok, {len(paths)} total — all packs valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
