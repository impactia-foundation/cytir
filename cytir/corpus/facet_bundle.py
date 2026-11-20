"""Canonical append-only entity-card facet events and deterministic compaction.

The JSONL event waves are truth.  Parquet is a reproducible current-state cache;
served points carry only ``bundle_digest``.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from cytir.consent import policy as CP
from cytir.core.digest import canonical_json_utf8
from cytir.corpus import entity_card_website as ECW

EVENT_SCHEMA = "aw.card-facet-event.v1"
CURRENT_SCHEMA = "aw.card-facet-current.v1"
MANIFEST_SCHEMA = "aw.card-facet-wave-manifest.v1"
FACETS = frozenset({
    "prose", "alias", "industry", "product", "category", "taxonomy",
    "person", "geo", "legal_form", "founded", "parent", "finma",
    "predecessor",
})
SOURCES = frozenset({
    "finma_license", "zweck_product", "zweck_category", "wikidata_facet",
    "wikipedia_lead", "register_field", "taxonomy_expansion", "homepage",
    "homepage_l1", "predecessor_inherited", "aw_template",
})
SOURCE_PROJECTION_ORDER = (
    "homepage", "homepage_l1", "wikipedia_lead", "wikidata_facet",
    "register_field", "zweck_category", "zweck_product", "finma_license",
    "taxonomy_expansion", "predecessor_inherited", "aw_template",
)


canonical = canonical_json_utf8


def _timestamp(value: str, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO timestamp")
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        try:
            parsed = dt.datetime.combine(dt.date.fromisoformat(raw), dt.time.min)
        except ValueError as exc:
            raise ValueError(f"{field} is not ISO date/time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def event_id(event: dict) -> str:
    immutable = {k: v for k, v in event.items() if k != "event_id"}
    return hashlib.sha256(canonical(immutable).encode("utf-8")).hexdigest()


def validate_event(event: dict) -> dict:
    if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
        raise ValueError("unknown facet event schema")
    required = (
        "entity", "facet", "source", "source_url", "source_id", "effective_at",
        "retrieved_at", "extractor_version", "license", "consent", "attribution",
        "task", "op", "event_id",
    )
    if any(event.get(key) in (None, "") for key in required):
        raise ValueError("incomplete facet source lifecycle evidence")
    if event["facet"] not in FACETS or event["source"] not in SOURCES:
        raise ValueError("unknown facet/source vocabulary")
    if event["op"] not in {"assert", "retract"}:
        raise ValueError("facet event op must be assert|retract")
    _timestamp(event["effective_at"], "effective_at")
    _timestamp(event["retrieved_at"], "retrieved_at")
    if event["op"] == "assert" and event.get("value") in (None, ""):
        raise ValueError("assert event requires value")
    if event["op"] == "retract" and not event.get("tombstone_of"):
        raise ValueError("retract event requires immutable tombstone_of pointer")
    receipt = _validated_source_receipt(event.get("source_receipt"))
    if receipt is not None:
        if event["source"] not in {"homepage", "homepage_l1"}:
            raise ValueError("website source receipt attached to a non-website event")
        consistent = {"source_url": event["source_url"],
                      "license": event["license"],
                      "attribution": event["attribution"]}
        if any(receipt[key] != value for key, value in consistent.items()):
            raise ValueError("source receipt disagrees with facet provenance")
        association_entity = ECW.entity_key(
            {"register_id": receipt["association_receipt"]["entity"]})
        event_entity = ECW.entity_key({"register_id": event["entity"]})
        if association_entity != event_entity:
            raise ValueError("source receipt association disagrees with facet entity")
        if event["op"] == "assert":
            if receipt["fetched_at"] != event["retrieved_at"]:
                raise ValueError("source receipt fetch time disagrees with facet retrieval")
            content_hash = hashlib.sha256(
                str(event["value"]).encode("utf-8")).hexdigest()
            if receipt["content_sha256"] != content_hash:
                raise ValueError("source receipt hash disagrees with facet value")
        if event.get("artifact_pointer") != receipt["raw_artifact_pointer"]:
            raise ValueError("source receipt raw artifact pointer drift")
    if event["event_id"] != event_id(event):
        raise ValueError("mutable or invalid facet event id")
    return event


def _validated_source_receipt(receipt: dict | None) -> dict | None:
    if receipt is None:
        return None
    if not isinstance(receipt, dict) or receipt.get("schema") != CP.WEBSITE_RECEIPT_SCHEMA:
        raise ValueError("unknown source receipt schema")
    CP.require_resolved_search_index_receipt(receipt)
    required = (
        "source_url", "consent_mode", "license", "attribution", "checked_at", "fetched_at",
        "content_sha256", "raw_sha256", "raw_artifact_pointer",
        "association_receipt", "license_evidence", "acquisition_receipt",
    )
    if any(receipt.get(key) in (None, "") for key in required):
        raise ValueError("incomplete source receipt evidence")
    if not all(isinstance(receipt[key], str)
               and re.fullmatch(r"[0-9a-fA-F]{64}", receipt[key])
               for key in ("content_sha256", "raw_sha256")):
        raise ValueError("source receipt hashes must be SHA-256 hex digests")
    association_seed = {
        "url": receipt["source_url"], "domain": receipt["association_receipt"]["domain"],
        "register_id": receipt["association_receipt"]["entity"],
    }
    ECW.validate_association_receipt(
        association_seed, receipt["association_receipt"])
    ECW.validate_license_evidence(
        receipt["source_url"], receipt["license"], receipt["license_evidence"])
    ECW.validate_acquisition_receipt(
        receipt["source_url"], receipt["fetched_at"], receipt["consent_mode"],
        receipt["acquisition_receipt"])
    return dict(receipt)


def make_event(*, entity, facet, value, lang, source, evidence, source_url,
               source_id, effective_at, retrieved_at, extractor_version, license,
               consent, attribution, task, op="assert", supersedes=None,
               tombstone_of=None, artifact_pointer=None, selection=None,
               source_receipt=None):
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) - {
                "depth", "rule", "token_count"}:
            raise ValueError("invalid bounded-page selection evidence")
        selection = {key: selection.get(key) for key in
                     ("depth", "rule", "token_count")}
    event = {
        "schema": EVENT_SCHEMA, "entity": entity, "facet": facet, "value": value,
        "lang": lang or "und", "source": source, "evidence": evidence,
        "source_url": source_url, "source_id": source_id,
        "effective_at": effective_at, "retrieved_at": retrieved_at,
        "extractor_version": extractor_version, "license": license,
        "consent": consent, "attribution": attribution, "task": task, "op": op,
        "supersedes": supersedes, "tombstone_of": tombstone_of,
        "artifact_pointer": artifact_pointer, "selection": selection,
        "source_receipt": _validated_source_receipt(source_receipt),
    }
    event["event_id"] = event_id(event)
    return validate_event(event)


def scope_key(event: dict) -> tuple:
    return (event["entity"], event["facet"], event["source"],
            event["source_id"], event.get("lang") or "und",
            " ".join(str(event.get("value") or "").split()).casefold())


def stream_key(event: dict) -> tuple:
    """Source stream independently refreshed by one extractor invocation."""
    return (event["entity"], event["facet"], event["source"],
            event["source_id"], event.get("lang") or "und")


def projection_key(event: dict) -> tuple:
    """Stable human/content composition order, independent of event arrival."""
    return (SOURCE_PROJECTION_ORDER.index(event["source"]), event["facet"],
            event.get("lang") or "und",
            " ".join(str(event.get("value") or "").split()).casefold(),
            event["source_id"])


def _precedence(event: dict) -> tuple:
    return (_timestamp(event["effective_at"], "effective_at"),
            _timestamp(event["retrieved_at"], "retrieved_at"))


def compact(events) -> list[dict]:
    """Fold events into current assertions with explicit lifecycle semantics."""
    by_id = {}
    ordered = []
    for ordinal, raw in enumerate(events):
        event = validate_event(dict(raw))
        if event["event_id"] in by_id:
            raise ValueError("duplicate immutable facet event id")
        by_id[event["event_id"]] = event
        ordered.append((ordinal, event))
    retracted = set()
    for _, event in ordered:
        if event["op"] == "retract":
            target = event["tombstone_of"]
            if target not in by_id:
                raise ValueError(f"retraction target absent from canonical log: {target}")
            if scope_key(by_id[target]) != scope_key(event):
                raise ValueError("retraction crosses facet/source scope")
            if _precedence(event) < _precedence(by_id[target]):
                raise ValueError("retraction predates its immutable target")
            retracted.add(target)
        if event.get("supersedes"):
            target = event["supersedes"]
            if target not in by_id:
                raise ValueError(f"supersedes target absent from canonical log: {target}")
            if stream_key(by_id[target]) != stream_key(event):
                raise ValueError("supersedes crosses facet/source stream")
            if _precedence(event) < _precedence(by_id[target]):
                raise ValueError("replacement predates its immutable predecessor")
    candidates = [(ordinal, event) for ordinal, event in ordered
                  if event["op"] == "assert" and event["event_id"] not in retracted]
    current = {}
    for ordinal, event in candidates:
        key = scope_key(event)
        previous = current.get(key)
        if previous is None:
            current[key] = (ordinal, event)
            continue
        prev_ordinal, prev = previous
        left, right = _precedence(prev), _precedence(event)
        if right == left and event["event_id"] != prev["event_id"]:
            raise ValueError("conflicting same-time assertions require explicit retraction")
        if right > left:
            current[key] = (ordinal, event)
    return sorted((event for _, event in current.values()),
                  key=lambda event: (event["entity"], event["facet"], event["source"],
                                     event["source_id"], event["lang"],
                                     event["effective_at"], event["retrieved_at"]))


def _semantic_assertion(event: dict) -> str:
    """Content/provenance equality, excluding append-lifecycle coordinates."""
    ignored = {
        "schema", "event_id", "effective_at", "retrieved_at", "task", "op",
        "supersedes", "tombstone_of",
    }
    return canonical({key: value for key, value in event.items()
                      if key not in ignored})


def _refresh_match(event: dict, selectors: set[tuple[str, str | None]]) -> bool:
    return ((event["source"], event["facet"]) in selectors
            or (event["source"], None) in selectors)


def _normalize_refresh_scopes(refreshed_sources) -> set[tuple[str, str | None]]:
    selectors = set()
    for value in refreshed_sources:
        if isinstance(value, str):
            source, facet = value, None
        elif (isinstance(value, (tuple, list)) and len(value) == 2
              and all(part is None or isinstance(part, str) for part in value)):
            source, facet = value
        else:
            raise ValueError(f"invalid refresh scope: {value!r}")
        if source not in SOURCES or (facet is not None and facet not in FACETS):
            raise ValueError(f"unknown refresh source/facet: {value!r}")
        selectors.add((source, facet))
    return selectors


def _replace_pointer(event: dict, predecessor: str) -> dict:
    replacement = dict(event)
    replacement.pop("event_id", None)
    replacement["supersedes"] = predecessor
    replacement["event_id"] = event_id(replacement)
    return validate_event(replacement)


def reconcile(current_rows: list[dict], proposed: list[dict], refreshed_sources,
              *, task: str, effective_at: str, retrieved_at: str) -> list[dict]:
    """Emit explicit source/facet retractions; preserve unrelated assertions."""
    selectors = _normalize_refresh_scopes(refreshed_sources)
    current, fresh = {}, {}
    for destination, rows, label in (
            (current, current_rows, "current"), (fresh, proposed, "proposed")):
        for raw in rows:
            row = validate_event(dict(raw))
            key = scope_key(row)
            if key in destination:
                raise ValueError(f"duplicate {label} assertion scope: {key!r}")
            destination[key] = row
    if any(not _refresh_match(row, selectors) for row in fresh.values()):
        raise ValueError("proposed event source/facet missing from refresh scopes")
    emitted = []
    replaced_by_key = {}
    retracted_by_stream = {}
    for key, old in current.items():
        if not _refresh_match(old, selectors):
            continue
        new = fresh.get(key)
        if new and _semantic_assertion(new) == _semantic_assertion(old):
            fresh.pop(key)              # semantic no-op; retain old lifecycle evidence
            continue
        emitted.append(make_event(
            entity=old["entity"], facet=old["facet"], value=old.get("value"),
            lang=old["lang"], source=old["source"], evidence="source refresh retraction",
            source_url=old["source_url"], source_id=old["source_id"],
            effective_at=effective_at, retrieved_at=retrieved_at,
            extractor_version=old["extractor_version"], license=old["license"],
            consent=old["consent"], attribution=old["attribution"], task=task,
            op="retract", tombstone_of=old["event_id"],
            artifact_pointer=old.get("artifact_pointer"),
            selection=old.get("selection"),
            source_receipt=old.get("source_receipt")))
        retracted_by_stream.setdefault(stream_key(old), []).append(old["event_id"])
        if new is not None:
            replaced_by_key[key] = old["event_id"]
    fresh_by_stream = {}
    for key, event in fresh.items():
        fresh_by_stream.setdefault(stream_key(event), []).append(key)
    for key, event in fresh.items():
        predecessor = replaced_by_key.get(key)
        if predecessor is None:
            prior = retracted_by_stream.get(stream_key(event), [])
            if len(prior) == 1 and len(fresh_by_stream[stream_key(event)]) == 1:
                predecessor = prior[0]
        emitted.append(_replace_pointer(event, predecessor) if predecessor else event)
    return emitted


def bundle_digest(rows: list[dict]) -> str:
    normalized = sorted((validate_event(dict(row)) for row in rows), key=scope_key)
    return hashlib.sha256(canonical(normalized).encode("utf-8")).hexdigest()


def event_set_digest(events: list[dict]) -> str:
    normalized = sorted((validate_event(dict(row)) for row in events),
                        key=lambda row: row["event_id"])
    return hashlib.sha256(canonical(normalized).encode("utf-8")).hexdigest()


def entity_bundle(rows, entity):
    selected = [validate_event(dict(row)) for row in rows if row["entity"] == entity]
    selected.sort(key=scope_key)
    return selected, bundle_digest(selected)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as tmp:
        tmp.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temp = Path(tmp.name)
    os.replace(temp, path)


def publish_event_wave(events: list[dict], path, *, task: str) -> dict:
    """Atomically publish one immutable append-only wave plus manifest."""
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"append-only facet wave already exists: {path}")
    validated = [validate_event(dict(event)) for event in events]
    if any(event["task"] != task for event in validated):
        raise ValueError("facet wave task must match every immutable event")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as tmp:
        for event in validated:
            tmp.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(tmp.name)
    os.replace(temp, path)
    manifest = {"schema": MANIFEST_SCHEMA, "task": task, "path": path.name,
                "rows": len(validated), "sha256": _sha256(path),
                "event_ids_sha256": hashlib.sha256(
                    "\n".join(event["event_id"] for event in validated).encode()).hexdigest()}
    _atomic_json(manifest_path, manifest)
    return manifest


def load_event_wave(path) -> list[dict]:
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"facet wave manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("sha256") != _sha256(path):
        raise ValueError(f"facet wave manifest/hash mismatch: {path}")
    rows = [validate_event(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest.get("rows"):
        raise ValueError(f"facet wave row-count mismatch: {path}")
    ids_digest = hashlib.sha256(
        "\n".join(event["event_id"] for event in rows).encode()).hexdigest()
    if manifest.get("event_ids_sha256") != ids_digest or manifest.get("path") != path.name:
        raise ValueError(f"facet wave identity manifest mismatch: {path}")
    if not manifest.get("task") or any(row["task"] != manifest["task"] for row in rows):
        raise ValueError(f"facet wave task manifest mismatch: {path}")
    return rows


def load_canonical_events(root) -> list[dict]:
    files = sorted(Path(root).glob("**/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no canonical facet event waves under {root}")
    return [event for path in files for event in load_event_wave(path)]


def write_parquet(rows, path, *, source_events: list[dict] | None = None) -> dict:
    """Atomically materialize the compact cache; never silently use JSON."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for facet parquet compaction; refusing JSON substitute") from exc
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite compacted facet table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = [validate_event(dict(row)) for row in rows]
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".",
                                     delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        pq.write_table(pa.Table.from_pylist(validated), temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    manifest = {
        "schema": CURRENT_SCHEMA, "rows": len(validated), "sha256": _sha256(path),
        "bundle_set_sha256": bundle_digest(validated),
        "source_events_sha256": (event_set_digest(source_events)
                                 if source_events is not None else None),
        "source_event_rows": (len(source_events) if source_events is not None else None),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def read_parquet(path, *, source_events: list[dict] | None = None,
                 require_event_lineage: bool = False) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read facet current-state parquet") from exc
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"compacted facet manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CURRENT_SCHEMA or manifest.get("sha256") != _sha256(path):
        raise ValueError("compacted facet parquet manifest/hash mismatch")
    rows = [validate_event(row) for row in pq.read_table(path).to_pylist()]
    if len(rows) != manifest.get("rows"):
        raise ValueError("compacted facet parquet row-count mismatch")
    if manifest.get("bundle_set_sha256") != bundle_digest(rows):
        raise ValueError("compacted facet parquet bundle-set mismatch")
    if require_event_lineage and not manifest.get("source_events_sha256"):
        raise ValueError("compacted facet parquet lacks canonical event lineage")
    if source_events is not None:
        if (manifest.get("source_events_sha256") != event_set_digest(source_events)
                or manifest.get("source_event_rows") != len(source_events)):
            raise ValueError("compacted facet parquet event-lineage mismatch")
        expected = compact(source_events)
        if bundle_digest(expected) != bundle_digest(rows):
            raise ValueError("compacted facet parquet is not the canonical event fold")
    return rows
