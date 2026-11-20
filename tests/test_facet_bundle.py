from __future__ import annotations

import json

import pytest

from cytir.corpus import facet_bundle as fb


def event(value="Registered purpose", *, facet="prose", source="register_field",
          source_id="CHE1", effective="2026-08-03", retrieved="2026-08-03",
          license="OGD", attribution="Zefix", **extra):
    return fb.make_event(
        entity="CHE1", facet=facet, value=value, lang="de", source=source,
        evidence=value, source_url="https://source.example/", source_id=source_id,
        effective_at=effective, retrieved_at=retrieved,
        extractor_version="extract-v1", license=license,
        consent="open-data-license", attribution=attribution, task="test", **extra)


def test_explicit_retraction_is_source_scoped_and_digest_is_deterministic():
    old = event()
    other = event("Vector8", facet="alias", source="wikidata_facet", source_id="Q1")
    retract = event(op="retract", tombstone_of=old["event_id"],
                    effective="2026-08-04", retrieved="2026-08-04")
    rows = fb.compact([old, other, retract])
    assert rows == [other]
    assert fb.bundle_digest(rows) == fb.bundle_digest(list(reversed(rows)))


def test_compaction_orders_by_validated_effective_then_retrieved_time():
    older = event(effective="2026-08-03T10:00:00+02:00", retrieved="2026-08-03T09:00:00Z")
    newer = event(effective="2026-08-03T08:00:00Z", retrieved="2026-08-03T10:00:00Z",
                  attribution="Updated attribution")
    assert fb.compact([newer, older]) == [newer]
    with pytest.raises(ValueError, match="same-time"):
        fb.compact([older, event(effective="2026-08-03T08:00:00Z",
                                 retrieved="2026-08-03T09:00:00Z",
                                 attribution="Conflict")])


def test_retraction_must_name_an_existing_nonfuture_same_scope_target():
    old = event(effective="2026-08-04")
    early = event(op="retract", tombstone_of=old["event_id"],
                  effective="2026-08-03")
    with pytest.raises(ValueError, match="predates"):
        fb.compact([old, early])
    missing = event(op="retract", tombstone_of="f" * 64,
                    effective="2026-08-05")
    with pytest.raises(ValueError, match="target absent"):
        fb.compact([missing])


def test_reconcile_partial_refresh_preserves_unrelated_source_and_emits_lineage():
    homepage = event("Old homepage", source="homepage", source_id="https://example.test/",
                     license="CC-BY-4.0", attribution="Example")
    register = event()
    replacement = event("New homepage", source="homepage",
                        source_id="https://example.test/", effective="2026-08-04",
                        retrieved="2026-08-04", license="CC-BY-4.0",
                        attribution="Example")
    changes = fb.reconcile(
        [homepage, register], [replacement], {("homepage", "prose")},
        task="refresh", effective_at="2026-08-04", retrieved_at="2026-08-04")
    retract = next(row for row in changes if row["op"] == "retract")
    asserted = next(row for row in changes if row["op"] == "assert")
    assert retract["tombstone_of"] == homepage["event_id"]
    assert asserted["supersedes"] == homepage["event_id"]
    current = fb.compact([homepage, register, *changes])
    assert register in current
    assert replacement["value"] in {row["value"] for row in current}
    assert homepage["value"] not in {row["value"] for row in current}


def test_metadata_change_is_a_replacement_not_a_false_noop():
    old = event(license="OGD-v1")
    changed = event(license="OGD-v2", effective="2026-08-04", retrieved="2026-08-04")
    changes = fb.reconcile(
        [old], [changed], {("register_field", "prose")}, task="metadata-refresh",
        effective_at="2026-08-04", retrieved_at="2026-08-04")
    assert [row["op"] for row in changes] == ["retract", "assert"]
    assert changes[0]["tombstone_of"] == old["event_id"]
    assert changes[1]["supersedes"] == old["event_id"]


def test_refresh_scope_does_not_conflate_register_purpose_and_people():
    purpose = event()
    person = event("Marc Maret (President)", facet="person")
    changed = event("New purpose", effective="2026-08-04", retrieved="2026-08-04")
    changes = fb.reconcile(
        [purpose, person], [changed], {("register_field", "prose")}, task="purpose",
        effective_at="2026-08-04", retrieved_at="2026-08-04")
    assert person in fb.compact([purpose, person, *changes])
    assert all(row.get("tombstone_of") != person["event_id"] for row in changes)


def test_manifested_event_wave_is_canonical_input_and_detects_tampering(tmp_path):
    path = tmp_path / "events" / "wave.jsonl"
    row = event()
    manifest = fb.publish_event_wave([row], path, task="test")
    assert manifest["rows"] == 1
    assert fb.load_canonical_events(tmp_path / "events") == [row]
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest/hash"):
        fb.load_event_wave(path)


def test_parquet_requires_real_dependency_not_a_silent_substitute(tmp_path):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pyarrow"):
            fb.write_parquet([], tmp_path / "current.parquet")
    else:
        row = event()
        fb.write_parquet([row], tmp_path / "current.parquet", source_events=[row])
        assert fb.read_parquet(
            tmp_path / "current.parquet", source_events=[row],
            require_event_lineage=True) == [row]
        with pytest.raises(ValueError, match="event-lineage"):
            fb.read_parquet(tmp_path / "current.parquet", source_events=[event("other")])
