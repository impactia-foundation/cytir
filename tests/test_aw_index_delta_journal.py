"""Pure contract tests for the task-367 chunk@2 delta journal."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args: str):
    return subprocess.run([sys.executable, "-m", "cytir.cli", "index", "delta-journal", *args], text=True,
                          capture_output=True, cwd=ROOT.parent)


def write(path: Path, value: str):
    path.write_text(value, encoding="utf-8")
    return path


def content_row():
    return json.dumps({"id": "new-chunk-id", "vector": {"dense": [0.1]},
                       "payload": {"url": "https://example.test/a", "source_type": "web",
                                   "stamps": {"chunk": 2}}}) + "\n"


def initialize(tmp_path: Path):
    cutoff = write(tmp_path / "base.json", json.dumps(base_cutoff()))
    journal = tmp_path / "journal"
    got = run("init", "--journal-dir", str(journal), "--cutoff-manifest", str(cutoff),
              "--created-at", "2026-07-31T00:00:00Z")
    assert got.returncode == 0, got.stderr
    return journal


def base_cutoff():
    receipt = {"status": "TWIN-PROD-VERIFIED-OK", "exact_post_apply_count": 10,
               "receipt_path": "/evidence/receipt.json", "receipt_sha256": "a" * 64}
    return {"schema": "aw.chunk2-base-candidate.v1", "app_git_sha": "b" * 40,
            "created_at": "2026-08-02T00:00:00Z",
            "base_snapshot": {"collection": "aw_b1", "qdrant_version": "1.18.3",
                              "snapshot_sha256": "c" * 64, "exact_count": 10},
            "settled_landing_operations": [dict(receipt, operation=name) for name in
                                           ("fedlex-358b", "register-363-dates-status", "register-363-people")],
            "first_journaled_operation": "task359-intl",
            "source_census": {"path": "/evidence/census.json", "sha256": "d" * 64},
            "inputs": [{"wave": "w2", "route": "dump", "sha256": "e" * 64}],
            "chunk": {"epoch": 2, "config_sha256": "f" * 64, "rule_legend": "struct-md-768w-o0@2"},
            "embed": {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "revision": "0" * 40,
                      "mrl_dim": 128, "max_seq": 2000}}


def test_init_rejects_unsettled_or_task359_inclusive_cutoff(tmp_path):
    bad = base_cutoff()
    bad["settled_landing_operations"] = bad["settled_landing_operations"][:-1]
    cutoff = write(tmp_path / "bad.json", json.dumps(bad))
    got = run("init", "--journal-dir", str(tmp_path / "journal"), "--cutoff-manifest", str(cutoff))
    assert got.returncode != 0
    assert "Fedlex and two Task363" in got.stderr

    bad = base_cutoff()
    bad["first_journaled_operation"] = "task359-already-applied"
    cutoff = write(tmp_path / "bad2.json", json.dumps(bad))
    got = run("init", "--journal-dir", str(tmp_path / "journal2"), "--cutoff-manifest", str(cutoff))
    assert got.returncode != 0
    assert "Task359" in got.stderr


def record_content(tmp_path: Path, journal: Path):
    source = write(tmp_path / "source.jsonl", '{"url":"https://example.test/a","markdown":"x"}\n')
    inp = write(tmp_path / "replay.jsonl", '{"url":"https://example.test/a"}\n')
    points = write(tmp_path / "points.jsonl", content_row())
    return run("record", "--journal-dir", str(journal), "--operation-id", "pack-1",
               "--mode", "content-rechunk-reembed", "--source", str(source),
               "--source-git-sha", "a" * 40, "--artifact", f"replay_input={inp}",
               "--artifact", f"chunk2_records={points}", "--recorded-at", "2026-07-31T01:00:00Z")


def test_content_record_copies_artifacts_and_verifies(tmp_path):
    journal = initialize(tmp_path)
    got = record_content(tmp_path, journal)
    assert got.returncode == 0, got.stderr
    # The journal uses a private copy, so a later source edit cannot change it.
    (tmp_path / "points.jsonl").write_text(content_row().replace("0.1", "0.2"), encoding="utf-8")
    checked = run("verify", "--journal-dir", str(journal))
    assert checked.returncode == 0, checked.stderr
    row = json.loads((journal / "journal.jsonl").read_text())
    assert row["sequence"] == 1
    assert row["source"]["git_sha"] == "a" * 40
    assert {x["role"] for x in row["artifacts"]} == {"replay_input", "chunk2_records"}


def test_record_rejects_chunk1_or_legacy_point_identity(tmp_path):
    journal = initialize(tmp_path)
    source = write(tmp_path / "source", "source")
    inp = write(tmp_path / "input", "input")
    bad_points = write(tmp_path / "bad.jsonl", content_row().replace('"chunk": 2', '"chunk": 1'))
    got = run("record", "--journal-dir", str(journal), "--operation-id", "bad",
              "--mode", "content-rechunk-reembed", "--source", str(source),
              "--source-git-sha", "b" * 40, "--artifact", f"replay_input={inp}",
              "--artifact", f"chunk2_records={bad_points}")
    assert got.returncode != 0
    assert "not stamped chunk@2" in got.stderr

    patch = write(tmp_path / "legacy.jsonl", '{"id":"old-point","set":{"country":"CH"}}\n')
    got = run("record", "--journal-dir", str(journal), "--operation-id", "legacy",
              "--mode", "payload-only", "--source", str(source),
              "--source-git-sha", "b" * 40, "--artifact", f"payload_patch={patch}")
    assert got.returncode != 0
    assert "point-id-only" in got.stderr


def test_hash_chain_and_artifact_drift_fail_closed(tmp_path):
    journal = initialize(tmp_path)
    assert record_content(tmp_path, journal).returncode == 0
    lines = (journal / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    artifact = journal / row["artifacts"][0]["stored_path"]
    artifact.write_text("tampered", encoding="utf-8")
    got = run("verify", "--journal-dir", str(journal))
    assert got.returncode != 0
    assert "artifact hash drift" in got.stderr


def test_payload_and_tombstone_schema_are_source_identity_only(tmp_path):
    journal = initialize(tmp_path)
    source = write(tmp_path / "source", "source")
    patch = write(tmp_path / "patch.jsonl", json.dumps({
        "identity": {"url": "https://example.test/a", "source_type": "web"},
        "set": {"country": "CH"}, "delete_keys": ["old"]}) + "\n")
    got = run("record", "--journal-dir", str(journal), "--operation-id", "p1",
              "--mode", "payload-only", "--source", str(source),
              "--source-git-sha", "c" * 40, "--artifact", f"payload_patch={patch}")
    assert got.returncode == 0, got.stderr
    tomb = write(tmp_path / "tomb.jsonl", json.dumps({
        "identity": {"url": "https://example.test/a", "source_type": "web"},
        "reason": "source withdrew consent"}) + "\n")
    got = run("record", "--journal-dir", str(journal), "--operation-id", "d1",
              "--mode", "tombstone", "--source", str(source),
              "--source-git-sha", "d" * 40, "--artifact", f"tombstones={tomb}")
    assert got.returncode == 0, got.stderr
    check = run("verify", "--journal-dir", str(journal))
    assert check.returncode == 0, check.stderr


def test_content_replay_of_a_card_row_keeps_the_page_rows_at_the_same_url(tmp_path, monkeypatch):
    """`replay_content` groups a row with its own entity markers: a refreshed card replaces
    the old card and never marks the page rows at the same url as superseded."""
    import pytest
    pytest.importorskip("qdrant_client")
    from qdrant_client.http import models as qm
    from cytir.core.qdrant import qdrant_client
    from cytir.index import supersede as US
    from cytir.index import delta_journal as J
    rows = json.loads((ROOT / "fixtures"
                       / "supersede-entity-marker-rows.json").read_text())
    ent = next(r for r in rows if r["payload"].get("qid"))
    key = US.AC.canonical_url(ent["payload"]["url"])
    pages = [r for r in rows if not r["payload"].get("qid")
             and US.AC.canonical_url(r["payload"]["url"]) == key]
    new_card = "11111111-1111-5111-8111-111111111111"
    client = qdrant_client(":memory:")
    client.create_collection("c", vectors_config={"dense": qm.VectorParams(
        size=3, distance=qm.Distance.COSINE)})
    client.upsert("c", points=[qm.PointStruct(id=r["id"], vector={"dense": [0.1, 0.2, 0.3]},
                                              payload=r["payload"]) for r in rows])
    monkeypatch.setattr(US.Superseder, "ensure_url_index", lambda self, create=False: None)
    artifact = tmp_path / "content.jsonl"
    artifact.write_text(json.dumps({"id": new_card, "vector": {"dense": [0.1, 0.2, 0.3]},
                                    "payload": {**ent["payload"], "title": "refreshed"}}) + "\n")
    out = J.replay_content(client, "c", artifact, batch=10, dry_run=False)
    left = {p.id for p in client.scroll("c", limit=100, with_payload=False)[0]}
    assert out["rows"] == 1
    assert ent["id"] not in left and new_card in left        # the old card is replaced
    assert {r["id"] for r in pages} <= left                  # the page rows stay
