#!/usr/bin/env python3
"""The `rollback-restore` null-arm procedure, tested on REAL recorded ids.

The ids under test are not invented: they are produced by running the REAL
recorder in `promote_sealed_pack.main()` over
`tests/fixtures/promote/task629_news_pack/` -- three actual news rows with their
sealed 1024-d fp16 vectors -- so the file this procedure reads is the file the
promote step actually writes, in the format it actually writes it, minted
through the same `remint-b1` identity path B01 will use.

(The news pack rather than the sibling technical-manufacturer one, for a reason
worth recording: the technical fixture's rows carry NO `license`, which
`aw_b1_projection` refuses, and `test_promote_sealed_pack` injects one to
exercise the licence path. Using it here would have meant either injecting a
licence -- which makes the "real file" claim thinner -- or passing
`--license-version`, which is a version and not a licence. The news pack carries
its own.)

That matters more than it sounds. The whole safety property of the rollback is
"delete exactly what was written", and it is checked by comparing two lists. A
test that hand-wrote the id file would be checking that two hand-written lists
match each other.

Run: python3 -m pytest tests/test_rollback_restore_null_arm.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys
from unittest import mock

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

from cytir.gates import content_gate_null_arm as NULL  # noqa: E402
from cytir.gates import rollback_restore_null_arm as R  # noqa: E402

# ---------------------------------------------------------------------------
# the real recorded-ids file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recorded_ids(tmp_path_factory) -> pathlib.Path:
    """The REAL recorder, over the REAL fixture pack, with only qdrant stubbed.

    `promote_sealed_pack.main()` is called in-process rather than shelled out,
    because the four things `plan()` needs from outside -- the target's declared
    dense width, which ids the collection already holds, which URLS it already
    serves, and the persisted-input store -- are network and filesystem state a
    unit test must not depend on. Everything between them is the production
    path: the same vector read, the same identity minting, the same
    `build_points`, and the same `--record-written-ids` block that B01 will run.

    `already_served_urls` joined that list in task 629 (the guard that refuses a
    net-new pack landing on pages the target already serves). It is stubbed to
    `[]` here for the same reason the other three are: this fixture is about the
    RECORDER, and a url census against `http://twin.invalid` is not a thing it
    is testing.

    Stubbing the write itself is what makes this a dry run; `--apply` is not
    passed, so `upsert` is never reached.
    """
    from cytir.index import promote_sealed_pack as P

    import test_promote_sealed_pack as TP        # the fixture's own constants
    out = tmp_path_factory.mktemp("rollback")
    ids_path = out / "written-ids.json"
    with mock.patch.object(P, "target_width", return_value=256), \
            mock.patch.object(P, "already_present", return_value=[]), \
            mock.patch.object(P, "already_served_urls", return_value=[]), \
            mock.patch.object(P.PI, "assert_pages_persisted",
                              return_value={"stubbed": True}), \
            mock.patch.object(P.LAYOUT, "fingerprint",
                              return_value={"points_count": 0}):
        P.main([
            "--pack-chunks", TP.NEWS_CHUNKS, "--vectors", TP.NEWS_VECTORS,
            "--qdrant", "http://twin.invalid", "--collection", "aw_e8",
            "--mrl", "256", "--identity-check", "remint-b1",
            "--identity-reason", "test fixture",
            "--family-payload", json.dumps(TP.NEWS_PAYLOAD),
            "--published-from", TP.NEWS_DATE["published_from"],
            "--published-status-from", TP.NEWS_DATE["published_status_from"],
            "--published-status-requires",
            TP.NEWS_DATE["published_status_requires"],
            "--require-published",
            *[arg for field in TP.NEWS_TITLE_FROM
              for arg in ("--title-from", field)],
            "--receipt", str(out / "receipt.json"),
            "--record-written-ids", str(ids_path),
        ])
    assert ids_path.is_file(), "the recorder wrote nothing"
    return ids_path


def test_the_recorder_writes_a_plain_list_of_string_ids(recorded_ids):
    """The format the loader depends on, asserted against the real producer."""
    raw = json.loads(recorded_ids.read_text())
    assert isinstance(raw, list), type(raw).__name__
    assert raw, "the fixture pack produced no ids"
    assert all(isinstance(value, str) for value in raw), raw[:3]
    assert len(set(raw)) == len(raw), "the recorder emitted duplicates"


def test_the_loader_round_trips_the_real_file(recorded_ids):
    ids = R.load_written_ids(recorded_ids)
    assert ids == [str(value) for value in json.loads(recorded_ids.read_text())]
    assert len(ids) == len(set(ids))


def test_the_recorded_ids_are_the_ids_the_receipt_counted(recorded_ids):
    """The receipt's own count and the id file must agree.

    They are written by the same call and could still disagree -- the count comes
    from the planned rows and the file from the built points -- so a drift
    between the two is exactly the kind of thing that makes a rollback delete a
    different number than it reports.
    """
    receipt = json.loads((recorded_ids.parent / "receipt.json").read_text())
    ids = R.load_written_ids(recorded_ids)
    assert receipt["written_ids_count"] == len(ids)
    assert receipt["written_ids_path"] == str(recorded_ids)


# ---------------------------------------------------------------------------
# the refusals, each one a way the rollback could delete the wrong set
# ---------------------------------------------------------------------------
def test_a_receipt_passed_instead_of_the_id_file_refuses(recorded_ids):
    """Iterating a dict yields its KEYS, which for a receipt are field names."""
    with pytest.raises(R.NullArmProcedureError, match="JSON LIST"):
        R.load_written_ids(recorded_ids.parent / "receipt.json")


def test_an_empty_id_file_refuses(tmp_path):
    """A rollback of nothing reads the state you already had, and looks fine."""
    empty = tmp_path / "none.json"
    empty.write_text("[]\n")
    with pytest.raises(R.NullArmProcedureError, match="looks like it worked"):
        R.load_written_ids(empty)


def test_duplicated_ids_refuse(recorded_ids, tmp_path):
    """A duplicate makes the delete's own count overstate what it removes."""
    ids = json.loads(recorded_ids.read_text())
    doubled = tmp_path / "dupes.json"
    doubled.write_text(json.dumps(ids + ids[:1]))
    with pytest.raises(R.NullArmProcedureError, match="duplicate"):
        R.load_written_ids(doubled)


def test_an_absent_id_refuses_before_anything_is_deleted(monkeypatch,
                                                         recorded_ids):
    ids = R.load_written_ids(recorded_ids)
    monkeypatch.setattr(R, "already_present",
                        lambda url, collection, wanted: list(wanted)[:-1])
    with pytest.raises(R.NullArmProcedureError, match="no-op dressed as"):
        R.assert_all_present("http://twin.invalid", "aw_e8", ids)


def test_all_present_passes_when_every_id_is_there(monkeypatch, recorded_ids):
    ids = R.load_written_ids(recorded_ids)
    monkeypatch.setattr(R, "already_present",
                        lambda url, collection, wanted: list(wanted))
    proof = R.assert_all_present("http://twin.invalid", "aw_e8", ids)
    assert proof == {"checked": len(ids), "present": len(ids)}


def test_the_delete_uses_an_id_list_and_never_a_filter(monkeypatch,
                                                       recorded_ids):
    """A filter is a predicate over payloads; a later rule epoch widens it."""
    ids = R.load_written_ids(recorded_ids)
    calls = []
    monkeypatch.setattr(R, "_api", lambda url, path, body=None, method=None:
                        calls.append({"path": path, "body": body}) or {})
    removed = R.delete_ids("http://twin.invalid", "aw_e8", ids, batch=2)
    assert removed == len(ids)
    assert calls, "nothing was sent"
    for call in calls:
        assert "points/delete" in call["path"]
        assert set(call["body"]) == {"points"}, call["body"]
        assert "filter" not in call["body"]
    sent = [pid for call in calls for pid in call["body"]["points"]]
    assert sent == ids, "the delete must send exactly the recorded ids, in order"


# ---------------------------------------------------------------------------
# the two readings that confirm the rollback
# ---------------------------------------------------------------------------
def _fingerprint(points):
    return {"points_count": points, "n_per_vector": points, "segment_sizes": []}


def test_a_survivor_refuses_even_when_the_count_is_right(monkeypatch,
                                                         recorded_ids):
    ids = R.load_written_ids(recorded_ids)
    monkeypatch.setattr(R, "already_present",
                        lambda url, collection, wanted: [ids[0]])
    monkeypatch.setattr(R.LAYOUT, "fingerprint",
                        lambda url, collection: _fingerprint(100 - len(ids)))
    with pytest.raises(R.NullArmProcedureError, match="survived"):
        R.assert_rolled_back("http://twin.invalid", "aw_e8", ids, 100)


def test_an_unchanged_count_refuses_even_when_every_id_is_gone(monkeypatch,
                                                              recorded_ids):
    """Absent ids with no count change means something else wrote meanwhile."""
    ids = R.load_written_ids(recorded_ids)
    monkeypatch.setattr(R, "already_present", lambda url, collection, wanted: [])
    monkeypatch.setattr(R.LAYOUT, "fingerprint",
                        lambda url, collection: _fingerprint(100))
    with pytest.raises(R.NullArmProcedureError, match="points_count is"):
        R.assert_rolled_back("http://twin.invalid", "aw_e8", ids, 100)


def test_both_readings_agreeing_is_what_passes(monkeypatch, recorded_ids):
    ids = R.load_written_ids(recorded_ids)
    monkeypatch.setattr(R, "already_present", lambda url, collection, wanted: [])
    monkeypatch.setattr(R.LAYOUT, "fingerprint",
                        lambda url, collection: _fingerprint(100 - len(ids)))
    proof = R.assert_rolled_back("http://twin.invalid", "aw_e8", ids, 100)
    assert proof["deleted"] == len(ids)
    assert proof["survivors"] == 0
    assert proof["points_after"] == 100 - len(ids)


# ---------------------------------------------------------------------------
# the basis receipt
# ---------------------------------------------------------------------------
def _convergence(queries=1_400):
    return {"schema": "aw.gate-convergence.v1", "verdict": "CONVERGED",
            "queries_compared": queries, "disagreements": [],
            "receipt_sha256": "c" * 64}


def _basis(**overrides):
    kwargs = dict(collection="aw_e8", points_count=33_737_287,
                  membership_sha256="a" * 64, convergence=_convergence(),
                  baseline_identity="b01:baseline",
                  baseline_arm_sha256="b" * 64,
                  null_arm_path="/tmp/null.json",
                  evidence=[{"path": "/tmp/ids.json", "sha256": "d" * 64,
                             "what": "the ids deleted"}])
    kwargs.update(overrides)
    return R.build_basis(**kwargs)


def test_the_basis_validates_through_the_ratified_reader():
    """A receipt only this module can read is not a receipt."""
    doc = _basis()
    assert doc["schema"] == NULL.BASIS_SCHEMA
    assert doc["protocol"] == "rollback-restore"
    NULL.validate_basis(doc, collection="aw_e8", points_count=33_737_287,
                        baseline_identity="b01:baseline",
                        baseline_arm_sha256="b" * 64)


def test_the_basis_refuses_a_non_converged_null_reading():
    with pytest.raises(NULL.NullArmError, match="CONVERGED"):
        _basis(convergence={**_convergence(), "verdict": "NOT-CONVERGED"})


def test_the_basis_must_name_this_runs_baseline_arm():
    """am.6 rule 3: another run's null arm is not this run's control."""
    doc = _basis()
    with pytest.raises(NULL.NullArmError, match="another run"):
        NULL.validate_basis(doc, collection="aw_e8", points_count=33_737_287,
                            baseline_identity="some-other-run:baseline",
                            baseline_arm_sha256="b" * 64)


def test_the_protocol_is_the_one_am6_admits_and_not_the_refused_pair():
    """The cheap alternative is named and refused, and must stay refused."""
    assert R.PROTOCOL in NULL.BASIS_PROTOCOLS
    assert "two-readings-converged" in NULL.REFUSED_PROTOCOLS
    with pytest.raises(NULL.NullArmError, match="refused"):
        doc = _basis()
        doc["protocol"] = "two-readings-converged"
        NULL.validate_basis(doc, collection="aw_e8", points_count=33_737_287,
                            baseline_identity="b01:baseline",
                            baseline_arm_sha256="b" * 64)


def test_an_unconverged_arm_refuses_at_the_procedure_boundary_too(monkeypatch):
    """The refusal exists on both sides: assembling, and cutting."""
    monkeypatch.setattr(R.CONV, "converged_arms",
                        lambda cut, **kw: {"verdict": "NOT-CONVERGED"})
    with pytest.raises(R.NullArmProcedureError, match="still moving"):
        R.converged_null_arm(lambda index: pathlib.Path("/tmp/x.json"))
