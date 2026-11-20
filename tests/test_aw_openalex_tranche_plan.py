import pytest

from cytir.extract import openalex_academic as subject


def test_selects_smallest_whole_part_at_or_above_target_and_excludes_pilot():
    rows = [
        {"relative_path": "pilot.parquet", "retained_legacy_records": 10001},
        {"relative_path": "b.parquet", "retained_legacy_records": 10020},
        {"relative_path": "a.parquet", "retained_legacy_records": 10020},
        {"relative_path": "thin.parquet", "retained_legacy_records": 9999},
    ]
    assert subject.choose_part(
        rows, minimum_records=10000,
        excluded_paths={"pilot.parquet"}) == rows[2]


def test_rejects_duplicate_frontier_paths_and_missing_minimum():
    duplicate = [
        {"relative_path": "a.parquet", "retained_legacy_records": 10000},
        {"relative_path": "a.parquet", "retained_legacy_records": 10001},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        subject.choose_part(duplicate, minimum_records=10000,
                            excluded_paths=set())
    with pytest.raises(ValueError, match="no disjoint"):
        subject.choose_part(duplicate[:1], minimum_records=10001,
                            excluded_paths=set())


def test_multipart_selects_fewest_largest_whole_parts_stably_and_excludes():
    rows = [
        {"relative_path": "excluded.parquet", "retained_legacy_records": 99},
        {"relative_path": "c.parquet", "retained_legacy_records": 30},
        {"relative_path": "b.parquet", "retained_legacy_records": 35},
        {"relative_path": "a.parquet", "retained_legacy_records": 35},
        {"relative_path": "d.parquet", "retained_legacy_records": 20},
    ]
    selected = subject.choose_parts(
        list(reversed(rows)), minimum_records=100,
        excluded_paths={"excluded.parquet"})
    assert [row["relative_path"] for row in selected] == [
        "a.parquet", "b.parquet", "c.parquet"]
    assert sum(row["retained_legacy_records"] for row in selected) == 100


def test_multipart_rejects_duplicates_and_insufficient_measured_rows():
    duplicate = [
        {"relative_path": "a.parquet", "retained_legacy_records": 40},
        {"relative_path": "a.parquet", "retained_legacy_records": 60},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        subject.choose_parts(duplicate, minimum_records=50,
                             excluded_paths=set())
    with pytest.raises(ValueError, match="no disjoint"):
        subject.choose_parts(duplicate[:1], minimum_records=41,
                             excluded_paths=set())


def test_partition_parts_is_contiguous_deterministic_and_conservative():
    rows = [
        {"relative_path": f"part-{index}.parquet",
         "retained_legacy_records": count,
         "sha256": str(index) * 64}
        for index, count in enumerate((11, 17, 23, 29, 31, 37, 41))
    ]
    first = subject.partition_parts(rows, partition_count=4)
    second = subject.partition_parts(list(rows), partition_count=4)
    assert first == second
    assert [path for partition in first
            for path in partition["relative_paths"]] == [
                row["relative_path"] for row in rows]
    assert [partition["selected_part_start"] for partition in first] == [
        0, first[0]["selected_part_end_exclusive"],
        first[1]["selected_part_end_exclusive"],
        first[2]["selected_part_end_exclusive"]]
    assert first[-1]["selected_part_end_exclusive"] == len(rows)
    assert sum(partition["selected_records"] for partition in first) == sum(
        row["retained_legacy_records"] for row in rows)
    assert all(partition["selected_part_count"] > 0 for partition in first)


def test_partition_parts_rejects_impossible_or_invalid_partition_count():
    rows = [{"relative_path": "a", "retained_legacy_records": 1}]
    with pytest.raises(ValueError, match="partition count"):
        subject.partition_parts(rows, partition_count=0)
    with pytest.raises(ValueError, match="partition count"):
        subject.partition_parts(rows, partition_count=2)


def test_plan_tranche_output_is_read_back_by_the_family_consumer(tmp_path):
    import hashlib
    import json

    root = tmp_path / "src"
    frontier = []
    for i in range(2216):       # the planner pins the terminal run at 2216 selected parts
        n = 30 if i < 4 else 1
        frontier.append({"relative_path": f"updated_date=2026-08-01/part_{i:04d}.parquet",
                         "retained_legacy_records": n, "upstream_records": n * 2,
                         "expected_bytes": 5})
    for row in frontier[:4]:
        rel, body = row["relative_path"], root / "original" / "parquet" / row["relative_path"]
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_bytes(b"12345")
        key = hashlib.sha256(rel.encode()).hexdigest()
        receipt = root / "receipts" / key[:2] / f"{key}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "schema": "aw.openalex-parquet-part-receipt.v1", "status": "complete",
            "relative_path": rel, "retained_legacy_records": 30,
            "body": {"bytes": 5, "sha256": hashlib.sha256(b"12345").hexdigest()}}))
    (root / "frontier.jsonl").write_text("\n".join(json.dumps(r) for r in frontier) + "\n")
    run = root / "RUN.json"
    run.write_text(json.dumps({"schema": "aw.openalex-snapshot-acquisition-run.v1",
                               "status": "complete", "counts": {"selected_parts": 2216}}))
    sha = hashlib.sha256(run.read_bytes()).hexdigest()
    out = tmp_path / "plan.json"
    result = subject.plan_tranche(
        source_root=root, source_run=run, output=out, artifact_root=tmp_path,
        expected_source_run_sha256=sha, minimum_records=100, excluded_paths=set(),
        multi_part=True, partition_count=2)
    assert result["version"] == subject.PARTITIONED_VERSION
    assert [len(p["relative_paths"]) for p in result["partitions"]] == [2, 2]
    paths, records = subject.planned_relative_paths(
        json.loads(out.read_text()), expected_source_run_sha256=sha, target_records=100)
    assert paths == [r["relative_path"] for r in frontier[:4]] and records == 120
    with pytest.raises(ValueError, match="fresh"):
        subject.plan_tranche(
            source_root=root, source_run=run, output=out, artifact_root=tmp_path,
            expected_source_run_sha256=sha, minimum_records=100, excluded_paths=set())


# --- the plan's bytes are pinned (review 1274 of #1071) ---------------------------------------
# A fixed fake frontier of 2216 parts (the count the plan demands); three parts have real bodies.
# The plan carries no clock and is written with indent=2, sort_keys=True, so its bytes are a pure
# function of the inputs. A DRIFT in a pinned digest means the plan FORMAT changed: check the
# chain consumers and the partition finaliser before updating the constant.
_PLAN_SHA256 = {
    "single": (
        "894a0944c8a16852a9f6cf7f7115517bfcd96338c27534c21d6bcdafc20f9fd8"),
    "multipart": (
        "0d5ec6a04decb8f633c5445dc0072d1101e1fc4e63fd98f5ae45c423f7bb759d"),
    "partitioned": (
        "1a3f805d18c07160cf80b218d59c775b30ec672b04dd4c73fdee65908d71813c"),
}


def _fixture(tmp_path):
    import hashlib
    import json
    root = tmp_path / "art"
    src = root / "src"
    counts = {"p0.parquet": 600, "p1.parquet": 500, "p2.parquet": 400}
    rows = [{"relative_path": name, "retained_legacy_records": n,
             "upstream_records": n + 7, "expected_bytes": len(name) + 10}
            for name, n in counts.items()]
    rows += [{"relative_path": f"filler-{i:04d}.parquet", "retained_legacy_records": 1,
              "upstream_records": 1, "expected_bytes": 1} for i in range(2216 - 3)]
    (src).mkdir(parents=True)
    (src / "frontier.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    for r in rows[:3]:
        rel = r["relative_path"]
        body = src / "original" / "parquet" / rel
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_bytes(b"x" * r["expected_bytes"])
        key = hashlib.sha256(rel.encode()).hexdigest()
        receipt = src / "receipts" / key[:2] / f"{key}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "schema": "aw.openalex-parquet-part-receipt.v1", "status": "complete",
            "relative_path": rel, "retained_legacy_records": r["retained_legacy_records"],
            "body": {"bytes": r["expected_bytes"],
                     "sha256": hashlib.sha256(body.read_bytes()).hexdigest()}}, sort_keys=True))
    run = root / "RUN.json"
    run.write_text(json.dumps({"schema": "aw.openalex-snapshot-acquisition-run.v1",
                               "status": "complete", "counts": {"selected_parts": 2216}}))
    return root, src, run, hashlib.sha256(run.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,kwargs,minimum", [
    ("single", {}, 600),
    ("multipart", {"multi_part": True}, 1000),
    ("partitioned", {"multi_part": True, "partition_count": 2}, 1500),
])
def test_plan_bytes_are_pinned(tmp_path, name, kwargs, minimum):
    import hashlib
    root, src, run, run_sha = _fixture(tmp_path)
    out = root / "plan.json"
    subject.plan_tranche(source_root=src, source_run=run, output=out, artifact_root=root,
                         expected_source_run_sha256=run_sha, minimum_records=minimum,
                         excluded_paths=set(), **kwargs)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == _PLAN_SHA256[name]
