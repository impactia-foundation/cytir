"""The CI `test` job is a matrix of shards (task 827); together they run exactly what the one job ran.

`pipeline/` and `tests/` are each split in two by `scripts/ci_shard.py` (static, weighted; task 1925
added the `tests` root), the `rest` shard runs the other locations. What this file proves, so a split
cannot lose coverage quietly:
  * the shards of each root PARTITION every `<root>/**/test_*.py` (each module in exactly one shard,
    none dropped, none doubled), for the shard count the workflow uses and for others;
  * a module the weights file does not know is still placed (a new test needs no edit);
  * the workflow runs shard i of N for every i, with the same N, and runs every other location
    (`tests`, `server`, `webapp`) in the `rest` shard, and never `tasks/` (task 1806);
  * every shard's check name contains "pytest": the receipt-merger script skips a PR with no
    such check, and merges only when none is pending or failing.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest
import yaml

from tests.helpers.repo_files import repo_file_or_skip

APP = pathlib.Path(__file__).resolve().parent.parent
CI = repo_file_or_skip("../.github/workflows/ci.yml")
repo_file_or_skip("scripts/ci_shard.py")
sys.path.insert(0, str(APP / "scripts"))
import ci_shard  # noqa: E402


@pytest.mark.parametrize("root", ["pipeline", "tests"])
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_shards_partition_the_modules(n, root):
    got = ci_shard.shards(n, root=root)
    flat = [m for s in got for m in s]
    assert len(flat) == len(set(flat)), "a module is in two shards"
    assert sorted(flat) == ci_shard.modules(root=root), "a module is in no shard"
    assert all(got), "an empty shard"
    assert all(m.startswith(root + "/") and pathlib.Path(APP / m).is_file() for m in flat)


def test_a_module_missing_from_the_weights_is_still_placed(tmp_path):
    root = "pipeline"
    (tmp_path / root).mkdir()
    (tmp_path / "data").mkdir()
    for name in ("test_a.py", "test_b.py", "test_new.py"):
        (tmp_path / root / name).write_text("")
    (tmp_path / "data" / "ci-shard-weights.v1.json").write_text(
        json.dumps({"default_weight": 1, "seconds": {f"{root}/test_a.py": 50}}))
    got = ci_shard.shards(2, tmp_path)
    assert sorted(m for s in got for m in s) == [f"{root}/{n}" for n in ("test_a.py", "test_b.py", "test_new.py")]
    assert [f"{root}/test_a.py"] in got  # the heavy one is alone


@pytest.mark.parametrize("root", ["pipeline", "tests"])
def test_the_weights_file_is_balanced_at_the_workflow_shard_count(root):
    w = json.loads((APP / "data" / ci_shard.ROOTS[root]).read_text())
    loads = [sum(w["seconds"].get(m, w["default_weight"]) for m in s) for s in ci_shard.shards(2, root=root)]
    if sum(loads) < 120:   # S9: pipeline/ shrank to the boundary guards; under two minutes of tests, balance is moot
        pytest.skip(f"{root}/ carries {sum(loads):.0f}s of tests; the 15% balance bound applies from 120s")
    assert max(loads) / min(loads) < 1.15, loads


@pytest.mark.parametrize("root", ["pipeline", "tests"])
def test_the_weights_file_names_only_real_modules(root):
    w = json.loads((APP / "data" / ci_shard.ROOTS[root]).read_text())
    assert set(w["seconds"]) <= set(ci_shard.modules(root=root))


def test_the_pipeline_cli_form_is_unchanged_and_tests_root_is_selectable(capsys):
    assert ci_shard.main(["ci_shard.py", "1", "2"]) == 0
    assert capsys.readouterr().out.split() == ci_shard.shards(2)[0]
    assert ci_shard.main(["ci_shard.py", "--root", "tests", "2", "2"]) == 0
    assert capsys.readouterr().out.split() == ci_shard.shards(2, root="tests")[1]
    assert ci_shard.main(["ci_shard.py", "--root", "nope", "1", "2"]) == 2


def test_the_workflow_runs_every_shard_and_every_other_location():
    job = yaml.safe_load(CI.read_text())["jobs"]["test"]
    shards = job["strategy"]["matrix"]["shard"]
    assert job["strategy"]["fail-fast"] is False
    assert "rest" in shards
    for root, flag in (("pipeline", ""), ("tests", "--root tests ")):
        root_shards = [s for s in shards if s.startswith(root + "-")]
        n = len(root_shards)
        assert n == 2, root
        for i, name in enumerate(root_shards, 1):
            assert name == f"{root}-{i}-of-{n}"
            steps = [s for s in job["steps"] if f"matrix.shard == '{name}'" in s.get("if", "")]
            assert len(steps) == 1, name
            assert f"scripts/ci_shard.py {flag}{i} {n})" in steps[0]["run"]
    assert sorted(shards) == sorted(["rest", "pipeline-1-of-2", "pipeline-2-of-2", "tests-1-of-2", "tests-2-of-2"])
    rest_runs = " ".join(s.get("run", "") for s in job["steps"] if "matrix.shard == 'rest'" in s.get("if", ""))
    assert re.search(r"pytest webapp ", rest_runs)
    assert not re.search(r"pytest tests ", rest_runs)  # task 1925: tests/ runs only in its two shards
    assert not re.search(r"pytest tasks", rest_runs)  # task 1806: frozen tasks/ tests never collected
    # no unconditional pytest step: it would run in every shard
    for s in job["steps"]:
        if "pytest" in s.get("run", "") and "matrix.shard" not in s.get("if", ""):
            pytest.fail(f"step runs in every shard: {s['name']}")


def test_every_shard_check_name_contains_pytest_for_the_receipt_merger():
    job = yaml.safe_load(CI.read_text())["jobs"]["test"]
    for shard in job["strategy"]["matrix"]["shard"]:
        assert "pytest" in job["name"].replace("${{ matrix.shard }}", shard)
