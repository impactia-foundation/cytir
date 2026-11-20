"""cytir.index.e10_ship -- what the flip reads of an aw_e10 build (kept after the builder was deleted,
task 1312: `deploy/scripts/flip-relocate.sh` step 2b still runs `python -m cytir.index.e10_ship`)."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pytest

from cytir.index import e10_ship as _SH

SCHEMA_MANIFEST = _SH.SCHEMA_MANIFEST
SCHEMA_DATA_CHECKS = _SH.SCHEMA_DATA_CHECKS


def _sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _write_json(p, doc) -> None:
    pathlib.Path(p).write_text(json.dumps(doc))


# ------------------------------------------- review 870 S4: what the flip reads (cytir.index.e10_ship)
def _ship_files(tmp_path, *, target="aw_e10", count=10, rnd=2, verdict_round=2, passed=True, shippable=True,
                dirs=None):
    """A plan directory as the builder leaves it: plan.json, the manifest, the verdict and one
    `catchup-<target>-rNN/` per round (`dirs` overrides the round directories on disk)."""
    (tmp_path / "plan.json").write_text('{"schema": "x"}')
    plan_sha = _sha256_file(tmp_path / "plan.json")
    for d in tmp_path.glob(f"catchup-{target}-r*"):
        shutil.rmtree(d)
    for k in (range(1, rnd + 1) if dirs is None else dirs):
        (tmp_path / f"catchup-{target}-r{k:02d}").mkdir()
    man = {"schema": SCHEMA_MANIFEST, "target": target, "points_exact": count, "plan_sha256": plan_sha,
           "shippable": shippable, "shippable_blockers": [] if shippable else ["x"],
           "catchup_rounds": [{"round": k + 1} for k in range(rnd)],
           "data_checks": {"pass": passed, "file": f"data-checks-{target}.json"}}
    doc = {"schema": SCHEMA_DATA_CHECKS, "pass": passed, "checks": {"german_parts": {"pass": passed}},
           "binding": {"target": target, "plan_sha256": plan_sha, "round": verdict_round, "points_exact": count}}
    _write_json(tmp_path / f"manifest-{target}.json", man)
    _write_json(tmp_path / f"data-checks-{target}.json", doc)
    return tmp_path / f"manifest-{target}.json"


def test_the_flip_reads_a_shippable_build_with_checks_bound_to_its_last_round(tmp_path):
    from cytir.index import e10_ship as SH
    assert SH.assert_shippable(_ship_files(tmp_path), target="aw_e10", expect_count=10)["round"] == 2
    for kw, needle in (({"verdict_round": 1}, "another state"), ({"passed": False}, "did not PASS"),
                       ({"shippable": False}, "not shippable"), ({"count": 11}, "the flip expects")):
        m = _ship_files(tmp_path, **kw)
        with pytest.raises(SH.ShipRefused, match=needle):
            SH.assert_shippable(m, target="aw_e10", expect_count=10)
    with pytest.raises(SH.ShipRefused, match="the flip ships"):
        SH.assert_shippable(_ship_files(tmp_path), target="aw_e11", expect_count=10)


def test_the_flip_refuses_a_round_on_disk_beyond_the_manifests_rounds(tmp_path):
    """Review 870 verification M4: Δ2 applied (or only started) on disk after `finalize` wrote the
    manifest: the manifest lists 2 rounds, the round-2 verdict and the count are unchanged, a third
    round directory exists -> refused. So is a manifest without its plan directory or its rounds."""
    import shutil as _sh
    from cytir.index import e10_ship as SH
    m = _ship_files(tmp_path, rnd=2, verdict_round=2, dirs=[1, 2, 3])
    (tmp_path / "catchup-aw_e10-r03" / "catchup.json").write_text('{"applied": true}')
    with pytest.raises(SH.ShipRefused, match=r"on disk beyond it: \['catchup-aw_e10-r03'\]"):
        SH.assert_shippable(m, target="aw_e10", expect_count=10)
    _sh.rmtree(tmp_path / "catchup-aw_e10-r03")
    (tmp_path / "catchup-aw_e10-r03").mkdir()                    # a round that is not applied: the same
    with pytest.raises(SH.ShipRefused, match="beyond it"):
        SH.assert_shippable(m, target="aw_e10", expect_count=10)
    m = _ship_files(tmp_path, rnd=2, verdict_round=2, dirs=[1])  # the rounds were not copied with it
    with pytest.raises(SH.ShipRefused, match=r"listed but absent: \['catchup-aw_e10-r02'\]"):
        SH.assert_shippable(m, target="aw_e10", expect_count=10)
    m = _ship_files(tmp_path)
    (tmp_path / "plan.json").write_text('{"schema": "another plan"}')
    with pytest.raises(SH.ShipRefused, match="plan_sha256"):
        SH.assert_shippable(m, target="aw_e10", expect_count=10)
    (tmp_path / "plan.json").unlink()
    with pytest.raises(SH.ShipRefused, match="inside its plan directory"):
        SH.assert_shippable(m, target="aw_e10", expect_count=10)
    copied = tmp_path / "elsewhere"
    copied.mkdir()
    _sh.copy(m, copied / "manifest-aw_e10.json")
    with pytest.raises(SH.ShipRefused, match="inside its plan directory"):
        SH.assert_shippable(copied / "manifest-aw_e10.json", target="aw_e10", expect_count=10)
    assert SH.main(["--manifest", str(tmp_path / "none.json"), "--target", "aw_e10", "--expect-count", "1"]) == 2
