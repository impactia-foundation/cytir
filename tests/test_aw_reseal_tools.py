"""Task 2535: the two re-seal tools, run only on COPIES of the sealed data in tmp_path."""
import hashlib
import json
import pathlib
import shutil

import pytest

from cytir.chunk import registry as R
from cytir.chunk import registry_repoint as RP
from cytir.core.paths import data_path
from cytir.sources import contract as C
from cytir.sources import fedlex_law_meta as F

POLICY = "chunk-policy-legacy-180w-o30.v1.json"
OLD = "0491ea77909c10956186f9821b63d5304ef0407b93b61a2ee2452f75d5dbfd56"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # the registry's --verify reads every policy and selector artifact, so copy them all
    src = data_path("chunk-policy-registry.v1.json").parent
    names = {p.name for pat in ("chunk-policy-*.json", "chunk-reference-selector*.json")
             for p in src.glob(pat)}
    for n in names | {"source-family-contracts.v1.json"}:
        shutil.copy(src / n, d / n)
    if _sha(d / POLICY) != OLD:
        pytest.skip("the legacy policy was already re-sealed; the fixture's OLD sha moved")
    for entry in ROOT.iterdir():   # a real checkout view: cytir/ etc. resolve, so code_home checks run for real
        if entry.name not in ("data", ".git"):
            (tmp_path / entry.name).symlink_to(entry)
    return tmp_path


def _args(repo, *extra, old=OLD, new=None):
    return ["--old-sha", old, "--new-sha", new or "0" * 64, "--reason", "test re-seal",
            "--registry", str(repo / "data/chunk-policy-registry.v1.json"),
            "--contracts", str(repo / "data/source-family-contracts.v1.json"),
            "--repo", str(repo), *extra]


def _snapshot(repo):
    return {p.name: p.read_bytes() for p in (repo / "data").iterdir()}


def _edit_policy(repo):
    p = repo / "data" / POLICY
    doc = json.loads(p.read_text())
    doc["producer"]["module"] = "cytir.chunk.legacy"
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return _sha(p)


def test_real_run_rewrites_registry_keys_and_appends_the_amendment_row(repo):
    new = _edit_policy(repo)
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, new=new)) == 0
    reg = (repo / "data/chunk-policy-registry.v1.json").read_text()
    assert OLD not in reg
    assert reg.count(new) == before["chunk-policy-registry.v1.json"].decode().count(OLD) == 3
    assert R.verify_against_files(json.loads(reg), repo) == []
    ctext = (repo / "data/source-family-contracts.v1.json").read_text()
    assert OLD not in ctext
    cdoc = json.loads(ctext)
    pinned = [f for f in cdoc["families"] if f.get("chunk_policy", {}).get("sha256") == new]
    assert len(pinned) == 1                                   # the new sha sits under chunk_policy.sha256
    row = cdoc["amendments"][-1]
    assert row["keys"] == ["chunk_policy.sha256"]
    assert row["old_file_sha256"] == hashlib.sha256(before["source-family-contracts.v1.json"]).hexdigest()
    rec, comp = C.pre_amendment_sha256(ctext)
    assert rec == comp


def test_dry_run_changes_nothing(repo, capsys):
    new = _edit_policy(repo)
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, "--dry-run", new=new)) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and new in out
    assert _snapshot(repo) == before


def test_refuses_new_sha_that_is_not_the_policy_file(repo, capsys):
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, new="a" * 64)) == 1
    assert "is not the sha256 of" in capsys.readouterr().err
    assert _snapshot(repo) == before


def test_refuses_old_sha_not_in_registry(repo, capsys):
    new = _edit_policy(repo)
    assert RP.repoint_main(_args(repo, old="b" * 64, new=new)) == 1
    assert "is not a key" in capsys.readouterr().err


def test_refuses_malformed_sha_and_equal_shas(repo, capsys):
    assert RP.repoint_main(_args(repo, new="xyz")) == 1
    assert RP.repoint_main(_args(repo, new=OLD)) == 1
    assert capsys.readouterr().err.count("REFUSED") == 2


def test_refuses_when_registry_check_would_fail_afterwards(repo, capsys):
    new = _edit_policy(repo)
    # an unregistered policy file on disk: --verify fails after the rewrite as well
    (repo / "data" / "chunk-policy-stray.v1.json").write_text('{"epoch": 9, "policy_id": "stray"}')
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, new=new)) == 1
    assert "--verify would fail" in capsys.readouterr().err
    assert _snapshot(repo) == before


def test_refuses_when_contract_check_would_fail(repo, capsys, monkeypatch):
    new = _edit_policy(repo)
    monkeypatch.setattr(RP, "_contract_problems",
                        lambda C_, cdoc, repo_, registry_text=None:
                        [] if registry_text is None else ["injected: contract breaks"])
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, new=new)) == 1
    assert "injected" in capsys.readouterr().err
    assert _snapshot(repo) == before


def test_refuses_without_reason(repo):
    new = _edit_policy(repo)
    args = _args(repo, new=new)
    args[args.index("--reason") + 1] = "  "
    assert RP.repoint_main(args) == 1


# ---- fedlex -----------------------------------------------------------------------------------

@pytest.fixture
def fed(tmp_path):
    names = tmp_path / F.NAMES_FILE
    shutil.copy(data_path(F.NAMES_FILE), names)
    art = tmp_path / "fedlex-law-meta.v1.json"
    art.write_text(json.dumps({"curated_names": {"path": F.TARGET_PATH, "sha256": _sha(names)}}))
    return names, art


def test_fedlex_dry_run_changes_nothing(fed, capsys):
    names, art = fed
    names.write_text(names.read_text() + " ")
    before = (art.read_bytes(), names.read_bytes())
    assert F.main(["reseal", "--dry-run", "--write-artifact", "--names", str(names),
                   "--artifact", str(art)]) == 0
    assert "artifact NOT written" in capsys.readouterr().out
    assert (art.read_bytes(), names.read_bytes()) == before


def test_fedlex_without_flag_prints_only(fed, capsys):
    names, art = fed
    names.write_text(names.read_text() + " ")
    before = art.read_bytes()
    assert F.main(["reseal", "--names", str(names), "--artifact", str(art)]) == 1   # CI-usable check
    out = capsys.readouterr().out
    assert _sha(names) in out and "DIFFERS" in out
    assert art.read_bytes() == before


def test_fedlex_matching_artifact_exits_zero(fed):
    names, art = fed
    assert F.main(["reseal", "--names", str(names), "--artifact", str(art)]) == 0


def test_fedlex_write_artifact_reseals(fed):
    names, art = fed
    names.write_text(names.read_text() + " ")
    assert F.main(["reseal", "--write-artifact", "--names", str(names), "--artifact", str(art)]) == 0
    assert F.sealed_entry(json.loads(art.read_text()))["sha256"] == _sha(names)


def test_fedlex_artifact_absent_prints_expected_sha(tmp_path, capsys):
    names = tmp_path / F.NAMES_FILE
    shutil.copy(data_path(F.NAMES_FILE), names)
    assert F.main(["reseal", "--names", str(names), "--artifact", str(tmp_path / "nope.json")]) == 0
    out = capsys.readouterr().out
    assert "absent" in out and _sha(names) in out


def test_fedlex_refusals(fed, tmp_path, capsys):
    names, art = fed
    assert F.main(["reseal", "--write-artifact", "--names", str(names),
                   "--artifact", str(tmp_path / "nope.json")]) == 1          # artifact absent
    assert F.main(["reseal", "--names", str(tmp_path / "missing.json"), "--artifact", str(art)]) == 1
    art.write_text(json.dumps({"x": 1}))                                       # no sealed entry
    assert F.main(["reseal", "--names", str(names), "--artifact", str(art)]) == 1
    assert capsys.readouterr().err.count("REFUSED") == 3


def test_both_verbs_are_reachable_from_the_cli_table(capsys):
    from cytir import cli
    assert cli._source_main(["fedlex-law-meta", "reseal", "--dry-run"]) == 0
    assert "computed sha256" in capsys.readouterr().out
    assert cli._chunk_main(["registry", "repoint-policy", "--old-sha", "b" * 64, "--new-sha", "c" * 64,
                            "--reason", "r", "--dry-run"]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_refuses_a_prose_occurrence_of_old_sha(repo, capsys):
    new = _edit_policy(repo)
    reg = repo / "data/chunk-policy-registry.v1.json"
    doc = json.loads(reg.read_text())
    doc["why"].append("see also " + OLD)
    reg.write_text(json.dumps(doc, indent=1) + "\n")
    before = _snapshot(repo)
    assert RP.repoint_main(_args(repo, "--dry-run", new=new)) == 1
    assert "prose" in capsys.readouterr().err
    assert _snapshot(repo) == before


def test_second_write_failure_restores_the_registry(repo, monkeypatch, capsys):
    import os
    new = _edit_policy(repo)
    before = _snapshot(repo)
    real = os.replace

    def flaky(src, dst):
        if str(dst).endswith("source-family-contracts.v1.json"):
            raise OSError("disk full (forced)")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    assert RP.repoint_main(_args(repo, new=new)) == 1
    assert "restored" in capsys.readouterr().err
    assert _snapshot(repo) == before
    assert OLD in (repo / "data/chunk-policy-registry.v1.json").read_text()
    assert not list((repo / "data").glob("*.tmp-repoint"))
