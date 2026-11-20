"""Sealed consent probe expiry guard (task 2532). Frozen --as-of, never the wall clock."""
import json

from cytir.consent import expiry_check
from tests.helpers.sealed_evidence import sealed_eric_probe  # noqa: F401  (skips where the cut lacks the probe)

PTR = "probes/p.json#host_consent_resolution"


def _run(capsys, *argv):
    rc = expiry_check.main(list(argv))
    return rc, capsys.readouterr().out


def test_real_contracts_fresh(capsys, sealed_eric_probe):
    rc, out = _run(capsys, "--within-days", "7", "--as-of", "2026-11-08T12:00:00Z")
    assert rc == 0
    assert "earliest r6 expiry: 2026-12-08T05:14:5" in out


def test_real_contracts_inside_window(capsys, sealed_eric_probe):
    rc, out = _run(capsys, "--within-days", "7", "--as-of", "2026-12-02T00:00:00Z")
    assert rc == 1 and "FAIL:" in out and "re-seal" in out


def test_real_contracts_expired(capsys, sealed_eric_probe):
    rc, out = _run(capsys, "--within-days", "7", "--as-of", "2026-12-09")
    assert rc == 1 and "FAIL:" in out


def _fixture(tmp_path, old_r6=False):
    def fam(fid, resolution):
        return {"family_id": fid, "status": "declared", "route": "direct",
                "consent": {"resolution": resolution, "search_ok": True, "train_ok": False, "evidence": [PTR]}}
    contracts = {"families": [fam("r6fam", "live-per-origin"), fam("oldfam", "explicit-grant")]}

    def rec(t):
        return {"status": "ok", "search_ok": True, "train_ok": False, "checked_at": t}
    fresh, old = "2026-11-08T00:00:00Z", "2026-01-01T00:00:00Z"
    probe = {"family_verdicts": {"r6fam": {"hosts": ["a.example"]}, "oldfam": {"hosts": ["b.example"]}},
             "host_consent_resolution": {"a.example": rec(old if old_r6 else fresh), "b.example": rec(old)}}
    (tmp_path / "probes").mkdir()
    (tmp_path / "probes" / "p.json").write_text(json.dumps(probe))
    cpath = tmp_path / "contracts.json"
    cpath.write_text(json.dumps(contracts))
    return ["--contracts", str(cpath), "--repo-root", str(tmp_path), "--as-of", "2026-11-09T00:00:00Z"]


def test_non_r6_old_record_ignored(tmp_path, capsys):
    rc, out = _run(capsys, *_fixture(tmp_path))
    assert rc == 0 and "b.example" not in out and "a.example" in out


def test_r6_old_record_fails(tmp_path, capsys):
    rc, out = _run(capsys, *_fixture(tmp_path, old_r6=True))
    assert rc == 1


def test_no_probe_files_skips(tmp_path, capsys):
    argv = _fixture(tmp_path)
    (tmp_path / "probes" / "p.json").unlink()
    rc, out = _run(capsys, *argv)
    assert rc == 0 and out.startswith("SKIP: no sealed probe files")
