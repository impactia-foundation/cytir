"""Drift ratchet for raw HTTP clients (task 1623, consent choke point spec section 4, test 7)."""
import json

from cytir.ops import http_drift
import pytest

pytestmark = pytest.mark.guard


def test_the_package_matches_the_allowlist():
    assert http_drift.check() == []


def test_the_recorded_max_equals_the_set():
    doc = json.loads(http_drift.ALLOWLIST.read_text())
    assert doc["max"] == len(doc["modules"])


def _tree(tmp_path, offenders, listed):
    pkg = tmp_path / "cytir" / "x"
    pkg.mkdir(parents=True)
    for name, body in offenders.items():
        (pkg / name).write_text(body)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"modules": {f"cytir/x/{n}": "r" for n in listed}, "max": len(listed)}))
    return allow


def test_a_new_offender_and_a_stale_entry_both_fail(tmp_path):
    allow = _tree(tmp_path, {"bad.py": "import httpx\nc = httpx.Client()\n", "ok.py": "x = 1\n"},
                  listed=["ok.py"])
    errs = http_drift.check(tmp_path, allow)
    assert any("bad.py" in e and "not in the allow-list" in e for e in errs)
    assert any("ok.py" in e and "--rebuild" in e for e in errs)


def test_a_listed_offender_passes_and_rebuild_keeps_reasons(tmp_path):
    allow = _tree(tmp_path, {"bad.py": "requests.get(u)\n"}, listed=["bad.py"])
    assert http_drift.check(tmp_path, allow) == []
    doc = http_drift.rebuild(tmp_path, allow, tmp_path / "none.md")
    assert doc == {"modules": {"cytir/x/bad.py": "r"}, "max": 1}
