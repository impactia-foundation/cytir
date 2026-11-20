"""Tests for cytir/ops/public_cut.py (refactor slice 1, task 696): the public cut is an allowlist
export that refuses a NEW leak, never ships a `never` path, and is byte-reproducible.

The fixtures spell the private names in pieces, so this file is not itself a leak."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.cli import main as aw_main
from cytir.ops import public_cut as PC

pytestmark = pytest.mark.guard

P, T, H = "pipe" + "line", "ta" + "sks", "heim" + "dall"
KEY = "AK" + "IA" + "ABCDEFGHIJKLMNOP"


def _w(root: Path, rel: str, text: str = "x = 1\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _allowlist(root: Path, **over) -> None:
    doc = {"schema": PC.SCHEMA,
           "rules": [{"id": "pkg", "why": "the package", "paths": ["cytir/**", "README.md"]},
                     {"id": "extra", "why": "named later", "optional": True,
                      "paths": ["LICENSE"]}],
           "never": [{"path": f"{P}/**", "why": "private"}, {"path": "plan.md", "why": "private"}],
           "scan_exempt": [PC.ALLOWLIST_REL, PC.BASELINE_REL]}
    doc.update(over)
    _w(root, PC.ALLOWLIST_REL, json.dumps(doc))


@pytest.fixture()
def app(tmp_path: Path) -> Path:
    r = tmp_path / "app"
    _w(r, "cytir/__init__.py", "")
    _w(r, "cytir/mod_a.py", "def f():\n    return 1\n")
    _w(r, "README.md", "hello\n")
    _w(r, f"{P}/secret_tool.py", "y = 2\n")
    _w(r, "plan.md", "plan\n")
    _allowlist(r)
    return r


def _record(root: Path) -> None:
    assert PC.main(["--root", str(root), "--record-baseline"]) == 0


# --------------------------------------------------------------------------- the allowlist


def test_glob_double_star_crosses_directories_and_star_does_not():
    assert PC._matches("cytir/**", "cytir/core/x.py") and PC._matches("cytir/**", "cytir/x.py")
    assert PC._matches("data/chunk-*.json", "data/chunk-a.json")
    assert not PC._matches("data/*.json", "data/sub/a.json")
    assert PC._matches("server/test_*.py", "server/test_a.py")
    assert not PC._matches("README.md", "docs/README.md")


def test_only_named_paths_ship_and_never_paths_are_swept_out(app):
    p = PC.plan(app)
    assert set(p["shipped"]) == {"cytir/__init__.py", "cytir/mod_a.py", "README.md"}
    assert not p["problems"]
    assert all(rule for rule in p["shipped"].values())


def test_a_wildcard_that_sweeps_a_never_path_does_not_ship_it(app):
    _allowlist(app, rules=[{"id": "all", "why": "too broad", "paths": ["**"]}])
    shipped, problems = PC.select(PC.load_allowlist(app), PC.tracked_files(app))
    assert f"{P}/secret_tool.py" not in shipped and "plan.md" not in shipped
    assert "README.md" in shipped and not problems


def test_a_literal_path_that_never_forbids_is_refused(app):
    _allowlist(app, rules=[{"id": "oops", "why": "contradiction", "paths": ["plan.md"]}])
    assert PC.main(["--root", str(app), "--check"]) == 2


def test_a_rule_that_matches_nothing_is_refused_unless_optional(app):
    _allowlist(app, rules=[{"id": "ghost", "why": "no file", "paths": ["docs/**"]}])
    assert PC.plan(app)["problems"]
    _allowlist(app, rules=[{"id": "ghost", "why": "no file", "optional": True,
                            "paths": ["docs/**"]}])
    assert not PC.plan(app)["problems"]


def test_an_optional_stranger_file_ships_the_moment_it_exists(app):
    _record(app)
    assert "LICENSE" not in PC.plan(app)["shipped"]
    _w(app, "LICENSE", "Apache\n")
    assert PC.plan(app)["shipped"]["LICENSE"] == "extra"


def test_every_rule_needs_an_id_a_reason_and_paths(app):
    _allowlist(app, rules=[{"id": "x", "paths": ["cytir/**"]}])
    with pytest.raises(PC.CutRefusal):
        PC.load_allowlist(app)


# --------------------------------------------------------------------------- the scan


def test_scan_names_each_kind_of_leak():
    text = (f"see {P}/build.py and {T}/12-scripts/x.sh\n"
            f"from {P} import thing\nroot = '/home/{'claude'}/x'  # {H}\n"
            f"aws = '{KEY}'\n")
    got = PC.scan_text(text)
    assert got["pipeline_ref"] == 2 and got["tasks_ref"] == 1
    assert got["host_literal"] == 2 and got["secret_literal"] == 1


def test_the_public_contact_mailbox_is_not_a_host_literal():
    assert PC.scan_text("write to ben" + "der@impactia.org\n") == {}
    assert PC.scan_text("ssh ben" + "der.impactia.org\n") == {"host_literal": 1}


def test_scan_ignores_prose_that_is_not_a_path():
    assert PC.scan_text(f"the build {P} is a word; {T} are done; {P}ing\n") == {}


def test_an_exempt_file_may_name_private_paths(app):
    _w(app, "cytir/rules.py", f"X = '{P}/x.py'\n")
    _allowlist(app, scan_exempt=["cytir/rules.py"])
    shipped = PC.plan(app)["shipped"]
    assert not any(PC.scan(app, shipped, ["cytir/rules.py"]).values())


# --------------------------------------------------------------------------- the ratchet


def test_a_new_leak_is_refused_with_the_list_and_nothing_is_exported(app, tmp_path, capsys):
    _record(app)
    _w(app, "cytir/mod_a.py", f"def f():\n    return '{P}/new_tool.py'\n")
    out = tmp_path / "out"
    assert PC.main(["--root", str(app), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "pipeline_ref: cytir/mod_a.py (1x) is not in the baseline" in err
    assert not out.exists()


def test_a_known_leak_passes_but_a_bigger_count_does_not(app):
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n")
    _record(app)
    assert PC.main(["--root", str(app), "--check"]) == 0
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n# {P}/two.py\n")
    assert PC.main(["--root", str(app), "--check"]) == 1


def test_a_fix_passes_and_tighten_lowers_the_baseline_but_never_raises_it(app, capsys):
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n# {P}/two.py\n")
    _record(app)
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n")
    assert PC.main(["--root", str(app), "--check"]) == 0
    assert "--tighten" in capsys.readouterr().err
    assert PC.main(["--root", str(app), "--tighten", "--scheduled"]) == 0
    assert PC.load_baseline(app)["pipeline_ref"] == {"cytir/mod_a.py": 1}
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n# {P}/two.py\n")
    assert PC.main(["--root", str(app), "--tighten", "--scheduled"]) == 1       # refuses to raise
    assert PC.load_baseline(app)["pipeline_ref"] == {"cytir/mod_a.py": 1}


def test_the_baseline_cannot_be_re_recorded_over_recorded_leaks(app):
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n")
    _record(app)
    assert PC.main(["--root", str(app), "--record-baseline"]) == 2


def test_a_missing_baseline_is_a_refusal_not_a_pass(app):
    assert PC.main(["--root", str(app), "--check"]) == 2


# --------------------------------------------------------------------------- the export


def test_export_is_a_fresh_reproducible_tree_with_a_verifiable_manifest(app, tmp_path):
    _record(app)
    a, b = tmp_path / "a", tmp_path / "b"
    assert PC.main(["--root", str(app), "--out", str(a)]) == 0
    assert PC.main(["--root", str(app), "--out", str(b)]) == 0
    assert (a / PC.MANIFEST_NAME).read_bytes() == (b / PC.MANIFEST_NAME).read_bytes()
    assert not (a / P).exists() and not (a / "plan.md").exists()
    assert not (a / ".git").exists()
    assert PC.verify_tree(a) == []
    # a second export into a used directory is refused: the cut is always a fresh tree
    assert PC.main(["--root", str(app), "--out", str(a)]) == 2


def test_verify_finds_a_changed_a_missing_and_an_extra_file(app, tmp_path):
    _record(app)
    out = tmp_path / "out"
    PC.main(["--root", str(app), "--out", str(out)])
    (out / "cytir/mod_a.py").write_text("tampered\n")
    (out / "README.md").unlink()
    _w(out, "extra.txt", "x")
    got = PC.verify_tree(out)
    assert {"changed: cytir/mod_a.py", "missing: README.md", "not in the manifest: extra.txt"} <= set(got)


def test_the_cli_verb_reaches_the_cut(app, capsys):
    _record(app)
    assert aw_main(["ops", "cut", "--root", str(app), "--check"]) == 0
    assert aw_main(["ops"]) == 2


# --------------------------------------------------------------------------- this repo


def test_this_repo_cut_is_consistent_and_has_no_new_leak():
    p = PC.plan()
    assert p["problems"] == []
    assert p["baseline"] is not None and p["new"] == [], p["new"]
    assert p["leaks"]["secret_literal"] == {}


def test_this_repo_never_ships_the_private_trees():
    shipped = PC.plan()["shipped"]
    bad = [f for f in shipped if f.startswith((f"{P}/", f"{T}/", "notes/", "deploy/"))
           or f in ("plan.md", "log.md", "decisions.md", "data/oss-refactor-live-entrypoints.v1.json")]
    assert bad == []
    assert "cytir/cli.py" in shipped and "pyproject.toml" in shipped


def test_operator_tests_directory_is_never_shipped_and_every_module_says_so():
    """Task 2412: `tests/operator/` holds the tests that need our overlay, box paths or pinned values."""
    doc = PC.load_allowlist()
    rows = [n for n in doc["never"] if (n["path"] if isinstance(n, dict) else n) == "tests/operator/**"]
    assert len(rows) == 1 and rows[0]["why"]
    plan = PC.plan()
    assert [f for f in plan["shipped"] if f.startswith("tests/operator")] == []
    here = Path(__file__).resolve().parent / "operator"
    if here.is_dir():      # absent in the cut itself
        for m in sorted(here.glob("test_*.py")):
            first = m.read_text().splitlines()[0]
            assert first == "# operator-only test: never shipped (tests/operator/)", m.name


def test_this_repo_names_the_four_stranger_files():
    doc = PC.load_allowlist()
    rule = next(r for r in doc["rules"] if r["id"] == "stranger-files")
    assert rule["optional"] is True
    assert set(rule["paths"]) == {"LICENSE", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md"}


# --------------------------------------------------------------------------- --admit-moved


def _g(root: Path, *args: str) -> None:
    import subprocess
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@e.x",
                    *args], check=True, capture_output=True)


MOVED_TEXT = f"# see {P}/mod_a.py and {P}/mod_b.py and {T}/9-x/y.sh\nX = '{H}'\n"


@pytest.fixture()
def moved(app: Path) -> Path:
    """main holds a private file with mentions; the branch moves it into the package."""
    _w(app, f"{P}/tool.py", MOVED_TEXT)
    _record(app)
    _g(app, "init", "-q", "-b", "main")
    _g(app, "add", "-A")
    _g(app, "commit", "-qm", "main")
    _g(app, "checkout", "-q", "-b", "move")
    return app


def _admit(root: Path, frm: str, to: str) -> int:
    return PC.main(["--root", str(root), "--admit-moved", frm, to, "--base", "main",
                    "--reason", "test"])


def _baseline(root: Path) -> dict:
    return json.loads((root / PC.BASELINE_REL).read_text(encoding="utf-8"))


def test_a_real_rename_is_admitted_with_exact_counts(moved):
    _g(moved, "mv", f"{P}/tool.py", "cytir/tool.py")
    assert PC.main(["--root", str(moved), "--check"]) == 1       # refused before
    assert _admit(moved, f"{P}/tool.py", "cytir/tool.py") == 0
    doc = _baseline(moved)
    assert doc["leaks"]["pipeline_ref"]["cytir/tool.py"] == 2
    assert doc["leaks"]["tasks_ref"]["cytir/tool.py"] == 1
    assert doc["leaks"]["host_literal"]["cytir/tool.py"] == 1
    (rec,) = doc["admitted"]
    assert rec["kind"] == "moved" and rec["from"] == f"{P}/tool.py" and rec["to"] == "cytir/tool.py"
    assert rec["counts"] == {"pipeline_ref": 2, "tasks_ref": 1, "host_literal": 1}
    assert rec["reason"] == "test" and rec["branch"] == "move" and rec["date"]
    assert PC.main(["--root", str(moved), "--check"]) == 0       # and the ratchet now passes


def test_a_moved_file_that_gained_a_mention_is_refused_for_the_excess(moved, capsys):
    _g(moved, "mv", f"{P}/tool.py", "cytir/tool.py")
    _w(moved, "cytir/tool.py", MOVED_TEXT + f"# and {P}/c.py\n")     # 3 pipeline mentions, had 2
    assert _admit(moved, f"{P}/tool.py", "cytir/tool.py") == 1
    assert "grew 2 -> 3" in capsys.readouterr().err
    assert _baseline(moved)["leaks"]["pipeline_ref"]["cytir/tool.py"] == 2


def test_a_new_file_is_not_admitted(moved, capsys):
    _w(moved, "cytir/fresh.py", f"# {P}/mod_a.py\n")
    _g(moved, "add", "-A")
    assert _admit(moved, f"{P}/tool.py", "cytir/fresh.py") == 2
    assert "does not show" in capsys.readouterr().err
    assert "cytir/fresh.py" not in _baseline(moved)["leaks"]["pipeline_ref"]
    assert "admitted" not in _baseline(moved)


def test_a_copy_is_not_admitted(moved):
    _w(moved, "cytir/copy.py", MOVED_TEXT)                           # original stays: not a move
    _g(moved, "add", "-A")
    assert _admit(moved, f"{P}/tool.py", "cytir/copy.py") == 2
    assert "cytir/copy.py" not in _baseline(moved)["leaks"]["pipeline_ref"]


def test_tighten_keeps_the_admitted_list(moved):
    _g(moved, "mv", f"{P}/tool.py", "cytir/tool.py")
    assert _admit(moved, f"{P}/tool.py", "cytir/tool.py") == 0
    _w(moved, "cytir/tool.py", f"# {P}/mod_a.py\n")                      # the file shrinks
    assert PC.main(["--root", str(moved), "--tighten", "--scheduled"]) == 0
    doc = _baseline(moved)
    assert doc["leaks"]["pipeline_ref"]["cytir/tool.py"] == 1
    assert len(doc["admitted"]) == 1 and doc["admitted"][0]["to"] == "cytir/tool.py"
    assert PC.load_admitted(moved) == doc["admitted"]


# --- task 739: declared frozen receipt labels ---------------------------------------------------

def test_a_declared_frozen_label_is_exempt_only_for_that_literal_in_that_file(app, monkeypatch):
    label = f"{P}/frozen_label.py"
    monkeypatch.setattr(PC, "FROZEN_RECEIPT_LABELS", (("cytir/seal.py", label, "test"),))
    _w(app, "cytir/seal.py", f"ID = '{label}'\nOTHER = '{P}/not_declared.py'\n")
    _w(app, "cytir/other.py", f"ID = '{label}'\n")
    shipped = PC.plan(app)["shipped"]
    exempted: dict = {}
    leaks = PC.scan(app, shipped, [], exempted)
    assert leaks["pipeline_ref"].get("cytir/seal.py") == 1      # the unlisted literal is still a leak
    assert leaks["pipeline_ref"].get("cytir/other.py") == 1     # same literal, other file: a leak
    assert exempted == {"cytir/seal.py": 1}


def test_a_stale_frozen_label_entry_is_refused(app, monkeypatch):
    monkeypatch.setattr(PC, "FROZEN_RECEIPT_LABELS", (("cytir/seal.py", f"{P}/gone.py", "test"),))
    _w(app, "cytir/seal.py", "ID = 'clean'\n")
    with pytest.raises(PC.CutRefusal, match="stale"):
        PC.plan(app)


def test_the_real_frozen_labels_are_plain_literals_in_their_files():
    relabeled = {r for r, rule in PC.strip_rules(json.loads((PC.APP_ROOT / PC.ALLOWLIST_REL).read_text())).items()
                 if rule["relabel"]} if (PC.APP_ROOT / PC.ALLOWLIST_REL).is_file() else set()
    for rel, literal, reason in PC.FROZEN_RECEIPT_LABELS:
        text = (PC.APP_ROOT / rel).read_text()
        # task 2313: inside the cut a relabeled contract carries the frozen:task-N label, not the path
        assert literal in text or (rel in relabeled and PC.frozen_label(literal) in text), (rel, literal)
        assert ("receipt bytes" in reason or reason == PC._RETIRED_PATH_REASON
                or reason == PC._REGISTRY_KEY_REASON
                or reason == PC._DESCRIPTOR_LABEL_REASON)


def test_split_string_spellings_exist_only_in_the_scanner_and_its_declared_labels():
    import re
    pattern = re.compile(r'"(?:pipe|ta)"\s*\+\s*"(?:line|sks)')
    offenders = [str(p.relative_to(PC.APP_ROOT)) for p in (PC.APP_ROOT / "cytir").rglob("*.py")
                 if pattern.search(p.read_text())]
    assert offenders == ["cytir/ops/public_cut.py"]


# --- task 1313, decision D13: the label exemption ------------------------------------------------

def test_a_label_exempt_file_may_hold_private_path_labels_but_not_hosts(app):
    doc = {"prose_exempt": [{"path": "cytir/a.json", "labels": True, "why": "t"},
                            {"path": "cytir/b.json", "why": "t"}]}
    assert PC.label_exempt_paths(doc) == ["cytir/a.json"]
    body = json.dumps({"label": f"{T}/12-x/y.json", "where": H})
    _w(app, "cytir/a.json", body)
    _w(app, "cytir/b.json", body)
    shipped = PC.plan(app)["shipped"]
    leaks = PC.scan(app, shipped, [], None, ["cytir/a.json", "cytir/b.json"], ["cytir/a.json"])
    assert "cytir/a.json" not in leaks["tasks_ref"]             # the label field is exempt
    assert leaks["tasks_ref"].get("cytir/b.json") == 1          # a plain prose-exempt file still counts it
    assert leaks["host_literal"].get("cytir/a.json") == 1       # a host is never a label


def test_the_real_allowlist_declares_the_six_label_files():
    # Task 1313 (D13) set the label-exempt list; task 2434 adds source-family-contracts (22 frozen labels).
    doc = json.loads((PC.APP_ROOT / PC.ALLOWLIST_REL).read_text(encoding="utf-8"))
    assert sorted(PC.label_exempt_paths(doc)) == [
        "data/chunk-policy-preserved-retained-passage.v1.json", "data/chunk-policy-registry.v1.json",
        "data/content-gate-convergence.v1.json", "data/persisted-input-stores.v1.json",
        "data/source-family-contracts.v1.json", "data/unpersisted-legacy-register.v1.json"]


# --------------------------------------------------------------------------- split literals
# Task 2264: a literal split across constants hides the path from the text scan.
_A, _B = P[:4], P[4:]          # "pipe", "line"
_C, _D = T[:2], T[2:]          # "ta", "sks"


@pytest.mark.parametrize("src,kind", [
    (f'x = "{_A}" "{_B}/mod_a.py"\n', "pipeline_ref"),                       # (a) adjacent constants
    (f'x = ("{_A}"\n     "{_B}/mod_a.py")\n', "pipeline_ref"),                # (a) across lines
    (f'x = "{_A}{_B}" + "/mod_a.py"\n', "pipeline_ref"),                     # (b) +
    (f'x = "{_C}" + "{_D}/12.md"\n', "tasks_ref"),                       # (b) +
    (f'x = "/".join(("{_A}{_B}", "mod_a.py"))\n', "pipeline_ref"),           # (c) join tuple
    (f'x = "/".join(["{_C}{_D}", "12.md"])\n', "tasks_ref"),             # (c) join list
    (f'import os\nx = os.path.join("{_A}{_B}", "mod_a.py")\n', "pipeline_ref"),    # (d) os.path.join
    (f'from pathlib import Path\nx = Path("{_A}{_B}") / "mod_a.py"\n', "pipeline_ref"),  # (d) pathlib
    (f'from pathlib import Path\nx = Path("{_C}" "{_D}") / "12.md"\n', "tasks_ref"),
])
def test_split_literal_is_a_leak(src, kind):
    assert PC.scan_split_literals(src) == {kind: 1}


@pytest.mark.parametrize("src", [
    f'from pathlib import Path\nx = REPO / "{_A}{_B}" / name\n',          # honest: non-constant part
    f'x = "/".join(("{_A}{_B}", name))\n',
    f'x = "{_A}{_B}" + suffix\n',
    f'import os\nx = os.path.join(root, "{_A}{_B}", "mod_a.py")\n',
    'x = "pipe" "lines"\n',                                               # no path after the name
    f'x = "{_A}{_B}/mod_a.py"\n',                                             # unsplit: the text scan owns it
    f'"""doc {_A}" "{_B}/mod_a.py"""\n',
])
def test_honest_parts_are_not_split_leaks(src):
    assert PC.scan_split_literals(src) == {}


def test_split_literal_counted_once_and_skips_registered_label():
    both = f'x = "{_A}{_B}/mod_a.py" + "mod_b.py"\n'     # the plain scan already sees the full literal
    assert PC.scan_split_literals(both) == {}
    src = f'x = "{_A}" "{_B}/mod_a.py"\n'
    assert PC.scan_split_literals(src, skip=(f"{P}/mod_a.py",)) == {}
    assert PC.scan_split_literals("def (:\n") == {}


def test_scan_reports_split_literal_in_python_only(tmp_path):
    _w(tmp_path, "cytir/mod_a.py", f'x = "{_A}" "{_B}/mod_a.py"\n')
    _w(tmp_path, "cytir/a.txt", f'x = "{_A}" "{_B}/mod_a.py"\n')
    leaks = PC.scan(tmp_path, {"cytir/mod_a.py": "", "cytir/a.txt": ""}, [])
    assert leaks["pipeline_ref"] == {"cytir/mod_a.py": 1}


def test_all_constant_fstring_is_a_split_leak():
    # an f-string whose every part is a constant or a name bound to one
    src = 'D = "' + _A + '" "' + _B + '"\nx = f"{D}/mod_a.py"\n'
    assert PC.scan_split_literals(src) == {"pipeline_ref": 1}
    src = 'x = f"{' + "'" + _A + "' '" + _B + "'" + '}/mod_a.py"\n'   # inner quotes differ: valid before 3.12
    assert PC.scan_split_literals(src) == {"pipeline_ref": 1}


def test_module_level_constant_name_in_fstring_or_join_is_a_split_leak():
    d = 'D = "' + _A + '" + "' + _B + '"\n'
    assert PC.scan_split_literals(d + 'x = "/".join((D, "mod_a.py"))\n') == {"pipeline_ref": 1}
    assert PC.scan_split_literals(d + 'x = D + "/mod_a.py"\n') == {"pipeline_ref": 1}
    assert PC.scan_split_literals(d + 'x = f"{D}/mod_a.py"\n') == {"pipeline_ref": 1}


def test_name_bound_to_a_non_constant_is_the_honest_form():
    d = 'D = "' + _A + '" + "' + _B + '"\n'
    assert PC.scan_split_literals('D = get()\nx = f"{D}/mod_a.py"\n') == {}
    assert PC.scan_split_literals(d + 'x = f"{D}/{name}"\n') == {}
    assert PC.scan_split_literals(d + 'x = "/".join((D, name))\n') == {}


# --- ruling 68(b): admitted stored values --------------------------------------------------------

STORED = "cytir/enum.json"
STORED_TEXT = f'["{H} reprobe a", "{H} reprobe b"]\n'


def _admit_stored(root: Path, n: int) -> int:
    return PC.main(["--root", str(root), "--admit-stored", STORED, "host_literal", str(n),
                    "--reason", "ruling 68(b): stored enum value"])


def test_an_admitted_stored_value_pair_is_subtracted(app):
    _w(app, STORED, STORED_TEXT)
    _record(app)
    assert PC.load_baseline(app)["host_literal"] == {STORED: 2}
    assert _admit_stored(app, 2) == 0
    assert PC.load_baseline(app)["host_literal"] == {}
    (rec,) = _baseline(app)["admitted"]
    assert rec["kind"] == "stored-value" and rec["file"] == STORED and rec["count"] == 2
    assert PC.main(["--root", str(app), "--check"]) == 0


def test_growth_above_the_admitted_count_refuses(app, capsys):
    _w(app, STORED, STORED_TEXT)
    _record(app)
    assert _admit_stored(app, 2) == 0
    _w(app, STORED, STORED_TEXT + f'["{H} reprobe c"]\n')
    assert PC.main(["--root", str(app), "--check"]) == 1
    assert f"host_literal: {STORED} (1x) is not in the baseline" in capsys.readouterr().err


def test_a_stale_admission_above_the_file_count_refuses(app, capsys):
    _w(app, STORED, STORED_TEXT)
    _record(app)
    assert _admit_stored(app, 2) == 0
    _w(app, STORED, f'["{H} reprobe a"]\n')
    assert PC.main(["--root", str(app), "--check"]) == 1
    assert "re-admit at the lower count" in capsys.readouterr().err
    assert _admit_stored(app, 2) == 2                 # cannot admit more than the file has


def test_tighten_keeps_the_stored_value_entry(app):
    _w(app, STORED, STORED_TEXT)
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n# {P}/two.py\n")
    _record(app)
    assert _admit_stored(app, 2) == 0
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n")
    assert PC.main(["--root", str(app), "--tighten", "--scheduled"]) == 0
    doc = _baseline(app)
    assert doc["admitted"][0]["kind"] == "stored-value" and doc["admitted"][0]["count"] == 2
    assert doc["leaks"]["pipeline_ref"] == {"cytir/mod_a.py": 1} and doc["leaks"]["host_literal"] == {}


# --------------------------------------------------------------------------- --tighten guard (task 2330)


@pytest.fixture()
def two_files(app: Path) -> Path:
    """main holds a STALE ceiling for mod_b (baseline 2, file already at 1); the branch fixes mod_a."""
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n# {P}/two.py\n")
    _w(app, "cytir/mod_b.py", f"# {P}/one.py\n# {P}/two.py\n")
    _record(app)
    _w(app, "cytir/mod_b.py", f"# {P}/one.py\n")
    _g(app, "init", "-q", "-b", "main")
    _g(app, "add", "-A")
    _g(app, "commit", "-qm", "main")
    _g(app, "checkout", "-q", "-b", "lane")
    _w(app, "cytir/mod_a.py", f"# {P}/one.py\n")
    return app


def _tighten(root: Path, *extra: str) -> int:
    return PC.main(["--root", str(root), "--tighten", "--base", "main", *extra])


def test_a_below_baseline_file_passes_check(two_files):
    assert PC.main(["--root", str(two_files), "--check"]) == 0


def test_tighten_refuses_files_the_branch_did_not_change(two_files, capsys):
    before = (two_files / PC.BASELINE_REL).read_text()
    assert _tighten(two_files) == 2
    assert "did not change" in capsys.readouterr().err
    assert (two_files / PC.BASELINE_REL).read_text() == before


def test_scheduled_tighten_lowers_every_file(two_files):
    assert _tighten(two_files, "--scheduled") == 0
    assert PC.load_baseline(two_files)["pipeline_ref"] == {"cytir/mod_a.py": 1, "cytir/mod_b.py": 1}


def test_tighten_passes_when_only_the_branchs_own_files_shrink(two_files):
    _w(two_files, "cytir/mod_b.py", f"# {P}/one.py\n# {P}/two.py\n")   # mod_b back at its ceiling
    assert _tighten(two_files) == 0
    assert PC.load_baseline(two_files)["pipeline_ref"] == {"cytir/mod_a.py": 1, "cytir/mod_b.py": 2}


def test_the_daily_tighten_script_parses():
    import subprocess
    found = list((Path(__file__).resolve().parents[1] / "deploy").glob("*/leak-baseline-tighten.sh"))
    if not found:
        pytest.skip("deploy scripts are not shipped in the public cut")
    script = found[0]
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert "--tighten --scheduled" in script.read_text()
