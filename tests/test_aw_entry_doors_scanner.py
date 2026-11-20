"""The entry-door scanner (cytir.ops.entry_doors) on synthetic modules: what it derives and what it refuses to guess."""
from __future__ import annotations

import pathlib

from cytir.ops import entry_doors as E

CLI = '''
_VERBS = {"serve": "cytir.pkg.verbed", "other": ("cytir.pkg.tupled", [])}
def run():
    from cytir.pkg import imported_mod
    import cytir.pkg.plain_import
'''


def _tree(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    for rel, text in {"cytir/__init__.py": "", "cytir/cli.py": CLI, **files}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_a_module_without_a_door_is_not_listed(tmp_path):
    app = _tree(tmp_path, {"cytir/pkg/lib.py": "def helper():\n    return 1\n"})
    assert "cytir.pkg.lib" not in E.scan_doors(app)


def test_def_main_and_name_guard_are_both_doors_and_lines_are_counted(tmp_path):
    app = _tree(tmp_path, {
        "cytir/pkg/a.py": "def main():\n    return 0\n",
        "cytir/pkg/b.py": "import sys\nif __name__ == '__main__':\n    sys.exit(0)\n",
        "cytir/pkg/c.py": "def main():\n    return 0\n\nif __name__ == '__main__':\n    main()\n",
    })
    d = E.scan_doors(app)
    assert (d["cytir.pkg.a"]["has_def_main"], d["cytir.pkg.a"]["has_name_guard"], d["cytir.pkg.a"]["block_lines"]) == (True, False, 2)
    assert (d["cytir.pkg.b"]["has_def_main"], d["cytir.pkg.b"]["has_name_guard"], d["cytir.pkg.b"]["block_lines"]) == (False, True, 2)
    assert d["cytir.pkg.c"]["block_lines"] == 4


def test_verb_means_the_cli_names_the_module_not_that_the_name_looks_like_one(tmp_path):
    body = "def main():\n    return 0\n"
    app = _tree(tmp_path, {f"cytir/pkg/{m}.py": body for m in
                           ("verbed", "tupled", "imported_mod", "plain_import", "serve", "loose")})
    d = E.scan_doors(app)
    assert {m: v["is_aw_verb"] for m, v in d.items() if m.startswith("cytir.pkg.")} == {
        "cytir.pkg.verbed": True, "cytir.pkg.tupled": True, "cytir.pkg.imported_mod": True,
        "cytir.pkg.plain_import": True, "cytir.pkg.serve": False, "cytir.pkg.loose": False}
    assert sorted(E.non_verb_doors(app)) == ["cytir.pkg.loose", "cytir.pkg.serve"]


def test_self_start_needs_a_spawn_of_the_module_itself(tmp_path):
    app = _tree(tmp_path, {
        "cytir/pkg/s1.py": "import subprocess, sys\ndef main():\n    subprocess.run([sys.executable, __file__, '--one'])\n",
        "cytir/pkg/s2.py": "import subprocess, sys\ndef main():\n    subprocess.run([sys.executable, '-m', 'cytir.pkg.s2'])\n",
        # spawns another module: a caller, not a self-starter
        "cytir/pkg/o1.py": "import subprocess, sys\ndef main():\n    subprocess.run([sys.executable, '-m', 'cytir.pkg.s1'])\n",
        # names its own file but never spawns
        "cytir/pkg/o2.py": "import sys\ndef main():\n    return [sys.executable, __file__]\n",
        # the repo-root idiom is not a self-start
        "cytir/pkg/o3.py": "import subprocess, pathlib\nR = pathlib.Path(__file__).parent\ndef main():\n    subprocess.run(['ls', str(R)])\n",
    })
    d = E.scan_doors(app)
    assert {m: v["self_start"] for m, v in d.items() if m.startswith("cytir.pkg.")} == {
        "cytir.pkg.s1": True, "cytir.pkg.s2": True, "cytir.pkg.o1": False, "cytir.pkg.o2": False, "cytir.pkg.o3": False}


def _row(module, kind="operator-tool"):
    return {"module": module, "kind": kind, "started_by": ["none"], "reason": "r", "test_that_starts_it": None, "verified": False}


def test_check_names_an_undeclared_door_and_a_stale_row_and_a_hidden_self_starter(tmp_path):
    app = _tree(tmp_path, {
        "cytir/pkg/new.py": "def main():\n    return 0\n",
        "cytir/pkg/kid.py": "import subprocess, sys\ndef main():\n    subprocess.run([sys.executable, __file__])\n",
    })
    reg = {"ceilings": {"non_verb_doors": 9, "docs_only": 9, "child_handlers_without_spawn_test": 9,
                        "live_unit_doors": 9, "unverified_rows": 9},
           "rows": [_row("cytir.pkg.gone"), _row("cytir.pkg.kid")]}
    msgs = "\n".join(E.problems(app, reg))
    assert "cytir.pkg.new: a new entry door" in msgs and "cyt ops entry-doors --write-candidates" in msgs
    assert "cytir.pkg.gone: stale row" in msgs
    assert "cytir.pkg.kid: the module starts itself as a child" in msgs


def test_ceilings_only_shrink_and_candidates_never_touch_the_data_file(tmp_path):
    app = _tree(tmp_path, {"cytir/pkg/new.py": "def main():\n    return 0\n"})
    reg = {"ceilings": {"non_verb_doors": 0, "docs_only": 0, "child_handlers_without_spawn_test": 0,
                        "live_unit_doors": 0, "unverified_rows": 0}, "rows": [_row("cytir.pkg.new")]}
    assert any(m.startswith("ceiling non_verb_doors: 1 > 0") for m in E.problems(app, reg))
    assert E.candidates(app, {"rows": [_row("cytir.pkg.new")]}) == []
    assert [c["module"] for c in E.candidates(app, {"rows": []})] == ["cytir.pkg.new"]
    assert not (app / "data").exists()


# --- review 2896: door shapes ---------------------------------------------------------------------
def _is_door(tmp_path, body: str, name: str = "d") -> bool:
    app = _tree(tmp_path, {f"cytir/pkg/{name}.py": body})
    return f"cytir.pkg.{name}" in E.scan_doors(app)


def test_every_door_shape_is_found(tmp_path):
    shapes = {
        "asyncmain": "async def main():\n    return 0\n",
        "reversed": "if '__main__' == __name__:\n    pass\n",
        "andexpr": "def f():\n    pass\n__name__ == '__main__' and f()\n",
        "andif": "import os\nif os.environ.get('X') and __name__ == '__main__':\n    pass\n",
        "intry": "try:\n    if __name__ == '__main__':\n        pass\nexcept Exception:\n    pass\n",
        "inwith": "import contextlib\nwith contextlib.suppress(Exception):\n    if __name__ == '__main__':\n        pass\n",
        "inmain": "if __name__ in ('__main__',):\n    pass\n",
        "inlist": "if __name__ in ['x', '__main__']:\n    pass\n",
        "inif": "import sys\nif sys.version_info:\n    if __name__ == '__main__':\n        pass\n",
    }
    app = _tree(tmp_path, {f"cytir/pkg/{k}.py": v for k, v in shapes.items()})
    found = E.scan_doors(app)
    assert {k for k in shapes if f"cytir.pkg.{k}" in found} == set(shapes)
    assert found["cytir.pkg.asyncmain"]["has_def_main"] is True


def test_non_doors_stay_non_doors(tmp_path):
    for i, body in enumerate(["def run():\n    pass\n", "x = 1\nif x == '__main__':\n    pass\n",
                              "class A:\n    def main(self):\n        pass\n",
                              "def f():\n    if __name__ == '__main__':\n        pass\n",
                              "if __name__ in ('other',):\n    pass\n"]):
        assert not _is_door(tmp_path / str(i), body)


def test_dunder_main_file_is_a_door_without_guard_or_def_main(tmp_path):
    app = _tree(tmp_path, {"cytir/pkg/__init__.py": "", "cytir/pkg/__main__.py": "from cytir.pkg import verbed\nverbed\n"})
    d = E.scan_doors(app)
    assert d["cytir.pkg.__main__"]["dunder_main_file"] is True and d["cytir.pkg.__main__"]["block_lines"] == 2
    assert "cytir.pkg.__main__" in E.non_verb_doors(app)


# --- review 2896: self-start forms ----------------------------------------------------------------
def _self(tmp_path, body: str) -> bool:
    app = _tree(tmp_path, {"cytir/pkg/me.py": "import os, sys, subprocess, pathlib\nfrom pathlib import Path\n"
                                           "def main():\n    pass\n" + body})
    return E.scan_doors(app)["cytir.pkg.me"]["self_start"]


def test_self_start_forms_are_found(tmp_path):
    yes = {
        "str_path": "subprocess.run([sys.executable, str(Path(__file__)), '--x'])\n",
        "resolve": "subprocess.run([sys.executable, str(Path(__file__).resolve())])\n",
        "rebuilt_name": "subprocess.run([sys.executable, str(Path(__file__).parent / 'me.py')])\n",
        "with_name": "subprocess.run([sys.executable, str(Path(__file__).with_name('me.py'))])\n",
        "join": "subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'me.py')])\n",
        "var_file": "me = Path(__file__).resolve()\nsubprocess.run([sys.executable, str(me)])\n",
        "var_interp": "py = sys.executable\nsubprocess.run([py, __file__])\n",
        "param_interp": "def go(python=sys.executable):\n    subprocess.run([python, '-c', 'BOOT', str(Path(__file__).resolve())])\n",
        "m_own": "subprocess.run([sys.executable, '-m', 'cytir.pkg.me'])\n",
        "m_name": "subprocess.run([sys.executable, '-m', __name__])\n",
        "argv0": "subprocess.run([sys.executable, sys.argv[0]])\n",
        "execv": "os.execv(sys.executable, [sys.executable] + sys.argv)\n",
        "fstring": "subprocess.run(f'{sys.executable} {__file__} --x', shell=True)\n",
        "bare_python": "subprocess.run(['python3', __file__])\n",
    }
    got = {k: _self(tmp_path / k, v) for k, v in yes.items()}
    assert got == {k: True for k in yes}, [k for k, v in got.items() if not v]


def test_self_start_is_not_over_matched(tmp_path):
    no = {
        # a sibling file or the directory is not the module itself (decision: only the same file name counts)
        "sibling": "subprocess.run([sys.executable, str(Path(__file__).parent / 'other.py')])\n",
        "dir": "subprocess.run([sys.executable, str(Path(__file__).parent)])\n",
        "parents": "subprocess.run(['ls', str(Path(__file__).resolve().parents[1])])\n",
        "other_m": "subprocess.run([sys.executable, '-m', 'cytir.pkg.other'])\n",
        "no_interp": "subprocess.run(['ls', __file__])\n",
        "no_spawn": "x = [sys.executable, __file__]\n",
        "fstring_no_py": "subprocess.run(f'ls {__file__}', shell=True)\n",
    }
    got = {k: _self(tmp_path / k, v) for k, v in no.items()}
    assert got == {k: False for k in no}, [k for k, v in got.items() if v]


# --- review 2896: the data file absent, and row fields --------------------------------------------
def test_check_without_the_registry_is_ok_only_in_a_public_cut(tmp_path, capsys):
    app = _tree(tmp_path / "full", {"cytir/pkg/x.py": "def main():\n    pass\n"})
    assert E.main(["--check", "--root", str(app)]) == 2
    assert "missing in a full checkout" in capsys.readouterr().out
    cut = _tree(tmp_path / "cut", {"cytir/pkg/x.py": "def main():\n    pass\n", E.CUT_MARKER: "{}"})
    assert E.main(["--check", "--root", str(cut)]) == 0
    assert "registry file not shipped in this tree; nothing to check" in capsys.readouterr().out


def test_row_fields_are_checked_against_the_tree(tmp_path):
    app = _tree(tmp_path, {
        "cytir/pkg/kid.py": "import subprocess, sys\ndef main():\n    subprocess.run([sys.executable, __file__])\n",
        "tests/test_lazy.py": "def test_x():\n    assert 'kid' in open(__file__).read()\n",
        "tests/test_real.py": "import subprocess\nsubprocess.run(['python', '-m', 'cytir.pkg.kid'])\n",
        "scripts/go.sh": "echo\n"})
    ceil = {k: 9 for k in E.CEILINGS}

    def row(**kw):
        return {**_row("cytir.pkg.kid", "child-handler"), **kw}
    bad = {"ceilings": ceil, "rows": [row(started_by=["unit:scripts/missing.sh", "doc:nope.md"],
                                           test_that_starts_it="tests/test_lazy.py")]}
    out = "\n".join(E.problems(app, bad))
    assert "started_by `unit:scripts/missing.sh` names a path that does not exist" in out
    assert "started_by `doc:nope.md`" in out
    assert "does not spawn the module" in out
    missing = "\n".join(E.problems(app, {"ceilings": ceil, "rows": [row(test_that_starts_it="tests/test_gone.py")]}))
    assert "test_that_starts_it `tests/test_gone.py` is not a file" in missing
    good = {"ceilings": ceil, "rows": [row(started_by=["unit:scripts/go.sh", "unit:unknown"],
                                            test_that_starts_it="tests/test_real.py")]}
    assert E.problems(app, good) == []
