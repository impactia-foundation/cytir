"""The `CYTIR_*` <-> `AW_*` environment-name mirror (task 2964, RULING 85).

What is proven here:
  1. `normalise` mirrors both ways, accepts equal pairs, refuses a different pair without printing
     a value, and is idempotent.
  2. Importing the package mirrors `os.environ` (subprocess), and a conflicting pair is refused.
  3. The settings loaders accept `CYTIR_*` in an INJECTED env and never change the caller's dict.
  4. Every prefix filter treats `CYTIR_` as it treats `AW_`.
  5. The public docs name no `AW_` variable.

KNOWN LIMIT. A `monkeypatch.setenv("AW_X")` after import keeps working: code reads `AW_X`. A
`CYTIR_X` put in `os.environ` AFTER import reaches the settings loaders (they mirror on every call)
but NOT a direct `os.environ["AW_X"]` read; such a program calls `normalise(os.environ)` itself.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
from unittest import mock
import sys

import pytest

from cytir.core import envnames as E
from cytir.core import paths as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = "import cytir, os; print(os.environ.get('AW_WORK_ROOT'))"

def _vals(env):
    return {k: v for k, v in env.items() if k != E.MARKER}



# ------------------------------------------------------------------------------ 1. normalise
def test_mirror_both_ways():
    env = {"CYTIR_A": "1", "AW_B": "2", "PATH": "/bin"}
    added = E.normalise(env)
    assert _vals(env) == {"CYTIR_A": "1", "AW_A": "1", "AW_B": "2", "CYTIR_B": "2", "PATH": "/bin"}
    assert sorted(added) == ["AW_A", "CYTIR_B"]


def test_equal_pair_is_fine_and_idempotent():
    env = {"CYTIR_A": "v", "AW_A": "v"}
    assert E.normalise(env) == []
    once = dict(env)
    E.normalise(env)
    E.normalise(env)
    assert env == once


def test_different_pair_raises_naming_both_without_values():
    env = {"CYTIR_TOKEN": "s3cr3t-one", "AW_TOKEN": "other-secret!"}
    with pytest.raises(E.EnvNameConflict) as exc:
        E.normalise(env)
    msg = str(exc.value)
    assert "CYTIR_TOKEN" in msg and "AW_TOKEN" in msg
    assert "s3cr3t" not in msg and "other-secret" not in msg
    assert str(len("s3cr3t-one")) in msg and str(len("other-secret!")) in msg


def test_conflict_changes_nothing_and_mirrored_never_touches_the_callers_dict():
    env = {"CYTIR_A": "1", "CYTIR_T": "x", "AW_T": "y"}
    before = dict(env)
    with pytest.raises(E.EnvNameConflict):
        E.normalise(env)
    assert env == before
    ok = {"CYTIR_A": "1"}
    out = E.mirrored(ok)
    assert ok == {"CYTIR_A": "1"} and _vals(out) == {"CYTIR_A": "1", "AW_A": "1"}


def test_empty_value_counts_as_set():
    env = {"CYTIR_A": ""}
    E.normalise(env)
    assert env["AW_A"] == ""


def test_shown_is_the_cytir_spelling():
    assert E.shown("AW_WORK_ROOT") == "CYTIR_WORK_ROOT" == E.shown("CYTIR_WORK_ROOT")
    assert E.is_knob("AW_X") and E.is_knob("CYTIR_X") and not E.is_knob("PATH")


@pytest.fixture
def clean_environ():
    """Restore `os.environ` exactly (the marker included), whatever a test did to it."""
    with mock.patch.dict(os.environ):
        yield


def test_live_environ_twin_made_by_the_mirror_follows_a_later_aw_change(clean_environ):
    os.environ["AW_STALE_T"] = "one"
    E.normalise(os.environ)
    assert os.environ["CYTIR_STALE_T"] == "one"
    os.environ["AW_STALE_T"] = "two"                 # a test / program changes the old name later
    E.normalise(os.environ)                           # no conflict: the twin was ours and stale
    assert os.environ["CYTIR_STALE_T"] == "two"
    del os.environ["AW_STALE_T"]
    E.normalise(os.environ)
    assert "CYTIR_STALE_T" not in os.environ and "AW_STALE_T" not in os.environ


def test_loaders_see_a_later_aw_change_without_touching_os_environ(clean_environ):
    os.environ["AW_STALE_U"] = "one"
    E.normalise(os.environ)
    os.environ["AW_STALE_U"] = "two"
    seen = E.mirrored(os.environ)
    assert seen["CYTIR_STALE_U"] == "two" and os.environ["CYTIR_STALE_U"] == "one"


def test_a_name_the_user_set_is_never_treated_as_stale(clean_environ):
    os.environ["AW_OWN_T"] = "a"
    E.normalise(os.environ)
    os.environ["CYTIR_OWN_T"] = "b"                   # the user takes the twin over
    with pytest.raises(E.EnvNameConflict):
        E.normalise(os.environ)


# -------------------------------------------------------------------------- 2. at package import
def _run(env_extra: dict) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if not E.is_knob(k)}
    base.update(env_extra)
    base["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-c", PROBE], env=base, capture_output=True, text=True,
                          timeout=120, cwd=str(ROOT))


def test_import_mirrors_cytir_into_aw_in_a_subprocess():
    cp = _run({"CYTIR_WORK_ROOT": "/x"})
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.strip() == "/x"


def test_import_refuses_a_conflicting_pair_naming_both():
    cp = _run({"CYTIR_WORK_ROOT": "/a", "AW_WORK_ROOT": "/b"})
    assert cp.returncode != 0
    assert "CYTIR_WORK_ROOT" in cp.stderr and "AW_WORK_ROOT" in cp.stderr
    assert "lengths 2 and 2" in cp.stderr


# ------------------------------------------------------------------------ 3. settings loaders
def test_load_settings_accepts_cytir_in_an_injected_env():
    via_aw = P.load_settings({"AW_WORK_ROOT": "/w"}, overlay=None)
    env = {"CYTIR_WORK_ROOT": "/w"}
    via_cytir = P.load_settings(env, overlay=None)
    assert via_cytir.work_root == via_aw.work_root == "/w"
    assert env == {"CYTIR_WORK_ROOT": "/w"}, "the caller's env must not be mutated"


def test_load_settings_aw_name_set_after_import_still_works(clean_environ):
    os.environ["AW_WORK_ROOT"] = "/early"
    E.normalise(os.environ)                           # as the import would have
    os.environ["AW_WORK_ROOT"] = "/late"              # what monkeypatch.setenv does in a test
    assert P.load_settings(overlay=None).work_root == "/late"


def test_load_settings_conflicting_pair_in_injected_env_is_refused():
    with pytest.raises(E.EnvNameConflict):
        P.load_settings({"CYTIR_WORK_ROOT": "/a", "AW_WORK_ROOT": "/b"}, overlay=None)


def test_refusal_text_tells_a_stranger_the_cytir_names():
    text = P._refusal_text(P._BY_NAME["work_root"])
    assert "CYTIR_WORK_ROOT" in text and not re.search(r"\bAW_[A-Z]", text)


def test_serve_settings_load_accepts_cytir_and_warns_alike_for_unknown_names():
    from cytir.serve import settings as S
    base = {"AW_QDRANT_URL": "http://q:6333"}
    want = S.Settings.load(dict(base))
    env = {"CYTIR_QDRANT_URL": "http://q:6333"}
    got = S.Settings.load(env)
    assert got.sources == want.sources and got.unknown_env == ()
    assert env == {"CYTIR_QDRANT_URL": "http://q:6333"}
    assert S.current("AW_QDRANT_URL", {"CYTIR_QDRANT_URL": "http://z:1"}) == "http://z:1"
    assert S.Settings.load({"AW_NOPE_X": "1"}).unknown_env == ("AW_NOPE_X",)
    assert S.Settings.load({"CYTIR_NOPE_X": "1"}).unknown_env == ("CYTIR_NOPE_X",)


def test_unknown_knob_is_reported_once_in_the_spelling_the_user_set(clean_environ):
    from cytir.serve import settings as S
    os.environ["AW_NOPE_ZZ"] = "1"
    E.normalise(os.environ)                           # as the import does: CYTIR_NOPE_ZZ is ours
    u = S.Settings.load().unknown_env
    assert "AW_NOPE_ZZ" in u and "CYTIR_NOPE_ZZ" not in u
    del os.environ["AW_NOPE_ZZ"], os.environ["CYTIR_NOPE_ZZ"]
    os.environ["CYTIR_NOPE_YY"] = "1"
    E.normalise(os.environ)
    u = S.Settings.load().unknown_env
    assert "CYTIR_NOPE_YY" in u and "AW_NOPE_YY" not in u


# ----------------------------------------------------- 3b. the marker: children inherit the record
CHILD = ("import cytir, os, sys; print(os.environ.get('AW_X'), os.environ.get('CYTIR_X'), "
         "'_CYTIR_ENV_MIRRORED' in os.environ)")


def _py(code: str, env: dict) -> subprocess.CompletedProcess:
    env = {**env, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=120, cwd=str(ROOT))


def _parent_env(**knobs) -> dict:
    """The environment of a process that imported `aw` with these knobs (real import, then copy)."""
    base = {k: v for k, v in os.environ.items() if not E.is_knob(k)}
    base.update(knobs)
    cp = _py("import cytir, os, json; print(json.dumps(dict(os.environ)))", base)
    assert cp.returncode == 0, cp.stderr[-800:]
    import json
    return json.loads(cp.stdout.strip().splitlines()[-1])


def test_child_with_overridden_aw_name_follows_it():                       # (a)
    parent = _parent_env(AW_X="1")
    assert parent["CYTIR_X"] == "1"
    cp = _py(CHILD, {**parent, "AW_X": "2"})
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.split()[:2] == ["2", "2"]


def test_child_with_overridden_cytir_name_follows_it():                    # (b)
    parent = _parent_env(CYTIR_X="1")                # the parent's user-set name is CYTIR_X
    assert parent["AW_X"] == "1"
    cp = _py(CHILD, {**parent, "CYTIR_X": "2"})
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.split()[:2] == ["2", "2"]


def test_overriding_the_twin_the_mirror_made_against_a_user_set_name_is_a_real_conflict():
    """Parent user-set AW_X=1 (CYTIR_X is the mirror's twin); a child that sets CYTIR_X=2 has two
    values a person chose that differ: refused, as with no record at all."""
    parent = _parent_env(AW_X="1")
    cp = _py(CHILD, {**parent, "CYTIR_X": "2"})
    assert cp.returncode != 0 and "AW_X" in cp.stderr and "CYTIR_X" in cp.stderr


def test_child_follows_a_change_the_parent_made_after_import(clean_environ):  # (c)
    os.environ["AW_X"] = "1"
    E.normalise(os.environ)
    os.environ["AW_X"] = "2"
    cp = _py(CHILD, dict(os.environ))
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.split()[:2] == ["2", "2"]


def test_two_user_set_names_that_differ_with_no_record_are_refused_without_values():  # (d)
    base = {k: v for k, v in os.environ.items() if not E.is_knob(k)}
    cp = _py(CHILD, {**base, "AW_X": "very-secret-1", "CYTIR_X": "very-secret-22"})
    assert cp.returncode != 0
    assert "AW_X" in cp.stderr and "CYTIR_X" in cp.stderr
    assert "very-secret" not in cp.stderr


def test_a_grandchild_follows_too():                                        # (e)
    parent = _parent_env(AW_X="1")
    code = ("import cytir, os, subprocess, sys; os.environ['AW_X'] = '3'; "
            f"print(subprocess.run([sys.executable, '-c', {CHILD!r}], env=dict(os.environ), "
            "capture_output=True, text=True).stdout)")
    cp = _py(code, {**parent, "AW_X": "2"})
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.split()[:2] == ["3", "3"]


def test_a_deleted_source_drops_its_recorded_twin_in_the_child():
    parent = _parent_env(AW_X="1")
    parent.pop("AW_X")
    cp = _py(CHILD, parent)
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.split() == ["None", "None", "False"]


def test_prod_env_run_with_an_ambient_differing_knob_does_not_die(clean_environ):   # (f)
    """The env `prod_env_run` builds is `os.environ` plus its own AW_QDRANT_URL: a child imports it."""
    os.environ["AW_QDRANT_URL"] = "http://ambient:1"
    E.normalise(os.environ)
    env = dict(os.environ)
    env["AW_QDRANT_URL"] = "http://new:2"
    cp = _py("import cytir, os; print(os.environ['CYTIR_QDRANT_URL'])", env)
    assert cp.returncode == 0, cp.stderr[-800:]
    assert cp.stdout.strip() == "http://new:2"


def test_the_marker_holds_hashes_never_values(clean_environ):             # (g)
    os.environ["AW_SECRET_KEY"] = "hunter2-very-secret"
    E.normalise(os.environ)
    m = os.environ[E.MARKER]
    assert "hunter2" not in m and "very-secret" not in m
    assert re.search(r"CYTIR_SECRET_KEY:[0-9a-f]{12}(,|$)", m)
    assert not E.is_knob("PATH") and E.is_knob(E.MARKER) and E.twin(E.MARKER) is None


def test_the_marker_is_stripped_with_the_knobs_by_the_arm_and_process_filters(clean_environ):
    from cytir.ops import serve as SV
    from cytir.retune import arm as RA
    os.environ["AW_X"] = "1"
    E.normalise(os.environ)
    assert E.MARKER in os.environ
    out = RA.build_env("w", "c", "http://q", "http://t", base=dict(os.environ))
    assert E.MARKER not in out and "CYTIR_X" not in out
    pe, dropped = SV.process_env({"AW_A": "1"}, ambient=dict(os.environ))
    assert E.MARKER not in pe and E.MARKER not in dropped


def test_pop_knob_removes_both_spellings_and_keeps_other_records():
    env = {"AW_A": "1", "AW_B": "2"}
    E.normalise(env)
    assert E.pop_knob(env, "AW_A") == "1"
    assert "AW_A" not in env and "CYTIR_A" not in env and env["CYTIR_B"] == "2"
    assert "CYTIR_B:" in env[E.MARKER] and "CYTIR_A" not in env[E.MARKER]


# ------------------------------------------------------ 3c. guards that ignore one named variable
def test_twin_guard_ignores_the_twin_variable_under_both_spellings(clean_environ, monkeypatch):
    from cytir.core import twin as TW
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    for k in [k for k in os.environ if "TWIN" in k]:
        del os.environ[k]
    want_url, want_named = TW.configured_url(), TW.url_is_named_by_overlay()
    os.environ["AW_TWIN_URL"] = "http://evil:1"
    E.normalise(os.environ)                            # as the import does: CYTIR_TWIN_URL appears
    assert TW.configured_url() == want_url != "http://evil:1"
    assert TW.url_is_named_by_overlay() == want_named
    os.environ.pop("AW_TWIN_URL")
    os.environ["CYTIR_TWIN_URL"] = "http://evil:2"     # a person using the new name is ignored alike
    assert TW.configured_url() == want_url


def test_roster_ci_operator_env_accepts_both_prefixes(tmp_path):
    """Operator-side tool: absent from the public cut, so the test skips there."""
    TR = pytest.importorskip("cytir.ops.test_suite_roster")
    ci = tmp_path / TR.CI_WORKFLOW_RELATIVE_TO_MONOREPO_ROOT
    ci.parent.mkdir(parents=True)
    ci.write_text("env:\n  AW_OPERATOR_URL: a\n  CYTIR_OPERATOR_CONTACT: b\n  AW_OTHER: c\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert TR.ci_operator_env(repo) == {"AW_OPERATOR_URL": "a", "CYTIR_OPERATOR_CONTACT": "b"}


# -------------------------------------------------------------------- 4. the prefix-filter sites
def test_parity_arm_env_drops_an_ambient_cytir_name():
    from cytir.gates import serve_parity as SP
    from cytir.retune import arm as RA
    ambient = {"CYTIR_STRAY": "1", "AW_STRAY": "1", "PATH": "/bin", "HOME": "/h"}
    out = RA.build_env("w", "c", "http://q", "http://t", base=ambient)
    assert "CYTIR_STRAY" not in out and "AW_STRAY" not in out and out["PATH"] == "/bin"
    assert SP._kv(["CYTIR_X=1"]) == {"CYTIR_X": "1"}


def test_aw_serve_process_env_drops_ambient_cytir_names():
    from cytir.ops import serve as SV
    out, dropped = SV.process_env({"AW_A": "1"}, ambient={"CYTIR_B": "1", "AW_B": "1", "CYTIR_A": "1",
                                                          "PATH": "/bin"})
    assert "CYTIR_B" not in out and "AW_B" not in out and "CYTIR_A" not in out
    assert "CYTIR_B" in dropped and "CYTIR_A" not in dropped


def test_prod_legs_env_file_keeps_both_prefixes(tmp_path):
    from cytir.gates import prod_legs as PL
    f = tmp_path / "e.env"
    f.write_text("AW_A=1\nCYTIR_B=2\nPATH=/bin\n")
    assert PL.load_env_file(str(f)) == {"AW_A": "1", "CYTIR_B": "2"}


def test_prod_env_run_lists_cytir_flags_like_aw_flags(monkeypatch, capsys):
    from cytir.ops import prod_env_run as PR
    monkeypatch.setattr(PR.subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setenv("CYTIR_STRAY_FLAG", "1")
    PR.main(["--collection", "c", "--qdrant-url", "http://q", "--mrl-dim", "128", "--", "true"])
    assert "AW_* flags" in capsys.readouterr().out


def test_no_package_module_filters_the_environment_on_the_aw_prefix_alone():
    """The settings probe, parity, arm, serve and prod_env sites all use `envnames.is_knob`; one
    grep keeps the next filter from being written the old way."""
    allowed = {("gates/prod_legs.py", 'AW_MONITOR_MAX_SMOKE_LATENCY=')}
    bad = []
    for p in (ROOT / "cytir").rglob("*.py"):
        rel = str(p.relative_to(ROOT / "cytir"))
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r'startswith\(\s*["\']AW_', line) and not any(
                    rel == r and a in line for r, a in allowed):
                bad.append(f"{rel}:{n}")
    assert bad == [], bad


# ------------------------------------------------------------------------------- 5. public docs
PUBLIC_DOCS = ("QUICKSTART.md", "README.md", "CONTRIBUTING.md", "ADDING-A-SOURCE.md")


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
def test_public_docs_name_no_aw_variable(doc):
    text = (ROOT / doc).read_text()
    assert not re.findall(r"\bAW_[A-Z]", text), doc


def test_every_cytir_name_in_the_public_docs_is_a_declared_name():
    """A documented name must be one the package reads (in any module, specs or direct reads)."""
    declared = set()
    for p in (ROOT / "cytir").rglob("*.py"):
        declared |= {E.shown(n) for n in re.findall(r"\bAW_[A-Z0-9_]*[A-Z0-9]", p.read_text())}
    seen = set()
    for doc in PUBLIC_DOCS:
        seen |= set(re.findall(r"\bCYTIR_[A-Z0-9_]*[A-Z0-9]", (ROOT / doc).read_text()))
    assert sorted(n for n in seen if n not in declared) == []


def test_quickstart_says_the_old_names_still_work():
    assert "AW_*" in (ROOT / "QUICKSTART.md").read_text()


def test_serve_parity_keeps_its_own_filter_in_step_with_envnames():
    """serve_parity is loaded by path against foreign trees, so it cannot import `envnames`."""
    from cytir.gates import serve_parity as SP
    assert SP._MIRROR_MARKER == E.MARKER and set(SP._KNOB_PREFIXES) == set(E.PREFIXES)
    for n in ("AW_X", "CYTIR_X", E.MARKER, "PATH", "HOME"):
        assert SP._is_knob(n) == E.is_knob(n)


def test_the_marker_never_shows_in_the_display_outputs(clean_environ, capsys, monkeypatch):
    """review 2965 follow-up: prod_env_run display, the config document and `ops serve --print-env`."""
    import json
    from cytir.ops import prod_env_run as PER
    from cytir.ops import serve as SV
    from cytir.serve import settings as SS
    os.environ["AW_X"] = "1"
    E.normalise(os.environ)
    assert E.MARKER in os.environ
    monkeypatch.setattr(PER.subprocess, "call", lambda *a, **k: 0)
    assert PER.main(["--collection", "c", "--qdrant-url", "http://q:1", "--mrl-dim", "256", "--", "true"]) == 0
    assert E.MARKER not in capsys.readouterr().out
    assert E.MARKER not in json.dumps(SS.config_document(), default=str)
    assert SV.run(["--collection", "c", "--print-env"]) == 0
    out = capsys.readouterr().out
    assert '"profile"' in out and E.MARKER not in out
