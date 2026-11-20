"""cyt ops guards --cut (task 2403): changed-module split, summary lines, call recording (no real export)."""
from cytir.ops import guards


def test_changed_test_modules_split(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    diff = ["tests/test_a.py", "tests/test_gone.py", "tests/helpers/repo_files.py", "cytir/ops/guards.py",
            "tests/data.json", "tests/test_a.py"]
    shipped, missing = guards.changed_test_modules(diff, tmp_path)
    assert shipped == ["tests/test_a.py"]
    assert missing == ["tests/test_gone.py"]


def test_module_statuses_and_line_format():
    report = "\n".join([
        "PASSED tests/test_a.py::t1", "SKIPPED [1] tests/test_a.py:9: no file",
        "SKIPPED [2] tests/test_b.py:3: withheld", "FAILED tests/test_c.py::t - boom",
        "ERROR tests/test_c.py - collection"])
    mods = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py", "tests/test_d.py"]
    st = guards.module_statuses(report, mods, 1)
    assert st == {"tests/test_a.py": "pass", "tests/test_b.py": "skipped-all",
                  "tests/test_c.py": "fail", "tests/test_d.py": "fail"}
    assert guards.summary_line("tests/test_a.py", "pass") == "guards: cut: tests/test_a.py: pass"


def test_run_cut_records_calls(tmp_path, capsys):
    export_dir = tmp_path / "x"
    calls = []

    def export(d):
        calls.append(("export", d))
        (d / "tests").mkdir(parents=True)
        (d / "tests" / "test_a.py").write_text("")
        return 0

    def run_tests(d, mods):
        calls.append(("run", d, list(mods)))
        return 0, "PASSED tests/test_a.py::t"

    rc = guards.run_cut("origin/main", export_dir, False,
                        diff_names=lambda base: ["tests/test_a.py", "tests/test_gone.py", "README.md"],
                        export=export, run_tests=run_tests)
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == [("export", export_dir), ("run", export_dir, ["tests/test_a.py"])]
    assert "tests/test_gone.py: not shipped" in out and "tests/test_a.py: pass" in out
    assert not export_dir.exists()


def test_run_cut_fail_and_keep(tmp_path):
    def export(d):
        (d / "tests").mkdir(parents=True)
        (d / "tests" / "test_a.py").write_text("")
        return 0

    rc = guards.run_cut("b", tmp_path / "y", True, diff_names=lambda b: ["tests/test_a.py"], export=export,
                        run_tests=lambda d, m: (1, "FAILED tests/test_a.py::t"))
    assert rc == 1 and (tmp_path / "y").exists()


# ---- task 2436: slot semaphore for small runs ----

def test_slot_count_sizing():
    assert guards.slot_count(5.0) == 3        # floor(5 * 0.5 / 0.7)
    assert guards.slot_count(100.0) == 4      # capped
    assert guards.slot_count(0.5) == 1        # never below one


def _paths(tmp_path):
    return str(tmp_path / "slot-{i}.lock"), str(tmp_path / "suite.lock")


def _take(tmp_path, n, wait_s=1):
    pattern, suite = _paths(tmp_path)
    return guards.acquire_slot(n, wait_s=wait_s, pattern=pattern, poll_s=0.05, suite_lock=suite)


def test_two_small_runs_hold_different_slots(tmp_path):
    a = _take(tmp_path, 2)
    b = _take(tmp_path, 2)
    assert a is not None and b is not None
    assert _take(tmp_path, 2, wait_s=0.2) is None      # slots exhausted: the third run waits, then gives up
    a.close()
    c = _take(tmp_path, 2)
    assert c is not None
    b.close()
    c.close()


def test_suite_lock_is_exclusive_against_slots(tmp_path):
    import fcntl
    _, suite = _paths(tmp_path)
    with open(suite, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)         # a suite-wide run holds the exclusive lock
        assert _take(tmp_path, 2, wait_s=0.2) is None
    held = _take(tmp_path, 2)
    assert held is not None
    with open(suite, "a") as fh:               # a small run holds it shared: the suite form must wait
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got = True
        except OSError:
            got = False
    assert not got
    held.close()


def test_run_in_slot_returns_none_without_a_slot(monkeypatch):
    monkeypatch.setattr(guards, "acquire_slot", lambda *a, **k: None)
    assert guards.run_in_slot(["true"]) is None


def test_run_in_slot_releases_after_run(tmp_path, monkeypatch):
    pattern, suite = _paths(tmp_path)
    real = guards.acquire_slot
    monkeypatch.setattr(guards, "acquire_slot",
                        lambda: real(1, wait_s=1, pattern=pattern, poll_s=0.05, suite_lock=suite))
    assert guards.run_in_slot(["true"]).returncode == 0
    assert real(1, wait_s=0.5, pattern=pattern, poll_s=0.05, suite_lock=suite) is not None


def test_run_cut_names_the_missing_tokenizer_before_and_after_a_red(tmp_path, capsys):
    def export(d):
        (d / "tests").mkdir(parents=True)
        (d / "tests" / "test_a.py").write_text("")
        return 0

    rc = guards.run_cut("b", tmp_path / "z", False, diff_names=lambda b: ["tests/test_a.py"], export=export,
                        run_tests=lambda d, m: (1, "FAILED tests/test_a.py::t"), tokenizer_cached=lambda: False)
    out = capsys.readouterr().out
    assert rc == 1 and out.count("fetch_models --tokenizer") == 2
    rc = guards.run_cut("b", tmp_path / "z", False, diff_names=lambda b: ["tests/test_a.py"], export=export,
                        run_tests=lambda d, m: (0, "PASSED tests/test_a.py::t"), tokenizer_cached=lambda: True)
    assert rc == 0 and "fetch_models" not in capsys.readouterr().out


def _main_cut(monkeypatch, argv, *, rc=0):
    """Run the real `main` with no guard steps; record the export dir `run_cut` receives."""
    seen = []

    def fake_run_cut(base, export_dir, keep, **kw):
        seen.append((export_dir, export_dir.is_dir()))
        return rc

    monkeypatch.setattr(guards, "_STEPS", ())
    monkeypatch.setattr(guards, "run_cut", fake_run_cut)
    assert guards.main(argv) == rc
    return seen


def test_default_export_dirs_are_fresh_per_run(monkeypatch):
    a = _main_cut(monkeypatch, ["--cut"])[0]
    b = _main_cut(monkeypatch, ["--cut"])[0]
    try:
        assert a[0] != b[0]
        assert a[1] and b[1]
        assert a[0].name.startswith(guards._EXPORT_PREFIX)
    finally:
        for d in (a[0], b[0]):
            d.rmdir()


def test_explicit_export_dir_is_used_as_given(monkeypatch, tmp_path):
    seen = _main_cut(monkeypatch, ["--cut", "--export-dir", str(tmp_path / "mine")])
    assert seen[0][0] == tmp_path / "mine"


def test_per_run_dir_is_gone_after_a_failing_cut(monkeypatch):
    made = []
    real = guards.default_export_dir

    def spy():
        d = real()
        made.append(d)
        return d

    def export(d):
        d.mkdir(parents=True, exist_ok=True)
        return 1

    real_cut = guards.run_cut
    monkeypatch.setattr(guards, "default_export_dir", spy)
    monkeypatch.setattr(guards, "_STEPS", ())
    monkeypatch.setattr(guards, "run_cut",
                        lambda base, d, keep, **kw: real_cut(base, d, keep, diff_names=lambda b: [], export=export))
    assert guards.main(["--cut"]) == 1
    assert len(made) == 1 and not made[0].exists()
