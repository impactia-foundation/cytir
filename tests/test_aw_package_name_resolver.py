"""Which package a source tree carries, and the parity gate across the package rename (task 2967).

The first release after the `aw` -> `cytir` rename compares an OLD base tree (package dir `cytir/`) with
a NEW head tree (`cytir/` plus a thin `cytir/` import shim). The gate and the release tools ask each tree
for its own package (`serve_layout.package_name`); `serve_parity` holds an inline copy of the resolver
(it is loaded by path inside an arm of an old tree) and these tests pin the two equal.
"""
from __future__ import annotations

import pathlib
import shutil

import pytest

from tests.helpers.repo_files import repo_file_or_skip
import test_aw_serve_parity as TP
from cytir.core import serve_layout as SL
from cytir.gates import serve_parity as sp

NEW, OLD = SL.PACKAGE_NAMES

#: First line of the import shim the rename writes (the marker is what the resolver relies on).
SHIM_TEXT = (f'"""{OLD} -- the OLD import name of the `{NEW}` package. {SL.SHIM_MARKER}\n\n'
             f'`import {OLD}.x.y` runs the `{NEW}.x.y` module itself."""\nimport {NEW}\n')

RESOLVERS = [pytest.param(SL.package_name, id="shared"), pytest.param(sp.package_name, id="inline")]


def _pkg(tree: pathlib.Path, name: str, init: str = "") -> pathlib.Path:
    d = tree / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text(init)
    return d


def _tree(tmp_path, kind: str) -> pathlib.Path:
    t = tmp_path / kind
    t.mkdir()
    if kind == "old":
        _pkg(t, OLD, "")
    elif kind == "new":
        _pkg(t, NEW)
    elif kind == "new+shim":
        _pkg(t, NEW)
        _pkg(t, OLD, SHIM_TEXT)
    elif kind == "shim-only":
        _pkg(t, OLD, SHIM_TEXT)
    return t


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_the_resolver_on_four_trees(tmp_path, resolve):
    assert resolve(_tree(tmp_path, "old")) == OLD
    assert resolve(_tree(tmp_path, "new")) == NEW
    assert resolve(_tree(tmp_path, "new+shim")) == NEW           # the shim never wins over the package
    with pytest.raises(SystemExit, match="no serving package"):
        resolve(_tree(tmp_path, "neither"))                        # an empty tree: refused, not guessed


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_a_shim_alone_and_a_dir_without_init_are_not_packages(tmp_path, resolve):
    with pytest.raises(SystemExit, match="no serving package"):
        resolve(_tree(tmp_path, "shim-only"))
    t = tmp_path / "noinit"
    (t / NEW).mkdir(parents=True)                                  # a directory, not a package
    _pkg(t, OLD)
    assert resolve(t) == OLD
    with pytest.raises(SystemExit):
        resolve(tmp_path / "does-not-exist")


def test_the_marker_is_read_from_the_head_of_the_file_only(tmp_path):
    t = tmp_path / "late"
    _pkg(t, OLD, "x = 1\n" * 200 + f"# {SL.SHIM_MARKER}\n")           # far past the head: a real package
    assert SL.package_name(t) == sp.package_name(t) == OLD


def test_the_inline_copy_is_pinned_to_the_shared_one():
    assert sp.PACKAGE_NAMES == SL.PACKAGE_NAMES
    assert sp.SHIM_MARKER == SL.SHIM_MARKER
    assert sp.SHIM_HEAD_BYTES == SL.SHIM_HEAD_BYTES
    assert sp.IMAGE_SERVE_PREFIXES == tuple(f"/app/{p}/serve/" for p in SL.PACKAGE_NAMES)


def test_serve_dir_and_package_follow_the_tree(tmp_path):
    old, new = _tree(tmp_path, "old"), _tree(tmp_path, "new+shim")
    assert sp.serve_dir(old) == old / OLD / "serve" and sp.serve_package(old) == f"{OLD}.serve"
    assert sp.serve_dir(new) == new / NEW / "serve" and sp.serve_package(new) == f"{NEW}.serve"
    assert SL.serve_rel(new) == f"{NEW}/serve" and SL.serve_rel(old) == f"{OLD}/serve"


# --- the gate across the rename ---------------------------------------------------------------- #

def _old_tree(tmp_path, name: str, **kw) -> pathlib.Path:
    """A fake server tree as it stands BEFORE the rename: the package dir carries the OLD name (the codemod
    turns the harness's own package dir into the new one, so this moves it back)."""
    tree = TP.make_tree(tmp_path, name, **kw)
    shutil.move(str(tree / NEW), str(tree / OLD))
    return tree


def _renamed(tmp_path, name: str, **kw) -> pathlib.Path:
    """A fake server tree of the harness (test_aw_serve_parity) as the rename leaves it: the package
    dir is `cytir`, and a one-file `aw` shim sits beside it."""
    tree = TP.make_tree(tmp_path, name, **kw)
    shutil.move(str(tree / "cytir"), str(tree / NEW))
    _pkg(tree, OLD, SHIM_TEXT)
    return tree


def test_an_old_base_and_a_renamed_head_compare_and_pass(tmp_path):
    base = _old_tree(tmp_path, "base")                          # package dir `aw`
    head = _renamed(tmp_path, "head")                              # `cytir/` + the shim
    rep = TP.run(tmp_path, base, head)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    assert rep["counts"]["EQUAL"] == len(TP.QUERIES)               # every request compared, none skipped
    assert rep["null_pair"] == {"first_attempt_unstable": 0, "unstable_after_retries": 0}
    assert rep["fingerprint_diffs"] == []
    assert rep["config_change"] is None
    # each arm ran ITS tree's modules: the report names both trees and their server code is the same bytes
    assert rep["trees"]["base"] == str(base) and rep["trees"]["head"] == str(head)
    assert rep["trees"]["same_server_code"] is True


def test_a_ranking_change_behind_the_rename_still_fails(tmp_path):
    base = _old_tree(tmp_path, "base")
    head = _renamed(tmp_path, "head", tweak="swap")
    rep = TP.run(tmp_path, base, head, deep=False)
    assert rep["verdict"] == "FAIL"                                # the comparison is real, not skipped
    assert [r["id"] for r in rep["not_equal"]] == ["t:2"]
    assert rep["trees"]["same_server_code"] is False


def test_a_renamed_head_without_its_package_modules_is_refused_not_served_from_the_base(tmp_path):
    base = _old_tree(tmp_path, "base")
    head = _renamed(tmp_path, "head")
    shutil.rmtree(head / NEW / "serve")
    with pytest.raises(Exception):                                 # the arm cannot start: no fallback
        TP.run(tmp_path, base, head, deep=False)


def _img(pkg: str, name: str = "x.json") -> str:
    return f"/app/{pkg}/serve/{name}"


def test_env_values_naming_the_image_package_compare_by_file_not_by_package_name():
    old = {"AW_FUSION_WEIGHTS": _img(OLD, "w.json"), "AW_X": "1"}
    new = {"AW_FUSION_WEIGHTS": _img(NEW, "w.json"), "AW_X": "1"}
    assert sp.env_changes(old, new, OLD, NEW) == {}                # (a) the same file, each tree's own spelling
    other = dict(new, AW_FUSION_WEIGHTS=_img(NEW, "w2.json"))
    assert list(sp.env_changes(old, other, OLD, NEW)) == ["AW_FUSION_WEIGHTS"]   # another file is a change
    assert list(sp.env_changes(old, dict(new, AW_X="2"), OLD, NEW)) == ["AW_X"]
    assert list(sp.env_changes(old, {**new, "AW_FUSION_WEIGHTS": "/elsewhere/w.json"}, OLD, NEW)) == [
        "AW_FUSION_WEIGHTS"]
    # values that merely contain the old name are untouched
    same = {"AW_COLLECTION": "aw_e10", "AW_HOST": "https://aw.example/awserve"}
    assert sp.env_changes(same, dict(same), OLD, NEW) == {}


def test_only_the_prefix_of_each_sides_own_package_is_canonicalised():
    old, new = {"X": _img(OLD)}, {"X": _img(NEW)}
    # (b) single layout: both trees carry the old package, the head value is respelled -> a change
    assert list(sp.env_changes(old, new, OLD, OLD)) == ["X"]
    # (c) a renamed head whose value still names the old package -> a change (not mapped onto its own tree)
    assert list(sp.env_changes(old, old, OLD, NEW)) == ["X"]
    # (d) the reverse: an old base whose value names the new package
    assert list(sp.env_changes(new, new, OLD, NEW)) == ["X"]
    # (e) the two calls the review quoted no longer return {} unless the trees' packages justify it
    assert sp.env_changes(old, new) != {} and sp.env_changes(new, old) != {}
    assert sp.env_changes(old, new, OLD, NEW) == {} and sp.env_changes(new, old, NEW, OLD) == {}
    assert sp.env_changes(old, new, NEW, OLD) != {}                # the packages swapped: not justified


def test_arm_env_refuses_a_value_with_the_other_packages_image_prefix(tmp_path):
    old, new = _tree(tmp_path, "old"), _tree(tmp_path, "new+shim")
    for t, pkg in ((old, OLD), (new, NEW)):
        (t / pkg / "serve").mkdir()
        (t / pkg / "serve" / "x.json").write_text("{}")
    kw = dict(collection="c", qdrant_url="http://x", deep=False, base={})
    # own spelling maps into the tree; a spelling from another tree maps only when it is declared as such
    env, files = sp.arm_env({"AW_W": _img(NEW)}, new, **kw)
    assert env["AW_W"] == str(new / NEW / "serve" / "x.json") and "AW_W" in files
    env, _ = sp.arm_env({"AW_W": _img(OLD)}, new, declared_pkg=OLD, **kw)   # the base env on the head tree
    assert env["AW_W"] == str(new / NEW / "serve" / "x.json")
    # (c) a renamed head whose value still says the old package: refused, naming var, prefix and package
    with pytest.raises(SystemExit, match=rf"AW_W=.*{OLD}/serve/.*prefix /app/{OLD}/serve/.*package {NEW}/"):
        sp.arm_env({"AW_W": _img(OLD)}, new, **kw)
    # (d) the reverse, for an old base
    with pytest.raises(SystemExit, match=rf"AW_W=.*prefix /app/{NEW}/serve/.*package {OLD}/"):
        sp.arm_env({"AW_W": _img(NEW)}, old, **kw)
    (new / NEW / "serve" / "x.json").unlink()
    with pytest.raises(SystemExit, match="does not exist"):        # a missing file is refused, never skipped
        sp.arm_env({"AW_W": _img(NEW)}, new, **kw)


def _with_x(tree: pathlib.Path, env_pkg: str, body: str = "{}") -> pathlib.Path:
    """A compose that declares AW_X_FILE under `/app/<env_pkg>/serve/`, and the file in the tree."""
    (sp.serve_dir(tree) / "x.json").write_text(body)
    return TP.with_compose(tree, {"AW_FAKE_KNOB": "1", "AW_X_FILE": _img(env_pkg)})


def _run_each_trees_env(tmp_path, base, head):
    return sp.run_parity(TP.requests(), base_tree=base, head_tree=head, collection="twin_x",
                         qdrant_url=TP.free_url(), vec_cache=TP.write_cache(tmp_path / "v.json"),
                         out_dir=tmp_path / "out", shards=1, deep=False,
                         guard=lambda url, coll: None, say=lambda m: None)   # no declared_env: each tree's own


def test_run_parity_with_each_trees_own_env_spelling_passes_across_the_rename(tmp_path):
    base = _with_x(_old_tree(tmp_path, "base"), OLD)             # compose says /app/<old>/serve/x.json
    head = _with_x(_renamed(tmp_path, "head"), NEW)                 # compose says /app/<new>/serve/x.json
    rep = _run_each_trees_env(tmp_path, base, head)
    assert rep["verdict"] == "PASS", sp.summary(rep)
    assert rep["config_change"] is None                             # the same file: no declared-env change
    assert rep["data_file_diffs"] == {}


def test_run_parity_fails_when_the_same_named_file_has_other_bytes_in_the_renamed_tree(tmp_path):
    base = _with_x(_old_tree(tmp_path, "base"), OLD, "{}")
    head = _with_x(_renamed(tmp_path, "head"), NEW, '{"a": 1}')
    rep = _run_each_trees_env(tmp_path, base, head)
    assert rep["verdict"] == "FAIL" and list(rep["data_file_diffs"]) == ["AW_X_FILE"]


def test_run_parity_refuses_a_renamed_head_compose_that_kept_the_old_image_path(tmp_path):
    base = _with_x(_old_tree(tmp_path, "base"), OLD)
    head = _with_x(_renamed(tmp_path, "head"), OLD)                                # (c) stale spelling
    with pytest.raises(SystemExit, match=rf"AW_X_FILE=.*prefix /app/{OLD}/serve/.*package {NEW}/"):
        _run_each_trees_env(tmp_path, base, head)
    assert not (tmp_path / "out" / "parity.json").exists()
    base2 = _with_x(_old_tree(tmp_path, "base2"), NEW)                          # (d) the reverse, a base
    head2 = _with_x(_renamed(tmp_path, "head2"), NEW)
    with pytest.raises(SystemExit, match=rf"prefix /app/{NEW}/serve/.*package {OLD}/"):
        _run_each_trees_env(tmp_path, base2, head2)
    base3, head3 = _with_x(_old_tree(tmp_path, "base3"), OLD), _with_x(_old_tree(tmp_path, "head3"), NEW)
    with pytest.raises(SystemExit, match="stale package spelling"):               # (b) single layout, respelled
        _run_each_trees_env(tmp_path, base3, head3)


def test_a_tree_with_a_real_old_package_beside_the_new_one_is_refused(tmp_path):
    for i, resolve in enumerate((SL.package_name, sp.package_name)):
        t = tmp_path / f"half{i}"
        _pkg(t, NEW)
        _pkg(t, OLD, "x = 1\n")                                      # a stale old dir that is not the shim
        with pytest.raises(SystemExit, match=rf"two real packages.*{NEW}.*{OLD}"):
            resolve(t)


def test_the_inline_resolver_is_source_equal_to_the_shared_one():
    import inspect
    for name in ("_is_shim", "package_name"):
        assert inspect.getsource(getattr(sp, name)) == inspect.getsource(getattr(SL, name)), name


def _inputs(tmp_path, name: str, pkg: str, body: str, extra: str = "") -> pathlib.Path:
    t = tmp_path / name
    _pkg(t, pkg)
    (t / pkg / "core").mkdir()
    (t / pkg / "core" / "m.py").write_text(body)
    (t / "data").mkdir()
    (t / "data" / "a.json").write_text("1")
    if pkg == NEW:
        _pkg(t, OLD, SHIM_TEXT)
    if extra:
        (t / pkg / "core" / "extra.py").write_text(extra)
    return t


def test_moved_inputs_pair_the_files_of_the_two_package_names(tmp_path):
    base = _inputs(tmp_path, "b", OLD, "X = 1\n")
    same = _inputs(tmp_path, "s", NEW, "X = 1\n")
    edited = _inputs(tmp_path, "e", NEW, "X = 2\n", extra="Y = 1\n")
    assert sp.moved_inputs(base, same) == []                       # the same bytes under another dir name
    assert sp.moved_inputs(base, edited) == [f"{NEW}/core/extra.py", f"{NEW}/core/m.py"]
    assert sp._value_change_blocked(f"{NEW}/core/m.py", NEW) is False
    assert sp._value_change_blocked(f"{NEW}/serve/app.py", NEW) is True
    assert sp._value_change_blocked("data/a.json", NEW) is True


def test_the_digest_never_hides_a_content_difference_behind_the_rename(tmp_path):
    base = _inputs(tmp_path, "b", OLD, "X = 1\n")
    renamed = _inputs(tmp_path, "r", NEW, "X = 1\n")
    edited = _inputs(tmp_path, "e", NEW, "X = 2\n")
    d_old, d_new, d_edit = (sp.serving_inputs_digest(t) for t in (base, renamed, edited))
    # Documented behaviour: the digest hashes relative paths, so the package dir name is part of it. A
    # pure rename therefore moves it (and the codemod's import edits move the bytes anyway); the
    # rename release is judged on the served answers, not on this digest. Every edit moves it too.
    assert d_old != d_new and d_new != d_edit and d_old != d_edit
    assert sp.serving_inputs_digest(renamed) == d_new               # stable
    # the shim beside the package is not a serving input of the head: it has no code of its own
    (renamed / OLD / "__init__.py").write_text(SHIM_TEXT + "# edited\n")
    assert sp.serving_inputs_digest(renamed) == d_new


def test_the_news_daily_gate_reads_the_release_trees_own_runner(tmp_path):
    from cytir.gates import news_daily
    for kind, pkg in (("old", OLD), ("new+shim", NEW)):
        t = _tree(tmp_path, kind)
        (t / pkg / "gates").mkdir()
        (t / pkg / "gates" / "run_arm.py").write_text("PINNED_ENV = {'AW_A': '1'}\n")
        assert news_daily.pinned_env(t)["AW_A"] == "1"
    with pytest.raises(SystemExit, match="no serving package"):
        news_daily.pinned_env(_tree(tmp_path, "neither"))


# --- the shell door of the gate (serving-gate.sh) ------------------------------------------------ #

APP = pathlib.Path(__file__).resolve().parents[1]


def _gate(tmp_path, head_pkg: str, box_pkg: str, head_src: str, box_src: str, stale_src: str | None = None):
    """`serving_gate --only-if-changed` on a git repo whose release carries `head_pkg/serve` and a box
    directory serving `box_pkg/serve`; ssh runs locally. `stale_src` = a SECOND serving dir on the box,
    under the other package name (a copy left without delete). Skips by name in the public cut, where
    deploy/scripts/serving-gate.sh is not shipped (looked up here, not at import, so the other tests run)."""
    import os
    import subprocess
    import sys
    GATE = repo_file_or_skip("deploy/scripts/serving-gate.sh")
    tmp_path.mkdir()
    repo, box, bin_dir = tmp_path / "repo", tmp_path / "box", tmp_path / "bin"
    for d in (repo, box, bin_dir):
        d.mkdir()
    for root, pkg, src in ((repo, head_pkg, head_src), (box, box_pkg, box_src)):
        (root / pkg / "serve").mkdir(parents=True)
        (root / pkg / "__init__.py").write_text("")
        (root / pkg / "serve" / "app.py").write_text(src)
    if head_pkg == NEW:
        _pkg(repo, OLD, SHIM_TEXT)                                 # the shim rides along in the release
    if stale_src is not None:
        other = OLD if box_pkg == NEW else NEW
        (box / other / "serve").mkdir(parents=True)
        (box / other / "serve" / "app.py").write_text(stale_src)
    (box / ".serving-release").write_text("a" * 40 + "\n")
    (bin_dir / "ssh").write_text('if [ "$1" = "-p" ]; then shift 2; fi\nshift\nexec bash -c "$*"\n')
    (bin_dir / "ssh").chmod(0o755)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "r"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "AW_SERVE_PARITY_PYTHON": sys.executable}
    return subprocess.run(["bash", "-c", f'. "{GATE}"; serving_gate box "{box}" "{repo}" "" {sha} --only-if-changed'],
                          cwd=APP, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("head_pkg,box_pkg", [(NEW, OLD), (OLD, OLD), (NEW, NEW)],
                         ids=["renamed-release-old-box", "old-both", "renamed-both"])
def test_the_shell_gate_finds_each_trees_own_package_dir(tmp_path, head_pkg, box_pkg):
    same = _gate(tmp_path / "same", head_pkg, box_pkg, "V = 1\n", "V = 1\n")
    assert same.returncode == 0, same.stdout + same.stderr         # the same bytes: both dirs were found
    assert "no serving-code change" in same.stdout
    diff = _gate(tmp_path / "diff", head_pkg, box_pkg, "V = 2\n", "V = 1\n")
    assert diff.returncode == 1                                    # a difference is never skipped
    assert "AW_PARITY_REPORT is not set" in diff.stderr


@pytest.mark.parametrize("stale_src,what", [("V = 1\n", "stale-equal-to-the-head"), ("V = 0\n", "stale-other")])
def test_the_shell_gate_refuses_a_box_with_two_serving_dirs(tmp_path, stale_src, what):
    # the box serves the old dir (V = 1); a stale `cytir/serve` equal to the release (or not) sits beside
    # it. Guessing the stale one would say "identical, no report needed" while the box serves the other.
    r = _gate(tmp_path / what, NEW, OLD, "V = 1\n", "V = 9\n", stale_src=stale_src)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "two serving dirs" in r.stderr and f"/{OLD}/serve" in r.stderr and f"/{NEW}/serve" in r.stderr
    assert "no serving-code change" not in r.stdout
