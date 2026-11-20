"""The old name `aw` is gone as a Python package and as a command (rename step 4, task 2982).

This test ships in the public cut and runs there. It fails when a shipped file still CALLS the old name:

  * `python -m aw.x`, `python -m aw`   (the old module path; any whitespace after `-m`; the argv-list form `"-m", "aw.x"`)
  * `importlib.import_module("aw.x")`  (the old module path as a string)
  * `["aw", ...]` as argv[0], `which("aw")`  (the old command as a list element or a lookup)
  * `import aw`, `from aw.x import y`  (the old import name)
  * `bin/aw`                           (the old console script)
  * `aw ops ...`, `aw index ...`       (the old command, any verb the CLI has)
  * `aw --version`, `aw `              (a back-quoted old command in a comment, docstring or markdown; task 2987)

What stays under the old name on purpose is NOT a caller and does not match: environment names `AW_*`, unit and timer
names `aw-*`, collection names `aw_*`, box paths (`/opt/aw`, `/etc/aw`), container and image names, schema ids
`aw.*.v1`, producer labels written into sealed records. The few lines that do match and must stay are listed in
`ALLOWED_LINE` and `ALLOWED_FILE` below, each with its reason. A new allowance is a decision, not a convenience.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]

# Every verb the `cyt` CLI has (cytir/cli.py). `aw <verb>` is the old command; `aw` followed by another word is prose.
VERBS = ("dump|apply|rehearse|torrent|source|embed|serve|read|ops|consent|news|chunk|index|gate|gates|corpus|"
         "extract|retune|bench|build|probe|sources|distribute")
CALLER = re.compile(
    r"(?<![\w.])-m\s+aw(?![\w-])"                  # python -m aw / python -m aw.x.y (any whitespace)
    r"|[\"']-m[\"']\s*,\s*[\"']aw(?![\w-])"            # argv list: ["python", "-m", "aw.x"]
    r"|import_module\(\s*[\"']aw(?![\w-])"            # importlib.import_module("aw.x")
    r"|\[\s*[\"']aw[\"']\s*,"                          # argv[0]: ["aw", "ops", ...]
    r"|\bwhich\(\s*[\"']aw[\"']"                      # shutil.which("aw")
    r"|\bimport aw\b(?![-_])"                      # import aw, import aw.x
    r"|\bfrom aw(?=[. ])"                          # from aw.x import y
    r"|bin/aw\b(?![-\w])"                          # the old console script
    rf"|(?<![\w./-])aw (?:{VERBS})\b"              # the old command and a verb
    r"|(?<![\w./-])aw \{\w+\}"                     # an f-string usage name: f"aw {group} {verb}"
    r"|usage: aw(?![-\w])"                                # a usage line that names the old command
    r"|[\"']aw [\"'] ?\+"                         # "aw " + " ".join(...)
    r"|`aw (?=[-\w])"                              # prose: `aw --version`, `aw source run` (a quoted old command)
    r"|`aw\s*[\"']?\s*$"                              # a quoted old command split at the line end: `aw "  /  "apply`
)

SUFFIXES = {".py", ".sh", ".service", ".timer", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".md", ".snippet"}
# History and frozen evidence are never scanned: the operator's task records, notes, the plan and the logs, the
# captured layouts of old releases, the retired specs, the probe run records and the site.
SKIP_DIRS = {"tasks", "notes", "archive", "probes", "presentation", "webapp", "node_modules", "__pycache__",
             ".git", ".pytest_cache", "news_layout1_shims", "news_release_layout", "dist", "build"}
SKIP_NAMES = {"log.md", "plan.md", "decisions.md", "spec.md", "NEXT.md", "RUNBOOK-v2.md"}

#: Files that name the old name as their subject.
ALLOWED_FILE = {
    "tests/test_cytir_no_old_name.py": "this guard names every pattern it forbids",
    "specs/cytir-code-rename.md": "the plan of the rename itself: it quotes the old forms it removes",
    "cytir/gates/authority.py": "sealed by digest (record b02 binds its sha256); a docstring edit would need a new seal",
}
#: Lines that match and stay, by a marker or shape in the line.
ALLOWED_LINE = (
    ("rename-codemod: keep", "an old tree, an old sealed receipt or a frozen release still has the old name"),
    ('"producer"', "producer label recorded in sealed artifacts (the labels are frozen)"),
    ('"verifier"', "verifier label recorded in sealed artifacts (the labels are frozen)"),
    ('put("AW_', "`source` label inside the sealed serve receipt (serve-receipt.json)"),
)


def _is_shebang_script(p: Path) -> bool:
    """An extension-less file that starts with `#!` is a shipped script (bin/, deploy/): scan it like a `.sh`."""
    if p.suffix:
        return False
    try:
        with p.open("rb") as fh:
            return fh.read(2) == b"#!"
    except OSError:
        return False


def _files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES or (p.suffix not in SUFFIXES and not _is_shebang_script(p)):
            continue
        rel = p.relative_to(root)
        if SKIP_DIRS.intersection(rel.parts[:-1]):
            continue
        yield p, rel.as_posix()


def scan_text(text: str) -> list[tuple[int, str]]:
    """(line number, line) of every live old-name caller in `text`, allowances applied."""
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if CALLER.search(line) and not any(marker in line for marker, _ in ALLOWED_LINE):
            hits.append((n, line.strip()[:160]))
    return hits


def _roots() -> list[Path]:
    roots = [APP]
    mono = APP.parent / ".github"                   # the full tree keeps the hosted CI at the repository root
    if APP.name == "authoritative-web-app" and mono.is_dir():
        roots.append(mono.parent)
    return roots


def test_no_shipped_file_calls_the_old_name():
    offenders = []
    for root in _roots():
        scan_root = root if root == APP else root / ".github"
        for p, rel in _files(scan_root):
            rel = rel if root == APP else f".github/{rel}"
            if rel in ALLOWED_FILE:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            offenders += [f"{rel}:{n}: {line}" for n, line in scan_text(text)]
    assert not offenders, ("the old name `aw` is called again (use `cytir` / `cyt`; see this module's header for what "
                           "may stay):\n" + "\n".join(offenders[:40]) + (f"\n... {len(offenders)} in all" if len(offenders) > 40 else ""))


@pytest.mark.parametrize("line", [
    "python -m aw.cli ops guards",
    "python3 -m aw.index.promote_queue --help",
    "/usr/bin/python3 -m aw",
    "import aw",
    "import aw.core.common as c",
    "from aw.core.common import Passage",
    "~/.local/bin/aw --version",
    "run `aw ops guards` before you push",
    "see `aw --version` for the data dir",
    '"refusal fired; `aw "',
    "aw index build --pack x",
    "    aw serve --collection c",
    'prog=f"aw {group} {verb}"',
    '"aw " + " ".join(door)',
    'startswith("usage: aw index apply")',
    "python3 -m  aw.cli ops guards",
    "python3 -m\taw.cli ops guards",
    'subprocess.run(["python3", "-m", "aw.index.apply"])',
    'subprocess.run(["python3", "-m", "aw"])',
    'mod = importlib.import_module("aw.core.common")',
    "mod = importlib.import_module('aw.core.common')",
    'subprocess.run(["aw", "ops", "guards"])',
    'cmd = [ "aw", "serve"]',
    'shutil.which("aw")',
])
def test_the_scanner_catches_each_old_form(line):
    assert scan_text(line), line


@pytest.mark.parametrize("line", [
    "python -m cytir.cli ops guards",
    "import cytir",
    "cyt index build",
    "export AW_QDRANT_URL=http://127.0.0.1:6333",
    "systemctl status aw-promotion-nightly",
    "collection aw_e10 on /opt/aw/current and /etc/aw/operator.json",
    "the schema aw.dump-manifest.v1 and the producer id aw-bm25-v1",
    "docker compose up aw-api; image aw-search-api:0.1",
    "awk '{print $1}' | aw-run",
    "from awesome import x",
    "an aw  index (prose with two spaces)",
    "a `cyt --version` line and `aw` alone as a word",
    'subprocess.run(["python3", "-m", "cytir.cli", "ops", "guards"])',
    'importlib.import_module("cytir.core.common")',
    'subprocess.run(["cyt", "ops", "guards"])',
    'shutil.which("cyt")',
    'run(["awk", "-F:", "x"])',
    'which("aws")',
])
def test_the_scanner_leaves_the_names_that_stay(line):
    assert not scan_text(line), line


def test_each_allowance_has_a_reason():
    assert all(reason for _, reason in ALLOWED_LINE) and all(ALLOWED_FILE.values())


def test_a_shebang_script_without_an_extension_is_scanned(tmp_path):
    script = tmp_path / "bin" / "tool"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\npython3 -m aw.cli ops guards\n")
    plain = tmp_path / "bin" / "data"
    plain.write_text("python3 -m aw.cli ops guards\n")        # no shebang, no suffix: not a script
    found = [rel for _p, rel in _files(tmp_path)]
    assert "bin/tool" in found and "bin/data" not in found
