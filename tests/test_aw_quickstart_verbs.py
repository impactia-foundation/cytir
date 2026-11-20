"""QUICKSTART.md names `cyt` verbs that exist, and no private script for the model fetch
(task 725, slice 12).

Parses the fenced bash blocks. Every command line that starts with `cyt ` must name a verb the
parser registers; the `VERB SUB` pair must answer `--help` with exit 0 (a verb that dispatches
to a retired or missing door fails here, not on a stranger's machine). The model fetch is the
package verb `cyt embed models`, never the old private script.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
QUICKSTART = REPO / "QUICKSTART.md"
if not QUICKSTART.is_file():
    pytest.skip("no QUICKSTART.md in this tree", allow_module_level=True)


def _aw_commands() -> list[list[str]]:
    text = QUICKSTART.read_text(encoding="utf-8")
    cmds = []
    for block in re.findall(r"```(?:bash|sh)\n(.*?)```", text, flags=re.S):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("cyt "):
                cmds.append(line.split())
    return cmds


def test_quickstart_has_aw_commands_and_the_model_fetch_is_one_of_them():
    cmds = _aw_commands()
    assert cmds, "no `cyt ...` command in a fenced block: the parser found nothing to check"
    assert ["cyt", "embed", "models"] in [c[:3] for c in cmds]


def test_quickstart_does_not_name_the_private_model_fetch_script():
    assert "fetch_models" + ".py" not in QUICKSTART.read_text(encoding="utf-8")


def test_every_quickstart_aw_command_is_a_verb_that_answers_help():
    from cytir.cli import build_parser
    registered = set(build_parser()._subparsers._group_actions[0].choices)
    seen = set()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    for cmd in _aw_commands():
        verb = cmd[1]
        assert verb in registered, f"QUICKSTART names `cyt {verb}`; registered: {sorted(registered)}"
        sub = cmd[2] if len(cmd) > 2 and not cmd[2].startswith("-") else None
        key = (verb, sub)
        if key in seen:
            continue
        seen.add(key)
        res = subprocess.run([sys.executable, "-m", "cytir.cli", verb, *([sub] if sub else []), "--help"],
                             cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, f"`cyt {verb} {sub or ''} --help` -> {res.returncode}: {res.stderr[-400:]}"


def test_aw_embed_models_check_fails_on_an_overridden_embedder_pin(monkeypatch, capsys):
    from cytir.embed import fetch_models
    monkeypatch.setenv("AW_MODEL_REVISION", "0" * 40)
    assert fetch_models.main(["--check", "--asset", "charsplit-ngram"]) == 1
    assert "OVERRIDE" in capsys.readouterr().out


def test_quickstart_references_only_files_the_public_cut_ships():
    """Task 905: a stranger holds the public cut, not this tree. Every `pip install -r FILE`
    and every relative markdown link in QUICKSTART.md must name a file the cut ships (the
    clean-machine run hit a private requirements file and a `specs/` link that were not
    there)."""
    from cytir.ops import public_cut

    try:
        shipped = set(public_cut.plan(REPO)["shipped"])
    except OSError:
        pytest.skip("no public-cut allowlist in this tree")
    except public_cut.CutRefusal as e:
        pytest.fail(f"public cut refused: {e}")
    text = QUICKSTART.read_text(encoding="utf-8")
    refs = set(re.findall(r"pip install (?:-q )?-r (\S+)", text))
    refs |= {m for m in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
             if not re.match(r"[a-z]+:", m)}
    missing = sorted(r for r in refs if r not in shipped)
    assert not missing, f"QUICKSTART.md names files the public cut does not ship: {missing}"


# --- task 911: the real-embedder path on a CPU-only box --------------------------------------

def _real_embedder_section() -> str:
    text = QUICKSTART.read_text(encoding="utf-8")
    start = text.index("## The real-embedder path on a CPU-only box")
    end = text.index("\n---\n", start)
    return text[start:end]


def test_real_embedder_section_names_the_cpu_flags_the_serve_extra_and_the_pinned_revision():
    sec = _real_embedder_section()
    flat = sec.replace("\\\n", " ")
    # `cyt embed shards` defaults to cuda:0 / float16 and dies on a CPU-only box
    assert "cyt embed shards" in flat and "--device cpu" in flat and "--compute-dtype float32" in flat
    # `.[pipeline]` alone has no fastapi; `cyt serve` needs the serve extra
    assert '".[pipeline,serve]"' in flat
    # the doc quotes the revision the code pins
    from cytir.core.model_pin import embed_model_revision
    assert embed_model_revision() in sec


def test_the_serve_extra_carries_fastapi_and_the_pipeline_extra_does_not():
    try:
        import tomllib
    except ModuleNotFoundError:  # python < 3.11
        import tomli as tomllib
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    names = lambda k: {re.split(r"[<>=\[ ;]", r, maxsplit=1)[0].lower() for r in extras[k]}  # noqa: E731
    assert "fastapi" in names("serve")
    assert "fastapi" not in names("pipeline")  # why the doc must say `.[pipeline,serve]`


def _undeclared_extras(text: str, declared: set[str]) -> list[str]:
    """Every `pip install -e ".[EXTRA]"` (or single-quoted) names a declared extra."""
    named = re.findall(r"pip install[^\n]*?-e\s+[\"']\.\[([A-Za-z0-9_,-]+)\]", text)
    wanted = {x.strip() for n in named for x in n.split(",")}
    return sorted(wanted - declared)


def test_undeclared_extra_is_refused_by_the_extras_check():
    """Negative control: the extras check must flag an extra pyproject.toml does not declare."""
    fake = 'pip install -e ".[nosuchextra]"\n~/venv/bin/pip install -e \'.[dev]\''
    assert _undeclared_extras(fake, {"dev"}) == ["nosuchextra"]
    assert _undeclared_extras(fake, {"dev", "nosuchextra"}) == []


def test_shards_cli_defaults_to_auto_so_the_doc_says_the_cpu_default():
    src = (REPO / "cytir" / "embed" / "shards.py").read_text(encoding="utf-8")
    assert re.search(r'add_argument\("--device", default="auto"', src), (
        "the default moved: update QUICKSTART's real-embedder section and this test")
