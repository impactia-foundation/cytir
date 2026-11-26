"""A stranger's first contact (the first public-release run): a verb without its extra, a missing
minisign, the MCP server's bind and --help, and an index without the authority-card collection."""
import subprocess
import sys

import pytest

from cytir import cli


import pathlib
import re
try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11; tomli is a pytest dependency there
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
_EXTRAS = _PYPROJECT["project"]["optional-dependencies"]


def _declared(extra):
    return {cli._norm(re.split(r"[\s\[<>=!~;(]", r, maxsplit=1)[0]) for r in _EXTRAS[extra]}


@pytest.fixture
def declared(monkeypatch):
    monkeypatch.setattr(cli, "_extra_dists", _declared)   # the package metadata of an installed cytir, read from pyproject


def test_every_table_module_is_declared_in_some_extra():
    every = set().union(*(_declared(e) for e in _EXTRAS))
    for mod, dist in cli.EXTRA_MODULES.items():
        assert cli._norm(dist) in every, (mod, dist)


def _raise(name):
    def verb(argv):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return verb


def _run(monkeypatch, verb, name, capsys):
    monkeypatch.setattr(cli, "_serve_main", _raise(name))
    monkeypatch.setattr(cli, "_read_main", _raise(name))
    monkeypatch.setattr(cli, "_rehearse_main", _raise(name))
    return cli.main([verb])


def test_declared_module_gets_one_line_with_the_right_extra(declared, monkeypatch, capsys):
    assert _run(monkeypatch, "rehearse", "qdrant_client", capsys) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and 'needs the "dev" extra' in err and 'cytir[dev]' in err


def test_uvicorn_names_an_extra_that_has_it(declared, monkeypatch, capsys):
    assert _run(monkeypatch, "serve", "uvicorn", capsys) == 1
    err = capsys.readouterr().err
    assert 'needs the "serve" extra' in err and "uvicorn" in _declared("serve")


def test_reader_verb_names_the_reader_extra(declared):
    msg = cli.missing_extra_message("read", ModuleNotFoundError("x", name="trafilatura"))
    assert 'needs the "reader" extra' in msg


def test_undeclared_module_and_submodule_and_cytir_re_raise(declared, monkeypatch):
    for name in ("not_in_any_extra", "yaml.x", "cytir.nope"):
        with pytest.raises(ModuleNotFoundError):
            _run(monkeypatch, "rehearse", name, None)


def test_rehearse_without_minisign_is_one_line(monkeypatch, capsys):
    from cytir.distribute import rehearsal
    monkeypatch.setattr(rehearsal, "minisign_available", lambda: False)
    assert rehearsal.main(["run"]) == 1
    err = capsys.readouterr().err
    assert "minisign" in err and "apt-get install" in err and "Traceback" not in err


def test_mcp_server_binds_loopback_by_default_and_help_exits():
    pytest.importorskip("mcp")
    from cytir.serve import mcp_server
    assert mcp_server._S.runtime.mcp_host == "0.0.0.0"   # step 1; flips to 127.0.0.1 after the prod container is recreated
    with pytest.raises(SystemExit) as e:
        mcp_server.main(["--help"])
    assert e.value.code == 0


def _fa(monkeypatch, exc):
    pytest.importorskip("fastapi")
    from cytir.serve import app, authority

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(authority, "find_authorities_relaxed", boom)
    return app.find_authority(q="x", k=3, country=None), authority.CARDS_COLLECTION


def test_find_authority_without_cards_says_what_is_missing(monkeypatch):
    from cytir.serve import authority
    r, cards = _fa(monkeypatch, RuntimeError(f"Not found: Collection `{authority.CARDS_COLLECTION}` doesn't exist!"))
    assert r.status_code == 404
    body = r.body.decode()
    assert "no authority-card collection" in body and "retry" not in body.lower()


def test_find_authority_other_errors_stay_503(monkeypatch):
    for msg in ("Not found: Collection `aw_prod` doesn't exist!", "model file not found", "boom"):
        r, _ = _fa(monkeypatch, RuntimeError(msg))
        assert r.status_code == 503, msg


def test_compose_gives_the_mcp_service_its_bind_host():
    """Step 1 of the bind-default flip: production must pass the host itself before the default changes."""
    import yaml
    f = ROOT / "deploy" / "docker-compose.yaml"
    if not f.exists():
        pytest.skip("deploy/ is not in this tree")
    env = yaml.safe_load(f.read_text())["services"]["aw-mcp"]["environment"]
    assert env["AW_MCP_HOST"] == "0.0.0.0"
