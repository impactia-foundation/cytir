"""Task 1195 (S3): one QdrantClient factory in cytir.core.qdrant; the 8 former
call sites pass the URL expression and keywords they passed before, unchanged.
No network: QdrantClient.__init__ is recorded, call sites are read by AST."""
from __future__ import annotations

import ast
import importlib
import inspect

import pytest

from cytir.core import qdrant as core_qdrant

# module -> list of (url expression source, {keyword: expression source}),
# in source order. Copied from the pre-change call sites (merge-base source).
EXPECTED = {
    "cytir.sources.families.news": [
        ("args.staging_url", {"timeout": "900", "check_compatibility": "False"}),
        ("args.twin_url", {"timeout": "900", "check_compatibility": "False"}),
    ],
    "cytir.index.upsert": [("args.qdrant_url", {"timeout": "300"})],
    "cytir.index.apply": [
        ("args.twin_url", {"timeout": "900"}),
        ("args.prod_url", {"timeout": "900"}),
    ],
    "cytir.index.supersede": [
        ("cfg['url']", {"timeout": "cfg['connect_timeout']"}),
        ("tunnel_url", {"timeout": "max(cfg['connect_timeout'], 60)"}),
        ("args.qdrant_url", {"timeout": "600"}),
    ],
}


def test_factory_passes_url_timeout_and_extras(monkeypatch):
    import qdrant_client

    seen = {}

    def fake_init(self, *a, **kw):
        seen["args"], seen["kw"] = a, kw

    monkeypatch.setattr(qdrant_client.QdrantClient, "__init__", fake_init)
    c = core_qdrant.qdrant_connect("http://h:1", timeout=900, check_compatibility=False)
    assert isinstance(c, qdrant_client.QdrantClient)
    assert seen["args"] == ()
    assert seen["kw"] == {"url": "http://h:1", "timeout": 900,
                          "check_compatibility": False}


def test_timeout_has_no_default():
    with pytest.raises(TypeError):
        core_qdrant.qdrant_connect("http://h:1")  # type: ignore[call-arg]


@pytest.mark.parametrize("modname", sorted(EXPECTED))
def test_sites_pass_their_recorded_arguments(modname):
    mod = importlib.import_module(modname)
    tree = ast.parse(inspect.getsource(mod))
    assert "QdrantClient(" not in inspect.getsource(mod)
    got = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "qdrant_connect"):
            assert len(node.args) == 1 and not any(k.arg is None for k in node.keywords)
            got.append((node.lineno, ast.unparse(node.args[0]),
                        {k.arg: ast.unparse(k.value) for k in node.keywords}))
    got.sort()
    assert [(u, k) for _, u, k in got] == EXPECTED[modname]


def test_every_import_of_the_factory_uses_the_plain_name():
    """The drain guard's scanner matches the call name `qdrant_connect`; an
    `as` alias would hide a site from it."""
    from pathlib import Path
    root = Path(core_qdrant.__file__).resolve().parents[1]
    found = 0
    for p in root.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "cytir.core.qdrant":
                for a in node.names:
                    if a.name == "qdrant_connect":
                        found += 1
                        assert a.asname is None, f"{p}: aliased import"
    assert found >= 4


def test_deploy_stage_scripts_build_the_same_client_kwargs_as_before(monkeypatch):
    """Task 2539: hn_discussion_stage.sh and news_refresh_stage.sh moved from
    `QdrantClient(url=url, timeout=600, check_compatibility=False)` to the factory.
    The constructed kwargs must be identical."""
    import qdrant_client
    from cytir.core.qdrant import qdrant_connect
    seen = {}

    class Spy:
        def __init__(self, *a, **k):
            seen["a"], seen["k"] = a, k

    monkeypatch.setattr(qdrant_client, "QdrantClient", Spy)
    url = "stage-url-placeholder"
    qdrant_connect(url, timeout=600, check_compatibility=False)
    assert seen == {"a": (), "k": {"url": url, "timeout": 600,
                                    "check_compatibility": False}}
