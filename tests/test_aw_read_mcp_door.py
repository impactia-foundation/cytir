"""`cyt read --mcp` is the package door for the cytir-reader MCP server (task 2754)."""
from unittest import mock

import pytest

from cytir.reader import cli as reader_cli


def test_mcp_flag_serves_without_url():
    pytest.importorskip("mcp.server.mcpserver")
    with mock.patch("cytir.reader.mcp.main") as serve:
        assert reader_cli.main(["--mcp"]) == 0
    serve.assert_called_once_with()


def test_read_without_url_or_mcp_is_an_error():
    with pytest.raises(SystemExit):
        reader_cli.main([])


def test_plain_read_does_not_import_mcp_module(monkeypatch):
    import sys
    sys.modules.pop("cytir.reader.mcp", None)
    monkeypatch.setattr(reader_cli.R, "read", lambda *a, **k: {"url": "u"})
    monkeypatch.setattr(reader_cli, "find_renderer", lambda *a, **k: None)
    assert reader_cli.main(["http://example.org/"]) == 0
    assert "cytir.reader.mcp" not in sys.modules
