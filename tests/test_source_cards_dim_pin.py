"""Task 2641: the source_cards dense width is the SEALED served profile's, and it equals the width
the deployed server embeds queries at; `--recreate` refuses a silent change of the card count."""
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from cytir.corpus import source_cards
from cytir.core.serving_profile import SERVED

APP = Path(__file__).resolve().parents[1]


def _deploy_env_dim() -> int:
    text = (APP / "deploy" / "docker-compose.flip.yaml").read_text()
    m = re.search(r"AW_MRL_DIM:\s*'?(\d+)'?", text)
    assert m, "the deploy file no longer names AW_MRL_DIM"
    return int(m.group(1))


def _server_query_dim(env_dim: int) -> int:
    """`cytir.serve.models.MRL_DIM` as the server computes it, in a fresh process with the deploy's env."""
    env = dict(os.environ, AW_MRL_DIM=str(env_dim))
    out = subprocess.run([sys.executable, "-c", "from cytir.serve import models; print(models.MRL_DIM)"],
                         cwd=APP, env=env, capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[-1])


def test_cards_width_is_served_profile_and_deployed_query_width(monkeypatch):
    if not (APP / "deploy" / "docker-compose.flip.yaml").is_file():
        pytest.skip("deploy/docker-compose.flip.yaml is not in this tree")
    monkeypatch.delenv("AW_MRL_DIM", raising=False)
    assert source_cards.cards_dim() == SERVED.dense_dim == 256
    deploy = _deploy_env_dim()
    assert deploy == SERVED.dense_dim
    assert _server_query_dim(deploy) == source_cards.cards_dim()


def test_env_that_disagrees_is_refused_with_both_values(monkeypatch):
    monkeypatch.setenv("AW_MRL_DIM", "128")
    with pytest.raises(SystemExit) as e:
        source_cards.cards_dim()
    assert "128" in str(e.value) and "256" in str(e.value)


def test_collection_created_at_served_width(monkeypatch):
    monkeypatch.delenv("AW_MRL_DIM", raising=False)
    import cytir.core.serving_profile as SP
    seen = {}
    monkeypatch.setattr(SP, "create_collection", lambda c, n, prof, **kw: seen.update(dim=prof.dense_dim))
    source_cards.ensure_source_cards_collection(None, None, "x", source_cards.cards_dim(), False)
    assert seen["dim"] == SERVED.dense_dim


class _Client:
    def __init__(self, n, payloads=()):
        self.n, self.payloads = n, list(payloads)

    def count(self, name, exact=True):
        return types.SimpleNamespace(count=self.n)

    def scroll(self, name, limit, offset, with_payload, with_vectors):
        return [types.SimpleNamespace(payload=p) for p in self.payloads], None


def _stub_stack(monkeypatch, client):
    monkeypatch.setattr(source_cards, "_load_stack",
                        lambda: (lambda url, timeout=0: client, None, None, None, None, None))


def _args(**kw):
    base = dict(qdrant_url="u", collection="source_cards", recreate=True, batch=8, allow_count_change=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_recreate_refuses_count_change_naming_both_counts(monkeypatch):
    _stub_stack(monkeypatch, _Client(2759))
    with pytest.raises(SystemExit) as e:
        source_cards._embed_cards(_args(), [{"domain": str(i)} for i in range(9111)])
    assert "2759" in str(e.value) and "9111" in str(e.value)


def test_reembed_refuses_a_non_cards_name_before_scrolling(monkeypatch):
    client = _Client(69_000_000)
    client.scroll = lambda *a, **k: pytest.fail("scrolled")
    _stub_stack(monkeypatch, client)
    with pytest.raises(SystemExit) as e:
        source_cards.cmd_reembed(_args(from_collection="aw_prod", collection=None))
    assert "aw_prod" in str(e.value)


def test_reembed_refuses_an_oversized_source_before_scrolling(monkeypatch):
    client = _Client(250_000)
    client.scroll = lambda *a, **k: pytest.fail("scrolled")
    _stub_stack(monkeypatch, client)
    with pytest.raises(SystemExit) as e:
        source_cards.cmd_reembed(_args(from_collection="source_cards", collection=None))
    assert "250000" in str(e.value)


def test_reembed_reads_the_collections_own_payloads(monkeypatch):
    payloads = [{"domain": "a.ch"}, {"domain": "b.ch"}]
    _stub_stack(monkeypatch, _Client(2, payloads))
    got = {}
    monkeypatch.setattr(source_cards, "_embed_cards", lambda args, cards: got.update(cards=cards, coll=args.collection))
    source_cards.cmd_reembed(_args(from_collection="source_cards", collection=None))
    assert got == {"cards": payloads, "coll": "source_cards"}
