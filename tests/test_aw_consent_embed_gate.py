#!/usr/bin/env python3
"""Task 677 (task 643 S5): the consent stack moved into `cytir/consent/` (and the
crawler identity into `cytir/core/user_agent.py`), and the recorded-decline
refusal is now enforced in the embed stage too.

Two halves:
  * MOVE: the old `consent_*.py` / `user_agent.py` paths are
    shims that alias the package module object; the data paths the moved files
    compute (one directory level deeper) still land on the repo's `data/`.
  * EMBED GATE: `cytir.consent.embed_gate.assert_stage_consent` refuses a stage
    holding a host with a recorded blocking decision, and both the shard worker
    (`cytir.embed.shards.main`) and `cytir.embed.stage.embed_stage` call it BEFORE a
    model loads or a GPU lease is taken.
"""
from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.consent import block, embed_gate  # noqa: E402

CRAWLER_ENV = {"AW_OPERATOR_URL": "https://example.org/about",
               "AW_OPERATOR_CONTACT": "ops@example.org"}


# --------------------------------------------------------------------- MOVE

@pytest.fixture()
def crawler_identity(monkeypatch):
    for key, value in CRAWLER_ENV.items():
        monkeypatch.setenv(key, value)


def test_the_moved_files_still_find_the_repos_data_directory():
    from cytir.consent import resolver
    assert pathlib.Path(block.registry_path()) == REPO / "data" / "consent_declined_hosts.json"
    assert pathlib.Path(block.registry_path()).is_file()
    assert pathlib.Path(resolver.CONSENT_UNRESOLVED_REGISTER) == (
        REPO / "data" / "consent-unresolved-register.yaml")
    assert pathlib.Path(resolver.CONSENT_UNRESOLVED_REGISTER).is_file()
    from cytir.core import user_agent
    assert user_agent.REPO_ROOT == REPO
    assert user_agent.EXAMPLE_FILE.is_file()


def test_the_cache_dir_default_is_unchanged_and_now_overridable(monkeypatch, tmp_path):
    from cytir.consent import resolver
    if "AW_CONSENT_CACHE_DIR" not in os.environ:
        assert resolver.CACHE_DIR == os.path.expanduser("~/aw-data/consent")
    code = ("import os; os.environ['AW_CONSENT_CACHE_DIR']=%r; "
            "from cytir.consent import resolver; print(resolver.CACHE_DIR)"
            % str(tmp_path))
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          env=dict(os.environ, **CRAWLER_ENV),
                          capture_output=True, text=True, timeout=60)
    assert done.stdout.strip() == str(tmp_path), done.stderr[-300:]


def test_the_modules_run_as_commands():
    """The two CLIs shell scripts call: the UA printer and the block check."""
    env = dict(os.environ, **CRAWLER_ENV)
    ua = subprocess.run([sys.executable, "-m", "cytir.core.user_agent"],
                        capture_output=True, text=True, env=dict(env, PYTHONPATH=str(REPO)), cwd="/", timeout=60)
    assert ua.returncode == 0 and "example.org" in ua.stdout, ua.stderr[-300:]
    ok = subprocess.run([sys.executable, "-m", "cytir.consent.block",
                         "www.fedlex.admin.ch"], capture_output=True, text=True,
                        env=dict(env, PYTHONPATH=str(REPO)), cwd="/", timeout=60)
    assert ok.returncode == 0 and ok.stdout.startswith("OK"), ok.stderr[-300:]
    bad = subprocess.run([sys.executable, "-m", "cytir.consent.block",
                          "prestations.vd.ch"], capture_output=True, text=True,
                         env=dict(env, PYTHONPATH=str(REPO)), cwd="/", timeout=60)
    assert bad.returncode == 1 and "BLOCKED" in bad.stdout


# --------------------------------------------------------------- EMBED GATE

def _registry(tmp_path, **hosts):
    path = tmp_path / "declined.json"
    path.write_text(json.dumps({"schema_version": 1, "hosts": {
        host: {"state": state, "decided_on": "2026-09-30",
               "unblock_condition": "test"} for host, state in hosts.items()}}))
    return str(path)


def _stage(tmp_path, shards: dict[str, list]):
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    for name, urls in shards.items():
        with gzip.open(stage / f"{name}.texts.jsonl.gz", "wt") as fh:
            for i, url in enumerate(urls):
                row = {"point_id": f"{name}-{i}", "text": "t"}
                if url is not None:
                    row["url"] = url
                fh.write(json.dumps(row) + "\n")
    return str(stage)


def test_a_clean_stage_passes_and_is_counted(tmp_path):
    reg = _registry(tmp_path, **{"bad.example": "declined"})
    stage = _stage(tmp_path, {"shard-0000": ["https://www.fedlex.admin.ch/a"] * 3})
    assert embed_gate.assert_stage_consent(stage, registry=reg) == {
        "rows": 3, "hosts": 1, "rows_without_url": 0}


def test_a_declined_host_refuses_and_names_host_rows_and_shard(tmp_path):
    reg = _registry(tmp_path, **{"bad.example": "declined"})
    stage = _stage(tmp_path, {
        "shard-0000": ["https://ok.example/a", "https://bad.example/x"],
        "shard-0001": ["https://bad.example/y", "https://bad.example/z"]})
    with pytest.raises(block.ConsentBlocked) as err:
        embed_gate.assert_stage_consent(stage, registry=reg)
    message = str(err.value)
    assert "EMBED REFUSED" in message and "bad.example rows=3" in message
    assert "shard-0000.texts.jsonl.gz" in message
    assert "shard-0001.texts.jsonl.gz" in message
    assert "ok.example" not in message


def test_a_blocked_parent_domain_covers_its_subdomains(tmp_path):
    reg = _registry(tmp_path, **{"bad.example": "declined"})
    stage = _stage(tmp_path, {"shard-0000": ["https://www.sub.bad.example/x"]})
    with pytest.raises(block.ConsentBlocked):
        embed_gate.assert_stage_consent(stage, registry=reg)


def test_unknown_consent_is_not_consent(tmp_path):
    """The registry's own rule: `unknown` blocks exactly like `declined`."""
    reg = _registry(tmp_path, **{"maybe.example": "unknown"})
    stage = _stage(tmp_path, {"shard-0000": ["https://maybe.example/x"]})
    with pytest.raises(block.ConsentBlocked):
        embed_gate.assert_stage_consent(stage, registry=reg)


def test_an_unreadable_registry_fails_closed(tmp_path):
    stage = _stage(tmp_path, {"shard-0000": ["https://ok.example/x"]})
    with pytest.raises(block.ConsentRegistryError):
        embed_gate.assert_stage_consent(stage, registry=str(tmp_path / "nope.json"))


def test_an_empty_stage_is_not_a_clean_one(tmp_path):
    (tmp_path / "stage").mkdir()
    with pytest.raises(FileNotFoundError):
        embed_gate.assert_stage_consent(str(tmp_path / "stage"),
                                        registry=_registry(tmp_path))


def test_rows_without_a_url_are_counted_and_logged_not_silently_passed(tmp_path):
    reg = _registry(tmp_path)
    stage = _stage(tmp_path, {"shard-0000": [None, "https://ok.example/x", ""]})
    lines = []
    got = embed_gate.assert_stage_consent(stage, registry=reg, log=lines.append)
    assert got == {"rows": 3, "hosts": 1, "rows_without_url": 2}
    assert lines and "2 of 3" in lines[0] and "could NOT be consent-checked" in lines[0]


def test_shards_only_limits_the_scope(tmp_path):
    reg = _registry(tmp_path, **{"bad.example": "declined"})
    stage = _stage(tmp_path, {"shard-0000": ["https://ok.example/a"],
                              "shard-0001": ["https://bad.example/b"]})
    embed_gate.assert_stage_consent(stage, registry=reg,
                                    shards_only=["shard-0000.texts.jsonl.gz"])
    with pytest.raises(block.ConsentBlocked):
        embed_gate.assert_stage_consent(stage, registry=reg)


def test_the_repos_own_registry_refuses_a_recorded_decline(tmp_path):
    stage = _stage(tmp_path, {"shard-0000": ["https://prestations.vd.ch/api/x"]})
    with pytest.raises(block.ConsentBlocked, match="prestations.vd.ch"):
        embed_gate.assert_stage_consent(stage)


def test_the_shard_worker_refuses_before_any_model_loads(tmp_path, monkeypatch):
    shards = pytest.importorskip("cytir.embed.shards")
    reg = _registry(tmp_path, **{"bad.example": "declined"})
    monkeypatch.setenv("AW_CONSENT_REGISTRY", reg)
    stage = _stage(tmp_path, {"shard-0000": ["https://bad.example/x"]})

    def loaded(_args):
        raise AssertionError("the model was loaded before the consent gate")

    monkeypatch.setattr(shards, "load_model", loaded)
    with pytest.raises(block.ConsentBlocked, match="EMBED REFUSED"):
        shards.main(["--in", stage, "--out", str(tmp_path / "vec")])


def test_the_shard_worker_embeds_a_clean_stage_past_the_gate(tmp_path, monkeypatch):
    """Not a vacuous refusal: a clean stage gets PAST the gate to load_model."""
    shards = pytest.importorskip("cytir.embed.shards")
    monkeypatch.setenv("AW_CONSENT_REGISTRY", _registry(tmp_path))
    stage = _stage(tmp_path, {"shard-0000": ["https://ok.example/x"]})

    class Reached(Exception):
        pass

    def loaded(_args):
        raise Reached

    monkeypatch.setattr(shards, "load_model", loaded)
    with pytest.raises(Reached):
        shards.main(["--in", stage, "--out", str(tmp_path / "vec")])


def test_embed_stage_refuses_before_the_gpu_check_or_the_lock(tmp_path, monkeypatch):
    from cytir.embed import stage as stage_mod
    monkeypatch.setenv("AW_CONSENT_REGISTRY",
                       _registry(tmp_path, **{"bad.example": "declined"}))
    chunks = _stage(tmp_path, {"shard-0000": ["https://bad.example/x"]})

    def boom(*_a, **_k):
        raise AssertionError("reached the GPU check / lock / subprocess")

    monkeypatch.setattr(stage_mod, "check_gpu_idle", boom)
    monkeypatch.setattr(stage_mod, "gpu_lease", boom)
    monkeypatch.setattr(stage_mod.subprocess, "run", boom)
    with pytest.raises(block.ConsentBlocked, match="EMBED REFUSED"):
        stage_mod.embed_stage(chunks, str(tmp_path / "vec"))
