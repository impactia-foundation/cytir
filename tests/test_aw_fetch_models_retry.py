"""fetch_models retries a truncated model download from scratch (task 1712).

The transport is monkeypatched: each call to ``make_client`` serves the next body
from a script, so no network is touched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from cytir.embed import fetch_models
from cytir.embed.model_assets import ModelAsset

FULL = b"authoritative-web-fake-model-bytes" * 4
TRUNCATED = FULL[:30]


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, _n: int):  # decoded body; iter_raw (wire bytes) broke gzip hosts, task 2251
        yield self._body


class _Ctx:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, bodies: list[bytes], calls: list[int]):
        self._bodies = bodies
        self._calls = calls

    def stream(self, _method: str, _url: str):
        self._calls[0] += 1
        return _Ctx(_Resp(self._bodies.pop(0)))


class _TmpAsset(ModelAsset):
    def __init__(self, dest: Path):
        super().__init__(
            name="splitter-test", filename="splitter.bin",
            url="https://example.invalid/splitter.bin",
            sha256=hashlib.sha256(FULL).hexdigest(), size_bytes=len(FULL),
            env_var="AW_TEST_SPLITTER", why="test fixture")
        self._dest = dest

    def resolve(self):
        return str(self._dest), False

    def default_dest(self):
        return self._dest


def _patch(monkeypatch, bodies: list[bytes]) -> list[int]:
    calls = [0]
    script = list(bodies)
    monkeypatch.setattr(
        fetch_models, "make_client",
        lambda _spec: _Ctx(_FakeClient(script, calls)))
    return calls


def test_truncated_then_full_succeeds_after_one_retry(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "splitter.bin"
    calls = _patch(monkeypatch, [TRUNCATED, FULL])
    assert fetch_models.handle(_TmpAsset(dest), check_only=False, force=False) is True
    assert calls[0] == 2
    assert dest.read_bytes() == FULL
    assert not (tmp_path / "splitter.bin.part").exists()
    err = capsys.readouterr().err
    assert "attempt 1/3 failed, retrying from scratch" in err
    assert "size mismatch" in err


def test_three_truncations_fail_with_mismatch_message(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "splitter.bin"
    calls = _patch(monkeypatch, [TRUNCATED, TRUNCATED, TRUNCATED])
    assert fetch_models.handle(_TmpAsset(dest), check_only=False, force=False) is False
    assert calls[0] == 3
    assert not dest.exists()
    assert not (tmp_path / "splitter.bin.part").exists()
    captured = capsys.readouterr()
    assert "attempt 2/3 failed" in captured.err
    assert "FETCH FAILED" in captured.err
    assert "size mismatch" in captured.err


def test_a_missing_crawler_identity_fails_at_once_without_a_retry(tmp_path, monkeypatch, capsys):
    """F8: an identity error cannot succeed on a second try; it names the variable to set."""
    from cytir.core.user_agent import UnidentifiedCrawler
    calls = [0]

    def no_identity(_spec):
        calls[0] += 1
        raise UnidentifiedCrawler("AW crawler identity is not configured.\n  Set it:  export AW_OPERATOR_URL=...")

    monkeypatch.setattr(fetch_models, "make_client", no_identity)
    dest = tmp_path / "splitter.bin"
    assert fetch_models.handle(_TmpAsset(dest), check_only=False, force=False) is False
    assert calls[0] == 1
    err = capsys.readouterr().err
    assert "retrying" not in err and "AW_OPERATOR_URL" in err
