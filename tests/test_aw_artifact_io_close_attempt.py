from pathlib import Path

from cytir.acquire import artifact_io as subject


def test_successor_boundary_closes_without_mutating_empty_metrics(tmp_path: Path):
    prefix = tmp_path / "attempt-0001"; successor = tmp_path / "attempt-0002"
    Path(str(prefix) + ".started").write_text("2026-08-15T00:00:00Z\n")
    Path(str(prefix) + ".start_offset").write_text("0\n")
    Path(str(prefix) + ".headers").write_text("HTTP/2 200\r\n")
    metrics = Path(str(prefix) + ".metrics"); metrics.write_bytes(b"")
    Path(str(successor) + ".started").write_text("2026-08-15T00:01:00Z\n")
    Path(str(successor) + ".start_offset").write_text("123\n")
    result = subject.close_attempt(prefix=prefix, successor_prefix=successor,
                           url="https://official/source.zip")
    assert result["end_offset"] == 123 and result["return_code"] == 143
    assert metrics.read_bytes() == b""
    assert Path(str(prefix) + ".end_offset").read_text() == "123\n"
    assert Path(str(prefix) + ".recovery.json").is_file()
