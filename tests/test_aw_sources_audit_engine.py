import gzip
import json

import pytest

from cytir.sources import audit as shared


def test_stage_rows_single_shard_refuses_a_stage_with_two_shards(tmp_path, monkeypatch):
    for name in ("a", "b"):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt", encoding="utf-8") as out:
            out.write(json.dumps({"n": name}) + "\n")
    receipt = {"artifacts": [{"uri": "artifact://a.jsonl.gz"},
                             {"uri": "artifact://b.jsonl.gz"}]}
    monkeypatch.setattr(shared, "_validate_stage", lambda *a, **k: receipt)
    assert [r["n"] for r in shared.stage_rows(tmp_path, {})] == ["a", "b"]
    with pytest.raises(ValueError, match="exactly one"):
        shared.stage_rows(tmp_path, {}, single_shard=True)
