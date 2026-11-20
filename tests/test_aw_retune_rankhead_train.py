"""Contract tests for the ADR-036 trust-head trainer (cytir.retune.rankhead_train, `cyt retune rankhead-train`).

  * RECORDING: on the fixture in tests/data/retune_rankhead/ (a deterministic subset of the task-204/206/222
    training data, see PROVENANCE.txt) the port writes the head the retired task-222 script wrote. `golden_head.json`
    is that script's own training code run on the same rows. Compared as parsed values at 1e-5 on `w` and `b`
    (another scikit-learn or BLAS build may differ in the last bits; a changed head does not pass). On the FULL
    inputs, run once on the private training box, the port is bit-identical to depth222/artifacts/rankhead_222.json (see PROVENANCE).
  * SCHEMA: the serving loader refuses a head file with missing keys (`cytir.serve.rankhead_schema`), and the
    trainer's own output passes it.
  * EMIT-ONLY: the trainer refuses to write onto an existing file or onto a head the server loads.
  * DOOR: `cyt retune rankhead-train` is the same trainer.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cytir.retune import rankhead_train
from cytir.serve.rankhead_schema import REQUIRED_KEYS, validate_head

ROOT = Path(__file__).resolve().parents[1]
FX = Path(__file__).resolve().parent / "data" / "retune_rankhead"


def argv(out: Path, *extra: str) -> list[str]:
    return ["--synth-vecs", str(FX / "synth_vecs.npy"), "--synth-meta", str(FX / "synth_meta.jsonl"),
            "--real-vecs", str(FX / "real_vecs.npy"), "--real-labels", str(FX / "real_labels.jsonl"),
            "--neg-vecs", str(FX / "neg_vecs.npy"), "--neg-labels", str(FX / "neg_labels.jsonl"),
            "--out", str(out), *extra]


def test_recording_port_reproduces_the_task_222_head(tmp_path: Path):
    pytest.importorskip("sklearn")
    out = tmp_path / "head.json"
    assert rankhead_train.run(argv(out)) == 0
    got, golden = json.loads(out.read_text()), json.loads((FX / "golden_head.json").read_text())
    assert {k: got[k] for k in ("variant", "beta_gen", "beta_max", "trust_scale")} == \
           {k: golden[k] for k in ("variant", "beta_gen", "beta_max", "trust_scale")}
    assert len(got["w"]) == len(golden["w"]) == 1024
    np.testing.assert_allclose(got["w"], golden["w"], atol=1e-5, rtol=0)
    assert got["b"] == pytest.approx(golden["b"], abs=1e-5)
    validate_head(got)


def test_report_records_input_hashes_and_validation(tmp_path: Path):
    pytest.importorskip("sklearn")
    out, report = tmp_path / "head.json", tmp_path / "report.json"
    assert rankhead_train.run(argv(out, "--report", str(report))) == 0
    r = json.loads(report.read_text())
    assert r["n_by_source"] == {"synth": 57, "real": 57, "neg": 56}
    assert set(r["inputs_sha256"]) == {"synth_vecs", "synth_meta", "real_vecs", "real_labels", "neg_vecs", "neg_labels"}


def test_schema_refuses_missing_keys():
    good = json.loads((FX / "golden_head.json").read_text())
    validate_head(good)
    for key in REQUIRED_KEYS:
        bad = {k: v for k, v in good.items() if k != key}
        with pytest.raises(ValueError, match=key):
            validate_head(bad)
    with pytest.raises(ValueError):
        validate_head({**good, "variant": "v9"})
    with pytest.raises(ValueError):
        validate_head({**good, "w": []})
    with pytest.raises(ValueError):
        validate_head({**good, "variant": "v3"})   # v3 also needs w_beta / b_beta
    with pytest.raises(ValueError):
        validate_head([])


def test_checked_in_served_head_passes_the_schema():
    validate_head(json.loads((ROOT / "cytir" / "serve" / "rankhead.json").read_text()))


def test_loader_disables_a_malformed_head(tmp_path: Path, monkeypatch):
    """A head file with a missing key is refused by the loader (head off), not half-loaded."""
    from cytir.serve import heads
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"variant": "v2", "beta_gen": 0.7}))
    monkeypatch.setattr(heads, "_RANKHEAD", True)
    monkeypatch.setattr(heads, "_RANKHEAD_PATH", str(bad))
    assert heads._load_rankhead() is None


def test_refuses_to_overwrite_or_to_write_a_served_head(tmp_path: Path):
    existing = tmp_path / "head.json"
    existing.write_text("{}")
    assert rankhead_train.run(argv(existing)) == 2
    assert existing.read_text() == "{}"
    served = ROOT / "cytir" / "serve" / "rankhead.json"
    before = served.read_bytes()
    assert rankhead_train.run(argv(served)) == 2
    assert served.read_bytes() == before


def test_refuses_out_equal_to_report(tmp_path: Path, capsys):
    same = tmp_path / "same.json"
    assert rankhead_train.run(argv(same, "--report", str(same))) == 2
    assert "out and report must be distinct" in capsys.readouterr().err
    assert not same.exists()


def test_refuses_row_count_mismatch(tmp_path: Path):
    short = tmp_path / "neg_labels.jsonl"
    short.write_text('{"intent": "general"}\n')
    args = argv(tmp_path / "head.json")
    args[args.index("--neg-labels") + 1] = str(short)
    assert rankhead_train.run(args) == 2
    assert not (tmp_path / "head.json").exists()


def test_cli_door_is_the_same_trainer(tmp_path: Path):
    pytest.importorskip("sklearn")
    out = tmp_path / "head.json"
    done = subprocess.run([sys.executable, "-m", "cytir.cli", "retune", "rankhead-train", *argv(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    direct = tmp_path / "direct.json"
    assert rankhead_train.run(argv(direct)) == 0
    assert json.loads(out.read_text()) == json.loads(direct.read_text())
