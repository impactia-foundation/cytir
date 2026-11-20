"""Contract tests for the country_scope head trainer (cytir.retune.qhead_scope_train, `cyt retune qhead-scope-train`).

  * RECORDING: on a small synthetic labelled set (fixed seed, built below) the port emits the knob entry the retired
    task-281 script emitted. ORIGINAL_SCRIPT_BLOB is the blob of that script as it stood on origin/main before
    task 1464; ENTRY_SHA256 was produced ONCE by running that script on this exact set and hashing the emitted
    `country_scope` entry as json.dumps(sort_keys=True). Pinned with scikit-learn 1.7.2.
  * EMIT-ONLY: the trainer refuses the served qhead, its own base file, and an existing output as --out.
  * DOOR: `cyt retune qhead-scope-train` is the same trainer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cytir.retune import qhead_scope_train

ORIGINAL_SCRIPT_BLOB = "c137e8199822aead5b0ea1df91935332ed0ccd60"
# Digest of the entry with W and b rounded to 2 decimals: the exact-float digest (72ed0679...) held only
# on sklearn 1.7.2; 1.9.1 moves W/b by ~3e-5 (solver rounding), so the digest pins the entry at 2 dp.
EXACT_SHA256 = "72ed067923973a3ecad37930bb54e120e213eaaa655561877726e97a70634f04"
ROUNDED_SHA256 = "a9d0285982b9f444f045c28020e17324de3a5a73a24542357f954c70dacb51fa"
ROOT = Path(__file__).resolve().parents[1]
QUERIES = [("Mietrecht Kanton Zürich", "CH"), ("tax law in Germany", "DE"), ("droit du travail France", "FR"),
           ("Ontario privacy act", "CA"), ("Art. 336 OR Schweiz", "CH"), ("python list sort", None),
           ("weather tomorrow", None), ("best pasta recipe", None), ("how to tie a tie", None)]
DIM = 12


def build_corpus(root: Path) -> dict[str, str]:
    """Three source dirs in the layout the trainer reads; every vector carries its scope's class offset."""
    rng = np.random.default_rng(1464)
    offsets = {c: rng.normal(size=DIM) * 3 for c in ("CH", "DE", "FR", "CA", None)}
    paths = {}
    for name, n in (("d205", 120), ("d206", 90), ("d222", 90)):
        rows, vecs = [], []
        for i in range(n):
            q, c = QUERIES[i % len(QUERIES)]
            rows.append({"query": q, "lang": "en"})
            vecs.append(offsets[c] + rng.normal(size=DIM))
        d = root / name
        (d / "artifacts").mkdir(parents=True)
        (d / "data").mkdir()
        vecs = np.array(vecs, dtype="float32")
        body = "\n".join(json.dumps(r) for r in rows) + "\n"
        if name == "d205":
            np.save(d / "train_head_vecs.npy", vecs)
            (d / "train_head_meta.jsonl").write_text(body)
        elif name == "d206":
            np.save(d / "artifacts" / "train_real_vecs.npy", vecs)
            (d / "data" / "train_real.jsonl").write_text(body)
        else:
            np.save(d / "neg222_vecs.npy", vecs)
            (d / "neg222_kept.jsonl").write_text(body)
        paths[name] = str(d)
    base = root / "base.json"
    base.write_text(json.dumps({"knobs": {"other": {"kind": "linear"}}, "phase": "x"}))
    paths["qhead"] = str(base)
    return paths


def argv(p: dict[str, str], out: Path) -> list[str]:
    return ["--d205", p["d205"], "--d206", p["d206"], "--d222", p["d222"], "--qhead", p["qhead"],
            "--out", str(out), "--threshold", "0.60"]


def test_recording_port_reproduces_the_task_281_entry(tmp_path: Path):
    pytest.importorskip("sklearn")
    p = build_corpus(tmp_path)
    out = tmp_path / "cand.json"
    assert qhead_scope_train.run(argv(p, out)) == 0
    spec = json.loads(out.read_text())
    assert spec["knobs"]["other"] == {"kind": "linear"}
    entry = spec["knobs"]["country_scope"]
    assert entry["train"]["C"] == 1.0 and entry["train"]["val_frac"] == 0.15 and entry["threshold"] == 0.6
    rounded = dict(entry, W=np.round(np.array(entry["W"]), 2).tolist(), b=np.round(np.array(entry["b"]), 2).tolist())
    assert hashlib.sha256(json.dumps(rounded, sort_keys=True).encode()).hexdigest() == ROUNDED_SHA256
    import sklearn
    if sklearn.__version__ == "1.7.2":  # the recording was made under 1.7.2; newer solvers drift ~3e-5 (task 1862/1867)
        assert hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest() == EXACT_SHA256
    assert "W" not in json.loads((tmp_path / "cand.scope.report.json").read_text())["country_scope"]


def test_refuses_served_base_and_existing_out(tmp_path: Path):
    pytest.importorskip("sklearn", reason="optional dependency: scikit-learn (pipeline extra)")
    p = build_corpus(tmp_path)
    served = ROOT / "cytir" / "serve" / "qhead.json"
    assert qhead_scope_train.run(argv(p, served)) == 2
    assert qhead_scope_train.run(argv(p, Path(p["qhead"]))) == 2
    existing = tmp_path / "there.json"
    existing.write_text("{}")
    assert qhead_scope_train.run(argv(p, existing)) == 2
    assert existing.read_text() == "{}"


def test_refuses_out_without_json_suffix(tmp_path: Path):
    pytest.importorskip("sklearn", reason="optional dependency: scikit-learn (pipeline extra)")
    p = build_corpus(tmp_path)
    assert qhead_scope_train.run(argv(p, tmp_path / "candidate.bin")) == 2
    assert not (tmp_path / "candidate.bin").exists()


def test_door_is_the_same_trainer(capsys):
    from cytir.cli import main
    with pytest.raises(SystemExit) as e:
        main(["retune", "qhead-scope-train", "--help"])
    assert e.value.code == 0
    assert "--d205" in capsys.readouterr().out


def test_trainer_raises_no_future_warning_and_sees_five_classes(tmp_path: Path):
    """The retired ``multi_class`` switch is gone: no FutureWarning, and the trainer sees 5 classes (multinomial)."""
    import warnings
    pytest.importorskip("sklearn")
    p = build_corpus(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert qhead_scope_train.run(argv(p, tmp_path / "cand.json")) == 0
    entry = json.loads((tmp_path / "cand.json").read_text())["knobs"]["country_scope"]
    assert len(entry["W"]) == 5
