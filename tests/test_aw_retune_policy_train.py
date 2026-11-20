"""Contract tests for the candidate-only query-policy trainer (cytir.retune.policy_train, `cyt retune policy-train`).

Besides the trainer's refusals, three claims are pinned here:
  * RECORDING: on the fixture below the port writes a candidate and a report that are BYTE-identical to what
    the retired script wrote, once with the numpy solver and once with sklearn (goldens in tests/data/retune_policy/;
    the manifest differs by design: stage id and code path/sha). The test compares parsed values (1e-9 numpy,
    1e-5 sklearn), so another BLAS or sklearn build does not fail it, but a changed head does.
  * ROUND TRIP: the candidate passes the serving loader (`cytir.serve.heads._load_query_policy_heads`) and the
    loaded head predicts what the artifact's own weights say.
  * DOOR: `cyt retune policy-train` is the same trainer.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "retune_policy"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(identifier: str, family: str, vector: list[float], country: str, binding: bool, foreign: bool, multi: bool, dense: int | None, bm25: int | None, state: str = "accepted") -> dict:
    measured = dense is not None and bm25 is not None
    value = {
        "schema": "aw.query-policy-example.v1", "example_id": identifier,
        "family_id": family, "query": f"query {identifier}", "embedding": vector,
        "record_state": state,
        "country": {"label": country, "probabilities": {country: 1.0}},
        "labels": {"jurisdiction_binding": binding, "foreign_authority_needed": foreign,
                   "multi_scope": multi, "country_dense_quota": dense,
                   "country_bm25_quota": bm25},
        "counterfactual": {"global": {"utility": 0.1 if measured else None}, "global_plus_country": {
            "utility": (0.9 if dense else 0.1) if measured else None,
            "country": (None if country == "_NONE_" else country) if measured else None,
            "dense_quota": 8 if measured else None, "bm25_quota": 8 if measured else None}},
    }
    if state == "rejected":
        value["tombstone"] = {"logical_artifact": f"candidate/{identifier}",
                              "predecessor_logical_artifact": "raw/counterfactual/v1",
                              "sha256": "a" * 64, "reason": "adjudication rejected this oracle label"}
    return value


def write_rows(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def invoke(tmp_path: Path, train: Path, calibration: Path, base: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    out, report, manifest = (tmp_path / "candidate.json", tmp_path / "report.json", tmp_path / "manifest.json")
    return subprocess.run([sys.executable, "-m", "cytir.retune.policy_train", "--train", str(train), "--calibration", str(calibration), "--base-qhead", str(base),
                           "--train-sha256", sha(train), "--calibration-sha256", sha(calibration), "--base-qhead-sha256", sha(base),
                           "--out", str(out), "--report", str(report), "--manifest", str(manifest),
                           "--max-country-dense-quota", "8", "--max-country-bm25-quota", "8", *extra], text=True, capture_output=True, cwd=ROOT)


def fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    # Every action has both values in both splits.  The country classes are stable
    # across the family-disjoint split, while the tombstone proves rejected rows are
    # retained in provenance but never used for fitting.
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    write_rows(train, [
        row("t1", "family-1", [3, 0, 0, 0], "CH", True, False, False, 8, 8),
        row("t2", "family-2", [2, 0, 0, 0], "CH", True, False, False, 8, 8),
        row("t3", "family-3", [0, 3, 0, 0], "CH", False, True, False, 0, 0),
        row("t4", "family-4", [0, 2, 0, 0], "CH", False, True, True, 0, 0),
        row("t5", "family-9", [0, 0, 3, 0], "_NONE_", False, False, False, None, None),
        row("tomb", "family-tomb", [9, 9, 9, 9], "CH", True, False, False, 8, 8, "rejected"),
    ])
    calibration_rows = [
        row("c1", "family-5", [2.5, 0, 0, 0], "CH", True, False, False, 8, 8),
        row("c2", "family-6", [1.5, 0, 0, 0], "CH", True, False, False, 8, 8),
        row("c3", "family-7", [0, 2.5, 0, 0], "CH", False, True, False, 0, 0),
        row("c4", "family-8", [0, 1.5, 0, 0], "CH", False, True, True, 0, 0),
        row("c5", "family-10", [0, 0, 2.5, 0], "_NONE_", False, False, False, None, None),
    ]
    # Unknown is not negative: this is deliberately withheld from the quota
    # loss and must survive as an explicit missing-label count in the artifact.
    calibration_rows[-1]["labels"]["country_dense_quota"] = None
    write_rows(calibration, calibration_rows)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"input_dim": 4, "knobs": {"legacy": {"alpha": 0}}}), encoding="utf-8")
    return train, calibration, base


def test_candidate_artifact_is_typed_content_addressed_and_inert(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 0, result.stderr
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["knobs"] == {"legacy": {"alpha": 0}}
    assert candidate["artifact_role"] == "candidate_only"
    assert set(candidate["policy_heads"]) == {"country_scope", "jurisdiction_binding", "foreign_authority_needed", "multi_scope", "country_dense_quota", "country_bm25_quota"}
    assert candidate["policy_heads"]["country_dense_quota"]["kind"] == "bounded_scalar"
    assert candidate["policy_heads"]["country_dense_quota"]["hi"] == 1.0
    assert candidate["policy_heads"]["country_dense_quota"]["missing_label_mask"]["calibration_missing"] == 1
    assert candidate["policy_heads"]["jurisdiction_binding"]["kind"] == "binary_logistic"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inputs"]["train"]["sha256"] == sha(train)
    assert manifest["deterministic_rerun"]["solver"]["implementation"] in {
        "sklearn.lbfgs/ridge", "numpy.full_batch_gd/normal_equation"}
    assert manifest["rejected_row_tombstone_pointers"]["train"][0]["example_id"] == "tomb"
    assert manifest["rejected_row_tombstone_pointers"]["train"][0]["tombstone"]["sha256_subject"] == "predecessor_logical_artifact"
    assert manifest["outputs"]["candidate_qhead"]["sha256"] == sha(tmp_path / "candidate.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["counterfactual_utility"]["train_missing_rows"] == 1
    assert report["counterfactual_utility"]["train_measured_rows"] == 4


def test_sklearn_binary_shape_preserves_positive_probability_and_artifact_contract(tmp_path: Path):
    """sklearn legitimately emits one row for a two-class classifier.

    The candidate artifact still stores one positive-class sigmoid logit, so it
    must exactly reproduce sklearn's uncalibrated class-True probability before
    its separately stored temperature is applied at serving time.
    """
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    train, calibration, base = fixtures(tmp_path)
    train_rows = [json.loads(line) for line in train.read_text().splitlines()]
    accepted = [value for value in train_rows if value["record_state"] == "accepted"]
    X = np.asarray([value["embedding"] for value in accepted], dtype="float32")
    y = np.asarray([int(value["labels"]["jurisdiction_binding"]) for value in accepted])
    model = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs", random_state=374).fit(X, y)
    assert sklearn.__version__
    assert model.classes_.tolist() == [0, 1]
    assert model.coef_.shape == (1, X.shape[1])
    assert model.intercept_.shape == (1,)

    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 0, result.stderr
    head = json.loads((tmp_path / "candidate.json").read_text())["policy_heads"]["jurisdiction_binding"]
    W = np.asarray(head["W"], dtype="float64")
    b = float(head["b"])
    probes = np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype="float32")
    artifact_probability = 1.0 / (1.0 + np.exp(-(probes @ W + b)))
    positive_index = model.classes_.tolist().index(1)
    np.testing.assert_allclose(artifact_probability, model.predict_proba(probes)[:, positive_index], rtol=1e-6, atol=1e-7)
    assert artifact_probability[0] > artifact_probability[1]
    assert head["kind"] == "binary_logistic"
    assert head["input_dim"] == X.shape[1]
    assert W.shape == (X.shape[1],)
    assert head["positive_class"] is True
    assert head["activation"] == "sigmoid"
    assert 0.0 < head["temperature"]

    # country_scope has exactly the same valid binary sklearn shape in this
    # fixture, but its artifact is genuinely multiclass: two aligned logits.
    # This proves the shared normalization keeps both representations usable.
    scope_y = np.asarray([0 if value["country"]["label"] == "_NONE_" else 1 for value in accepted])
    scope_model = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs", random_state=374).fit(X, scope_y)
    assert scope_model.coef_.shape == (1, X.shape[1])
    scope = json.loads((tmp_path / "candidate.json").read_text())["policy_heads"]["country_scope"]
    scope_W = np.asarray(scope["W"], dtype="float64")
    scope_b = np.asarray(scope["b"], dtype="float64")
    scope_logits = probes @ scope_W.T + scope_b
    scope_probabilities = np.exp(scope_logits - scope_logits.max(axis=1, keepdims=True))
    scope_probabilities /= scope_probabilities.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(scope_probabilities, scope_model.predict_proba(probes), rtol=1e-6, atol=1e-7)
    assert scope_W.shape == (2, X.shape[1])
    assert scope_b.shape == (2,)


def test_refuses_family_overlap_and_existing_output(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in calibration.read_text().splitlines()]
    values[0]["family_id"] = "family-1"
    write_rows(calibration, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "family overlap" in result.stderr

    # A candidate never gets silently repaired in place: rerunning with the same
    # output paths must fail rather than create unverifiable provenance.
    train, calibration, base = fixtures(tmp_path)
    first = invoke(tmp_path, train, calibration, base)
    assert first.returncode == 0, first.stderr
    second = invoke(tmp_path, train, calibration, base)
    assert second.returncode == 2
    assert "refusing to overwrite" in second.stderr


def test_rejected_row_requires_tombstone(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[-1].pop("tombstone")
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "tombstone" in result.stderr


def test_positive_quota_must_be_measured_and_strictly_better(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[0]["labels"]["country_dense_quota"] = 4
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "must equal its evaluated" in result.stderr

    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[0]["counterfactual"]["global_plus_country"]["utility"] = 0.1
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "does not strictly improve" in result.stderr


def test_semantic_rows_may_mask_action_evidence_but_not_invent_partial_arms(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    semantic = next(value for value in values if value["example_id"] == "t5")
    assert semantic["counterfactual"]["global"]["utility"] is None
    assert semantic["labels"]["country_dense_quota"] is None
    # Claiming even part of an arm while utility is missing is dishonest.
    semantic["counterfactual"]["global_plus_country"]["country"] = "CH"
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "wholly measured or wholly null" in result.stderr


def test_quota_supervision_requires_same_country_as_evaluated_arm(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[0]["counterfactual"]["global_plus_country"]["country"] = "DE"
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "country label must exactly match" in result.stderr


def test_ids_and_tombstone_predecessor_pointers_are_unique(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[-1]["example_id"] = "t1"
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "duplicate example_id" in result.stderr

    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    duplicate = json.loads(json.dumps(values[-1]))
    duplicate["example_id"] = "tomb-duplicate"
    duplicate["family_id"] = "family-tomb-duplicate"
    duplicate["query"] = "query tomb duplicate"
    values.append(duplicate)
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "duplicate tombstone pointer" in result.stderr

    train, calibration, base = fixtures(tmp_path)
    values = [json.loads(line) for line in train.read_text().splitlines()]
    values[-1]["tombstone"]["logical_artifact"] = values[-1]["tombstone"]["predecessor_logical_artifact"]
    write_rows(train, values)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 2
    assert "must differ from predecessor" in result.stderr


def _assert_close(got, want, where, tol=1e-9):
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), where
        for key in want:
            _assert_close(got[key], want[key], f"{where}.{key}", tol)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for index, (g, w) in enumerate(zip(got, want)):
            _assert_close(g, w, f"{where}[{index}]", tol)
    elif isinstance(want, float):
        assert got == pytest.approx(want, abs=tol), where
    else:
        assert got == want, where


def test_recording_port_reproduces_the_retired_script_output(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    result = invoke(tmp_path, train, calibration, base)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stage"] == "aw.retune.policy_train.v1"
    # Two solver paths, one recording each (the old script's output, byte-identical to the port's at record time).
    solver = "sklearn" if manifest["deterministic_rerun"]["solver"]["implementation"].startswith("sklearn") else "numpy"
    tol = 1e-5 if solver == "sklearn" else 1e-9
    for name in ("candidate", "report"):
        want = json.loads((GOLDEN / f"golden_{solver}_{name}.json").read_text())
        _assert_close(json.loads((tmp_path / f"{name}.json").read_text()), want, name, tol)


_LOADER_PROBE = """
import json
import numpy as np
from cytir.serve import heads
h = heads._QUERY_POLICY_HEADS
assert h is not None, "serving loader rejected the candidate"
x = np.asarray([2.0, 0.0, 0.0, 0.0], dtype="float32")
b = h["scalar"]["jurisdiction_binding"]
print(json.dumps({"dim": h["dim"], "classes": h["country"]["classes"], "heads": sorted(h["scalar"]),
                  "logit": float(x @ b["W"] + b["b"])}))
"""


def test_round_trip_candidate_passes_the_serving_loader_and_predicts(tmp_path: Path):
    import os
    train, calibration, base = fixtures(tmp_path)
    assert invoke(tmp_path, train, calibration, base).returncode == 0
    candidate = tmp_path / "candidate.json"
    env = {**os.environ, "AW_QHEAD_COUNTRY_LEGS": "1", "AW_QHEAD_PATH": str(candidate)}
    probe = subprocess.run([sys.executable, "-c", _LOADER_PROBE], text=True, capture_output=True, cwd=ROOT, env=env)
    assert probe.returncode == 0, probe.stderr[-800:]
    loaded = json.loads(probe.stdout.strip().splitlines()[-1])
    head = json.loads(candidate.read_text())["policy_heads"]["jurisdiction_binding"]
    expected = float(np.asarray([2.0, 0.0, 0.0, 0.0]) @ np.asarray(head["W"]) + head["b"])
    assert loaded["dim"] == 4 and "CH" in loaded["classes"]
    assert loaded["heads"] == sorted(["jurisdiction_binding", "foreign_authority_needed", "multi_scope",
                                      "country_dense_quota", "country_bm25_quota"])
    assert loaded["logit"] == pytest.approx(expected, abs=1e-5)
    # Inert by default: the same artifact is ignored unless the serving flag is set.
    off = subprocess.run([sys.executable, "-c", "from cytir.serve import heads; print(heads._QUERY_POLICY_HEADS)"],
                         text=True, capture_output=True, cwd=ROOT, env={**env, "AW_QHEAD_COUNTRY_LEGS": "0"})
    assert off.stdout.strip().splitlines()[-1] == "None"


def test_cli_door_is_the_same_trainer_and_refuses_the_live_qhead(tmp_path: Path):
    train, calibration, base = fixtures(tmp_path)
    live = ROOT / "cytir" / "serve" / "qhead.json"
    args = ["--train", str(train), "--calibration", str(calibration), "--base-qhead", str(base),
            "--train-sha256", sha(train), "--calibration-sha256", sha(calibration), "--base-qhead-sha256", sha(base),
            "--report", str(tmp_path / "r.json"), "--manifest", str(tmp_path / "m.json")]
    door = [sys.executable, "-m", "cytir.cli", "retune", "policy-train"]
    refused = subprocess.run([*door, *args, "--out", str(live)], text=True, capture_output=True, cwd=ROOT)
    assert refused.returncode == 2 and "refusing to mutate" in refused.stderr
    ok = subprocess.run([*door, *args, "--out", str(tmp_path / "c.json"),
                         "--max-country-dense-quota", "8", "--max-country-bm25-quota", "8"],
                        text=True, capture_output=True, cwd=ROOT)
    assert ok.returncode == 0, ok.stderr[-500:]
    assert set(json.loads((tmp_path / "c.json").read_text())["policy_heads"]) == set(
        json.loads((GOLDEN / "golden_numpy_candidate.json").read_text())["policy_heads"])
