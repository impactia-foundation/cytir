"""Train a candidate-only, typed unified query-policy head.

This is deliberately an *offline* teacher-to-student stage.  Its inputs are
content-addressed JSONL counterfactuals: every accepted example records the
global retrieval utility and the utility of ``global + country`` retrieval.
The action labels are supplied by the counterfactual oracle; this trainer never
pretends that a country mention alone proves that a country leg should run.

The output is a NEW candidate copy of a qhead artifact with ``policy_heads``.
It refuses to overwrite either an existing artifact or the checked-in/live
``cytir/serve/qhead.json``.  Serving treats an artifact without that block
as inert, so merely adding this trainer cannot change retrieval.

Counterfactual row schema (``aw.query-policy-example.v1``)::

  {
    "schema": "aw.query-policy-example.v1", "example_id": "...",
    "family_id": "...", "query": "...", "embedding": [float, ...],
    "record_state": "accepted" | "rejected",
    "country": {"label": "CH" | "_NONE_" | null,
                "probabilities": {"CH": 0.9, "_NONE_": 0.1}},
    "labels": {"jurisdiction_binding": true | false | null,
               "foreign_authority_needed": true | false | null,
               "multi_scope": true | false | null,
               "country_dense_quota": number | null,
               "country_bm25_quota": number | null},
    "counterfactual": {
      "global": {"utility": number | null},
      "global_plus_country": {"utility": number | null,
        "country": "CH" | null, "dense_quota": number | null,
        "bm25_quota": number | null}
    },
    "tombstone": { ... }                  // required only for rejected rows
  }

``null`` labels are intentional missing-label masks, not zeroes.  Semantic-only
rows may set both utilities and the complete action arm to null, but then both
quota labels must also be null.  Rejected records cannot disappear: their
immutable predecessor pointer is retained in the run manifest.  Families and
normalised queries are disjoint between train and calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cytir.serve.settings import SETTINGS


SCHEMA = "aw.query-policy-example.v1"
ARTIFACT_SCHEMA = "aw.query-policy.candidate.v1"
NONE = "_NONE_"
BINARY_HEADS = (
    "jurisdiction_binding",
    "foreign_authority_needed",
    "multi_scope",
)
QUOTA_HEADS = ("country_dense_quota", "country_bm25_quota")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_sha(path: Path, expected: str, name: str) -> str:
    if not SHA256_RE.fullmatch(expected):
        raise ValueError(f"{name} sha256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _number_or_none(value: Any, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _bool_or_none(value: Any, field: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{field} must be bool or null")
    return value


def _country_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be string or null")
    if value != NONE and not re.fullmatch(r"[A-Z]{2}", value):
        raise ValueError(f"{field} must be ISO alpha-2 or {NONE}")
    return value


def _verify_country(country: Any, where: str) -> tuple[str | None, dict[str, float]]:
    if not isinstance(country, dict) or set(country) != {"label", "probabilities"}:
        raise ValueError(f"{where}.country must contain exactly label and probabilities")
    label = _country_or_none(country["label"], f"{where}.country.label")
    probabilities = country["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError(f"{where}.country.probabilities must be a non-empty mapping")
    clean: dict[str, float] = {}
    for key, value in probabilities.items():
        _country_or_none(key, f"{where}.country.probabilities key")
        score = _number(value, f"{where}.country.probabilities[{key!r}]")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{where}.country probability must be in [0, 1]")
        clean[key] = score
    if not 0.999 <= sum(clean.values()) <= 1.001:
        raise ValueError(f"{where}.country probabilities must sum to one")
    if label is not None and label not in clean:
        raise ValueError(f"{where}.country label must occur in probabilities")
    return label, clean


def _verify_tombstone(value: Any, where: str) -> dict[str, str]:
    required = {"logical_artifact", "predecessor_logical_artifact", "sha256", "reason"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{where}.tombstone must contain exactly {sorted(required)}")
    for key in ("logical_artifact", "predecessor_logical_artifact", "reason"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{where}.tombstone.{key} must be non-empty text")
    if not isinstance(value["sha256"], str) or not SHA256_RE.fullmatch(value["sha256"]):
        raise ValueError(f"{where}.tombstone.sha256 must be the lowercase SHA-256 of predecessor_logical_artifact")
    if value["logical_artifact"] == value["predecessor_logical_artifact"]:
        raise ValueError(f"{where}.tombstone logical_artifact must differ from predecessor_logical_artifact")
    # Preserve the input contract while making the digest's subject explicit in
    # the run manifest.  It identifies the immutable predecessor, not the rejected
    # row and not the newly corrected logical artifact.
    return {**{key: value[key] for key in sorted(required)},
            "sha256_subject": "predecessor_logical_artifact"}


@dataclass(frozen=True)
class Example:
    example_id: str
    family_id: str
    query: str
    embedding: np.ndarray
    country_label: str | None
    labels: dict[str, bool | float | None]
    global_utility: float | None
    country_utility: float | None


def _parse_row(raw: Any, path: Path, line_number: int, dim: int | None) -> tuple[Example | None, dict[str, Any] | None, int]:
    where = f"{path}:{line_number}"
    required = {"schema", "example_id", "family_id", "query", "embedding", "record_state", "country", "labels", "counterfactual"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        missing = sorted(required - set(raw)) if isinstance(raw, dict) else sorted(required)
        raise ValueError(f"{where}: missing required fields {missing}")
    if raw["schema"] != SCHEMA:
        raise ValueError(f"{where}: schema must be {SCHEMA}")
    for key in ("example_id", "family_id", "query"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{where}.{key} must be non-empty text")
    if raw["record_state"] not in {"accepted", "rejected"}:
        raise ValueError(f"{where}.record_state must be accepted or rejected")
    country_label, _ = _verify_country(raw["country"], where)
    embedding = raw["embedding"]
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"{where}.embedding must be a non-empty list")
    vector = np.asarray(embedding, dtype="float32")
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{where}.embedding must be a finite one-dimensional vector")
    if dim is not None and vector.shape[0] != dim:
        raise ValueError(f"{where}.embedding has dim {vector.shape[0]}, expected {dim}")
    labels = raw["labels"]
    if not isinstance(labels, dict) or set(labels) != set(BINARY_HEADS + QUOTA_HEADS):
        raise ValueError(f"{where}.labels must contain exactly {list(BINARY_HEADS + QUOTA_HEADS)}")
    parsed_labels: dict[str, bool | float | None] = {}
    for key in BINARY_HEADS:
        parsed_labels[key] = _bool_or_none(labels[key], f"{where}.labels.{key}")
    for key in QUOTA_HEADS:
        value = labels[key]
        parsed_labels[key] = None if value is None else _number(value, f"{where}.labels.{key}")
        if parsed_labels[key] is not None and parsed_labels[key] < 0:  # type: ignore[operator]
            raise ValueError(f"{where}.labels.{key} cannot be negative")
    cf = raw["counterfactual"]
    if not isinstance(cf, dict) or set(cf) != {"global", "global_plus_country"}:
        raise ValueError(f"{where}.counterfactual must contain global and global_plus_country")
    global_arm, country_arm = cf["global"], cf["global_plus_country"]
    if not isinstance(global_arm, dict) or set(global_arm) != {"utility"}:
        raise ValueError(f"{where}.counterfactual.global must contain exactly utility")
    country_fields = {"utility", "country", "dense_quota", "bm25_quota"}
    if not isinstance(country_arm, dict) or set(country_arm) != country_fields:
        raise ValueError(f"{where}.counterfactual.global_plus_country must contain {sorted(country_fields)}")
    global_utility = _number_or_none(global_arm["utility"], f"{where}.counterfactual.global.utility")
    country_utility = _number_or_none(country_arm["utility"], f"{where}.counterfactual.global_plus_country.utility")
    arm_country = _country_or_none(country_arm["country"], f"{where}.counterfactual.global_plus_country.country")
    dense_arm = _number_or_none(country_arm["dense_quota"], f"{where}.counterfactual.global_plus_country.dense_quota")
    bm25_arm = _number_or_none(country_arm["bm25_quota"], f"{where}.counterfactual.global_plus_country.bm25_quota")
    if (dense_arm is not None and dense_arm < 0) or (bm25_arm is not None and bm25_arm < 0):
        raise ValueError(f"{where}.counterfactual country quotas cannot be negative")
    quota_supervision = any(parsed_labels[key] is not None for key in QUOTA_HEADS)
    measured = (global_utility, country_utility, arm_country, dense_arm, bm25_arm)
    if any(item is None for item in measured):
        if any(item is not None for item in measured):
            raise ValueError(f"{where}: counterfactual utilities and country arm must be wholly measured or wholly null")
        if quota_supervision:
            raise ValueError(f"{where}: quota supervision requires complete measured counterfactual utility and arm config")
    elif quota_supervision:
        labelled_country = None if country_label in (None, NONE) else country_label
        if labelled_country != arm_country:
            raise ValueError(f"{where}: country label must exactly match the evaluated country arm for quota supervision")
    for key, arm_value in (("country_dense_quota", dense_arm), ("country_bm25_quota", bm25_arm)):
        value = parsed_labels[key]
        if value is not None and value > 0.0:
            # A learned action can only reproduce an actually evaluated action;
            # interpolation to an untested quota would fabricate counterfactual
            # evidence.  Zero is intentionally allowed for neutral/losing arms.
            assert arm_value is not None and global_utility is not None and country_utility is not None
            if not math.isclose(float(value), arm_value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{where}.labels.{key} must equal its evaluated counterfactual quota when positive")
            if country_utility <= global_utility:
                raise ValueError(f"{where}.labels.{key} is positive but global_plus_country utility does not strictly improve global")
    if any((parsed_labels[key] or 0.0) > 0.0 for key in QUOTA_HEADS) and (country_label in (None, NONE) or arm_country is None):
        raise ValueError(f"{where}: non-zero quota needs a labelled country and country counterfactual")
    if raw["record_state"] == "rejected":
        tombstone = _verify_tombstone(raw.get("tombstone"), where)
        return None, {"example_id": raw["example_id"], "tombstone": tombstone}, vector.shape[0]
    if "tombstone" in raw:
        raise ValueError(f"{where}: accepted rows must not carry a tombstone")
    return Example(raw["example_id"], raw["family_id"], raw["query"], vector, country_label, parsed_labels, global_utility, country_utility), None, vector.shape[0]


def load_examples(path: Path) -> tuple[list[Example], list[dict[str, Any]], int]:
    rows: list[Example] = []
    tombstones: list[dict[str, Any]] = []
    ids: set[str] = set()
    tombstone_keys: set[tuple[str, str, str]] = set()
    dim: int | None = None
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        row, tombstone, row_dim = _parse_row(raw, path, line_number, dim)
        identifier = raw["example_id"]
        if identifier in ids:
            raise ValueError(f"{path}:{line_number}: duplicate example_id {identifier!r} across accepted/rejected rows")
        ids.add(identifier)
        if dim is None:
            dim = row_dim
        if row is not None:
            rows.append(row)
        elif tombstone is not None:
            pointer = tombstone["tombstone"]
            key = (pointer["logical_artifact"], pointer["predecessor_logical_artifact"], pointer["sha256"])
            if key in tombstone_keys:
                raise ValueError(f"{path}:{line_number}: duplicate tombstone pointer")
            tombstone_keys.add(key)
            tombstones.append(tombstone)
    if not rows or dim is None:
        raise ValueError(f"{path}: no accepted rows")
    return rows, tombstones, dim


def verify_split(train: list[Example], calibration: list[Example]) -> None:
    train_families = {row.family_id for row in train}
    calibration_families = {row.family_id for row in calibration}
    overlap = train_families & calibration_families
    if overlap:
        raise ValueError(f"train/calibration family overlap: {sorted(overlap)[:3]}")
    train_queries = {norm_query(row.query) for row in train}
    calibration_queries = {norm_query(row.query) for row in calibration}
    overlap = train_queries & calibration_queries
    if overlap:
        raise ValueError(f"train/calibration normalised-query overlap: {sorted(overlap)[:3]}")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_multinomial(X: np.ndarray, y: np.ndarray, classes: int, C: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic multinomial logistic fit.

    Production training environments carry scikit-learn, which gives us its
    mature LBFGS solver.  The small NumPy fallback deliberately keeps this
    contract runnable in a minimal verifier image; it is not a silent change of
    method because the manifest names the actual solver.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        # Do not force the retired ``multi_class`` switch: current sklearn uses
        # multinomial loss for genuine multiclass data and its normal binary
        # representation for two classes.  The latter deliberately exposes one
        # class-1 decision row rather than two softmax rows.
        model = LogisticRegression(C=C, max_iter=4000, solver="lbfgs", random_state=seed)
        model.fit(X, y)
        if model.classes_.tolist() != list(range(classes)):
            raise ValueError("classifier dropped a declared class")
        coef = np.asarray(model.coef_, dtype="float32")
        intercept = np.asarray(model.intercept_, dtype="float32")
        if coef.shape == (1, X.shape[1]) and intercept.shape == (1,) and classes == 2:
            # sklearn's binary row is the signed decision score for classes_[1]
            # (the positive integer label here).  Anchor classes_[0] at zero so
            # softmax([0, z])[1] == sigmoid(z), preserving both orientation and
            # the probability that temperature calibration observes below.
            return np.vstack((np.zeros_like(coef[0]), coef[0])), np.array((0.0, intercept[0]), dtype="float32")
        if coef.shape != (classes, X.shape[1]) or intercept.shape != (classes,):
            raise ValueError(
                "unexpected LogisticRegression parameter shape: "
                f"coef={coef.shape}, intercept={intercept.shape}, classes={classes}, input_dim={X.shape[1]}"
            )
        return coef, intercept
    except ModuleNotFoundError:
        # Fixed-order, full-batch gradient descent.  The head's input vectors are
        # already normalised; a conservative step and L2 make this a useful,
        # deterministic verifier fallback for small fixtures.
        n, dim = X.shape
        W = np.zeros((classes, dim), dtype="float64")
        b = np.log(np.bincount(y, minlength=classes) / n + 1e-12).astype("float64")
        one_hot = np.eye(classes, dtype="float64")[y]
        reg = 1.0 / C
        for _ in range(1200):
            p = softmax((X @ W.T + b).astype("float64"))
            grad_w = ((p - one_hot).T @ X) / n + reg * W / n
            grad_b = (p - one_hot).mean(axis=0)
            W -= 0.35 * grad_w
            b -= 0.35 * grad_b
        return W.astype("float32"), b.astype("float32")


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """Ridge with an unpenalised intercept; no dependency is needed to rerun it."""
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=l2, fit_intercept=True)
        model.fit(X, y)
        return model.coef_.astype("float32"), float(model.intercept_)
    except ModuleNotFoundError:
        mean_x = X.mean(axis=0)
        mean_y = float(y.mean())
        centered_x = X - mean_x
        w = np.linalg.solve(centered_x.T @ centered_x + l2 * np.eye(X.shape[1]), centered_x.T @ (y - mean_y))
        return w.astype("float32"), float(mean_y - mean_x @ w)


def solver_identity() -> dict[str, str]:
    """Return the solver that this environment will actually execute."""
    try:
        import sklearn
        return {"implementation": "sklearn.lbfgs/ridge", "sklearn": sklearn.__version__,
                "numpy": np.__version__}
    except ModuleNotFoundError:
        return {"implementation": "numpy.full_batch_gd/normal_equation",
                "numpy": np.__version__}


def fit_temperature(logits: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    grid = np.concatenate((np.linspace(0.20, 1.0, 17), np.linspace(1.10, 4.0, 30), np.linspace(4.5, 8.0, 8)))
    best = (float("inf"), 1.0)
    for temperature in grid:
        probs = softmax(logits / temperature)
        nll = float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())
        if (nll, float(temperature)) < best:
            best = (nll, float(temperature))
    return best[1], best[0]


def choose_threshold(probability: np.ndarray, truth: np.ndarray) -> tuple[float, dict[str, float]]:
    # F1 is solely a calibration diagnostic.  Tie-breaking to a higher threshold
    # favours abstention: an unnecessary country leg costs latency and can crowd a
    # pool, while the global legs remain available.
    best: tuple[float, float, dict[str, float]] | None = None
    for threshold in np.arange(0.20, 0.96, 0.05):
        predicted = probability >= threshold
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        metrics = {"f1": round(f1, 6), "precision": round(precision, 6), "recall": round(recall, 6)}
        choice = (f1, float(threshold), metrics)
        if best is None or choice[:2] > best[:2]:
            best = choice
    assert best is not None
    return best[1], best[2]


def choose_country_threshold(probabilities: np.ndarray, truth: np.ndarray, none_index: int) -> tuple[float, dict[str, float]]:
    """Select abstention threshold while counting a wrong country as wrong.

    A generic binary scope/not-scope metric would make a CH prediction on an FR
    query look useful.  Country-leg routing cannot make that mistake: both the
    decision to activate and the selected partition must be correct.
    """
    choice = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(choice)), choice]
    true_scope = truth != none_index
    best: tuple[float, float, dict[str, float]] | None = None
    for threshold in np.arange(0.20, 0.96, 0.05):
        predicted_scope = (choice != none_index) & (confidence >= threshold)
        correct_scope = predicted_scope & true_scope & (choice == truth)
        tp = int(correct_scope.sum())
        precision = tp / max(int(predicted_scope.sum()), 1)
        recall = tp / max(int(true_scope.sum()), 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        metrics = {"f1": round(f1, 6), "precision": round(precision, 6), "recall": round(recall, 6)}
        choice_value = (f1, float(threshold), metrics)
        if best is None or choice_value[:2] > best[:2]:
            best = choice_value
    assert best is not None
    return best[1], best[2]


def _masked(rows: Iterable[Example], label: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row.labels[label] is not None]
    if not selected:
        raise ValueError(f"no non-missing labels for {label}")
    return np.vstack([row.embedding for row in selected]), np.asarray([row.labels[label] for row in selected])


def _binary_head(name: str, train: list[Example], calibration: list[Example], C: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    X_train, y_train_raw = _masked(train, name)
    X_cal, y_cal_raw = _masked(calibration, name)
    y_train, y_cal = y_train_raw.astype(bool), y_cal_raw.astype(bool)
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"{name}: training split requires both boolean classes")
    if len(np.unique(y_cal)) != 2:
        raise ValueError(f"{name}: calibration split requires both boolean classes")
    W_all, b_all = fit_multinomial(X_train, y_train.astype(int), 2, C, seed)
    # softmax([z0,z1]) is equivalent to sigmoid((z1-z0)); retain one vector for
    # the compact binary serving representation.
    W = W_all[1] - W_all[0]
    b = float(b_all[1] - b_all[0])
    # Binary logistic P(True) equals softmax([0, z])[1]; reuse the categorical
    # temperature scaler so all classifier confidences have one interpretation.
    cal_logits = np.column_stack((np.zeros(len(X_cal)), X_cal @ W + b))
    temperature, nll = fit_temperature(cal_logits, y_cal.astype(int))
    probability = softmax(cal_logits / temperature)[:, 1]
    threshold, metrics = choose_threshold(probability, y_cal)
    # This temperature is part of the artifact contract.  Serving must apply it
    # as sigmoid((W·x+b)/T) before comparing with the calibrated threshold.
    mask_counts = {"train_present": int(len(X_train)), "train_missing": len(train) - len(X_train), "calibration_present": int(len(X_cal)), "calibration_missing": len(calibration) - len(X_cal)}
    return ({"kind": "binary_logistic", "input_dim": int(X_train.shape[1]), "W": W.tolist(), "b": b,
             "activation": "sigmoid", "lo": 0.0, "hi": 1.0, "temperature": round(temperature, 5), "threshold": threshold,
             "positive_class": True, "missing_label_mask": mask_counts},
            {"temperature_nll": round(nll, 6), "threshold_metrics": metrics, "masks": mask_counts})


def _quota_head(name: str, train: list[Example], calibration: list[Example], max_quota: float, l2: float) -> tuple[dict[str, Any], dict[str, Any]]:
    X_train, y_train = _masked(train, name)
    X_cal, y_cal = _masked(calibration, name)
    y_train = y_train.astype("float32")
    y_cal = y_cal.astype("float32")
    if (y_train > max_quota).any() or (y_cal > max_quota).any():
        raise ValueError(f"{name}: oracle target exceeds configured max quota {max_quota}")
    # Store a fraction rather than an absolute candidate count.  Serving maps this
    # to its separately capped budget, so a budget change does not require a
    # retrain or silently reinterpret an old artifact.
    y_train_fraction = y_train / max_quota
    y_cal_fraction = y_cal / max_quota
    W, b = fit_ridge(X_train, y_train_fraction, l2)
    predicted = np.clip(X_cal @ W + b, 0.0, 1.0)
    mae = float(np.abs(predicted - y_cal_fraction).mean())
    masks = {"train_present": int(len(X_train)), "train_missing": len(train) - len(X_train), "calibration_present": int(len(X_cal)), "calibration_missing": len(calibration) - len(X_cal)}
    return ({"kind": "bounded_scalar", "input_dim": int(X_train.shape[1]), "W": W.tolist(), "b": b,
             "activation": "identity", "lo": 0.0, "hi": 1.0, "training_max_quota": max_quota,
             "missing_label_mask": masks},
            {"calibration_fraction_mae": round(mae, 6), "training_max_quota": max_quota, "masks": masks})


def _country_head(train: list[Example], calibration: list[Example], C: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    tr = [row for row in train if row.country_label is not None]
    cal = [row for row in calibration if row.country_label is not None]
    if not tr or not cal:
        raise ValueError("country_scope: no non-missing country labels")
    classes = [NONE] + sorted({row.country_label for row in tr if row.country_label != NONE})
    if len(classes) < 2:
        raise ValueError("country_scope: training split requires _NONE_ and at least one country")
    index = {label: number for number, label in enumerate(classes)}
    if any(row.country_label not in index for row in cal):
        raise ValueError("country_scope: calibration contains country absent from training")
    X_train = np.vstack([row.embedding for row in tr])
    y_train = np.asarray([index[row.country_label] for row in tr])
    X_cal = np.vstack([row.embedding for row in cal])
    y_cal = np.asarray([index[row.country_label] for row in cal])
    if set(np.unique(y_train)) != set(range(len(classes))):
        raise ValueError("country_scope: training split lacks a declared class")
    W, b = fit_multinomial(X_train, y_train, len(classes), C, seed)
    temperature, nll = fit_temperature(X_cal @ W.T + b, y_cal)
    probabilities = softmax((X_cal @ W.T + b) / temperature)
    none_index = index[NONE]
    threshold, metrics = choose_country_threshold(probabilities, y_cal, none_index)
    masks = {"train_present": len(tr), "train_missing": len(train) - len(tr), "calibration_present": len(cal), "calibration_missing": len(calibration) - len(cal)}
    return ({"kind": "classifier", "input_dim": int(X_train.shape[1]), "classes": classes, "none_class": NONE,
             "W": W.tolist(), "b": b.tolist(), "temperature": round(temperature, 5), "threshold": threshold,
             "missing_label_mask": masks},
            {"temperature_nll": round(nll, 6), "scope_activation_metrics": metrics, "masks": masks,
             "class_counts": dict(sorted(Counter(row.country_label for row in tr).items()))})


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def utility_summary(rows: list[Example], prefix: str) -> dict[str, int | float | None]:
    measured = [row for row in rows if row.global_utility is not None and row.country_utility is not None]
    return {
        f"{prefix}_measured_rows": len(measured),
        f"{prefix}_missing_rows": len(rows) - len(measured),
        f"{prefix}_global_mean": None if not measured else round(float(np.mean([row.global_utility for row in measured])), 6),
        f"{prefix}_global_plus_country_mean": None if not measured else round(float(np.mean([row.country_utility for row in measured])), 6),
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--base-qhead", required=True, type=Path)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--base-qhead-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-country-dense-quota", type=float, default=50.0)
    parser.add_argument("--max-country-bm25-quota", type=float, default=50.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=374)
    args = parser.parse_args(argv)
    if args.C <= 0 or args.l2 <= 0 or args.max_country_dense_quota <= 0 or args.max_country_bm25_quota <= 0:
        raise ValueError("C, l2, and max quotas must be positive")
    outputs = [args.out.resolve(), args.report.resolve(), args.manifest.resolve()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("out, report, and manifest must be distinct")
    live_qheads = {Path(__file__).resolve().parents[1] / "serve" / "qhead.json",
                   Path(SETTINGS.runtime.qhead_path).resolve()}
    if args.out.resolve() == args.base_qhead.resolve() or args.out.resolve() in live_qheads:
        raise ValueError("refusing to mutate a base or live qhead; output must be a new candidate artifact")
    if any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite an output artifact")
    if not args.train.is_file() or not args.calibration.is_file() or not args.base_qhead.is_file():
        raise ValueError("train, calibration, and base-qhead must be files")
    train_hash = _require_sha(args.train, args.train_sha256, "train")
    calibration_hash = _require_sha(args.calibration, args.calibration_sha256, "calibration")
    base_hash = _require_sha(args.base_qhead, args.base_qhead_sha256, "base-qhead")
    train, train_tombstones, dim = load_examples(args.train)
    calibration, calibration_tombstones, calibration_dim = load_examples(args.calibration)
    if dim != calibration_dim:
        raise ValueError(f"train/calibration embedding dim mismatch: {dim} != {calibration_dim}")
    train_ids = {row.example_id for row in train} | {row["example_id"] for row in train_tombstones}
    calibration_ids = {row.example_id for row in calibration} | {row["example_id"] for row in calibration_tombstones}
    if train_ids & calibration_ids:
        raise ValueError(f"train/calibration example_id overlap: {sorted(train_ids & calibration_ids)[:3]}")
    def tombstone_key(row: dict[str, Any]) -> tuple[str, str, str]:
        pointer = row["tombstone"]
        return pointer["logical_artifact"], pointer["predecessor_logical_artifact"], pointer["sha256"]
    train_pointers = {tombstone_key(row) for row in train_tombstones}
    calibration_pointers = {tombstone_key(row) for row in calibration_tombstones}
    if train_pointers & calibration_pointers:
        raise ValueError("train/calibration duplicate tombstone pointer")
    verify_split(train, calibration)
    base = json.loads(args.base_qhead.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or int(base.get("input_dim", dim)) != dim:
        raise ValueError(f"base-qhead input_dim must equal counterfactual embedding dim {dim}")
    country_head, country_report = _country_head(train, calibration, args.C, args.seed)
    policy_heads: dict[str, Any] = {"country_scope": country_head}
    report_heads: dict[str, Any] = {"country_scope": country_report}
    for name in BINARY_HEADS:
        policy_heads[name], report_heads[name] = _binary_head(name, train, calibration, args.C, args.seed)
    quota_max = {"country_dense_quota": args.max_country_dense_quota, "country_bm25_quota": args.max_country_bm25_quota}
    for name in QUOTA_HEADS:
        policy_heads[name], report_heads[name] = _quota_head(name, train, calibration, quota_max[name], args.l2)
    candidate = dict(base)
    # This top-level role is part of the serving contract: a candidate artifact
    # cannot become active merely by being copied beside qhead.json.
    candidate["artifact_role"] = "candidate_only"
    candidate["policy_heads"] = policy_heads
    candidate["query_policy_candidate"] = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_role": "candidate_only",
        "activation": "none; requires an explicit serving experiment flag and gate",
        "base_qhead_sha256": base_hash,
        "input_dim": dim,
        "train_sha256": train_hash,
        "calibration_sha256": calibration_hash,
    }
    report = {"schema": ARTIFACT_SCHEMA, "artifact_role": "candidate_only", "heads": report_heads,
              "counterfactual_utility": {**utility_summary(train, "train"),
                                         **utility_summary(calibration, "calibration")}}
    # Write candidate then report then manifest.  A failed later write cannot turn the
    # candidate live; it is deliberately outside the live qhead path and marked candidate-only.
    write_new_json(args.out, candidate)
    write_new_json(args.report, report)
    manifest = {"schema": ARTIFACT_SCHEMA, "stage": "aw.retune.policy_train.v1", "deterministic_rerun": {"seed": args.seed, "solver": solver_identity(), "input_order": "JSONL order", "overwrite": "forbidden"},
                "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
                "inputs": {"train": {"path": str(args.train), "sha256": train_hash, "accepted_rows": len(train), "rejected_rows": len(train_tombstones)}, "calibration": {"path": str(args.calibration), "sha256": calibration_hash, "accepted_rows": len(calibration), "rejected_rows": len(calibration_tombstones)}, "base_qhead": {"path": str(args.base_qhead), "sha256": base_hash}},
                "rejected_row_tombstone_pointers": {"train": train_tombstones, "calibration": calibration_tombstones},
                "config": {"C": args.C, "l2": args.l2, "max_country_dense_quota": args.max_country_dense_quota, "max_country_bm25_quota": args.max_country_bm25_quota},
                "outputs": {"candidate_qhead": {"path": str(args.out), "sha256": sha256_file(args.out)}, "report": {"path": str(args.report), "sha256": sha256_file(args.report)}}}
    write_new_json(args.manifest, manifest)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


def run(argv: list[str] | None = None, prog: str | None = None) -> int:
    """`main` with the contract's refusals reported as exit code 2 (what the old script did)."""
    try:
        return main(argv, prog=prog)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
