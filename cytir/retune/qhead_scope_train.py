"""ADR-062 Phase-2b (task 281) — train the LEARNED country_scope head (knob 13,
amended Q2): a CALIBRATED, ABSTAINING, SOFT classifier over the frozen 1024-d
arctic query embedding that emits P(country | query).

Design (spec §7 Q2 AMENDED / §1 knob-13):
  * COLD-START = self-distil the deterministic detector: the per-query label is
    `_query_jurisdiction(q)` (keyword lexicon) rolled up, ELSE the gazetteer
    rollup, ELSE _NONE_ (abstain). Homograph queries (gazetteer abstain) are
    labelled _NONE_. A head trained on THIS reproduces the keyword+gazetteer prior
    by construction, then GENERALISES to markerless/aliased forms via the embedding
    (the embedder already knows Ontario→CA), which a table cannot enumerate.
  * CALIBRATION = temperature scaling fit on a held-out split (min NLL), so the
    confidence threshold is meaningful.
  * ABSTENTION = argmax==_NONE_ OR calibrated top-prob < threshold → None. "A wrong
    scope is worse than no scope" (ADR-054).
  * Gold/probe sets are TEST — never trained on (ADR-036 discipline).

Merges the `country_scope` classifier entry (kind=classifier, enabled, classes,
W: C×1024, b: C, temperature, threshold, none_class) into a COPY of the base
qhead, leaving every other (linear) knob untouched. Serving reads it via
_load_scope_head(). Moved from the task-281 script (task 1464); the only change
to the recipe is emit-only: the base qhead is read, the result goes to --out, and
the served qhead file is refused as --out (promotion is a Tim PR).

Usage (where the vecs + arctic model live; paths are placeholders):
  cyt retune qhead-scope-train \
     --d205 /path/to/depth205/artifacts \
     --d206 /path/to/depth206 \
     --d222 /path/to/depth222/artifacts \
     --qhead cytir/serve/qhead.json --out /scratch/qhead.candidate.json \
     --threshold 0.60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NONE = "_NONE_"


def load_corpus(d205, d206, d222):
    Xs = np.load(os.path.join(d205, "train_head_vecs.npy"))
    ms = [json.loads(l) for l in open(os.path.join(d205, "train_head_meta.jsonl"))]
    Xr = np.load(os.path.join(d206, "artifacts", "train_real_vecs.npy"))
    mr = [json.loads(l) for l in open(os.path.join(d206, "data", "train_real.jsonl"))]
    Xn = np.load(os.path.join(d222, "neg222_vecs.npy"))
    mn = [json.loads(l) for l in open(os.path.join(d222, "neg222_kept.jsonl"))]
    assert Xs.shape[0] == len(ms) and Xr.shape[0] == len(mr) and Xn.shape[0] == len(mn)
    X = np.vstack([Xs, Xr, Xn]).astype("float32")
    meta = ms + mr + mn
    return X, meta


def scope_label(query, lang, juris_fn, gaz_fn):
    """The cold-start label: keyword scope > gazetteer rollup > _NONE_ (abstain).
    A gazetteer homograph (abstain hit, no rollup) => _NONE_."""
    _cantons, q_country = juris_fn(query)
    if q_country:
        return q_country
    gaz_country, _abstain = gaz_fn(query)
    if gaz_country:
        return gaz_country
    return NONE


def fit_temperature(logits, y_idx, grid=None):
    """Temperature scaling: pick T>0 minimising validation NLL of softmax(logits/T)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.5, 8.0, 10)])
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


def _served_qheads():
    from pathlib import Path
    from cytir.serve.settings import SETTINGS
    return {(Path(__file__).resolve().parents[1] / "serve" / "qhead.json").resolve(),
            Path(SETTINGS.runtime.qhead_path).resolve()}


def main(argv=None, prog=None):
    from cytir.serve.retrieval import _query_jurisdiction, _gazetteer_scope

    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--d205", required=True)
    ap.add_argument("--d206", required=True)
    ap.add_argument("--d222", required=True)
    ap.add_argument("--qhead", required=True, help="base qhead.json (read only)")
    ap.add_argument("--out", required=True, help="the NEW qhead file (never the served one)")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 reg")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression

    if not args.out.endswith(".json"):
        raise ValueError("--out must end with .json (the scope report path is derived from it)")
    out_real = os.path.realpath(args.out)
    report_path = str(Path(args.out).with_suffix(".scope.report.json"))
    if out_real in {str(p) for p in _served_qheads()} or out_real == os.path.realpath(args.qhead):
        raise ValueError("refusing to write the base or served qhead; emit a candidate and promote it by PR")
    if os.path.exists(args.out) or os.path.exists(report_path):
        raise ValueError("refusing to overwrite an output artifact")

    X, meta = load_corpus(args.d205, args.d206, args.d222)
    n, dim = X.shape
    y = np.array([scope_label(m["query"], m.get("lang"),
                              _query_jurisdiction, _gazetteer_scope) for m in meta])
    labels, counts = np.unique(y, return_counts=True)
    dist = dict(zip(labels.tolist(), counts.tolist()))
    print(f"corpus N={n} dim={dim}")
    print("label distribution:", dict(sorted(dist.items(), key=lambda kv: -kv[1])))

    # class order: _NONE_ first, then the scope countries present (stable).
    classes = [NONE] + sorted([c for c in labels if c != NONE])
    cidx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cidx[c] for c in y])

    # train/val split (stratified-ish by simple shuffle; scope classes are rare).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val, tr = perm[:n_val], perm[n_val:]

    # Do not force the retired ``multi_class`` switch (removed in sklearn 1.8):
    # lbfgs uses the multinomial loss for genuine multiclass data (here >= 3
    # classes: _NONE_ plus the scope countries); the binary case is expanded below.
    # ``n_jobs`` only ever parallelised one-vs-rest, so it was inert here and warns on >= 1.8.
    clf = LogisticRegression(
        C=args.C, max_iter=2000, class_weight="balanced",
        solver="lbfgs")
    # sklearn drops classes absent from the training fold; guard by ensuring the
    # split keeps every class (rare ones may all fall in one side).
    present = set(np.unique(y_idx[tr]).tolist())
    if len(present) < len(classes):
        # fall back to fitting on ALL data (calibration on val still informative).
        clf.fit(X, y_idx)
        used_classes = list(clf.classes_)
    else:
        clf.fit(X[tr], y_idx[tr])
        used_classes = list(clf.classes_)

    # Build a full (C, dim) weight matrix aligned to `classes`; sklearn emits rows
    # only for classes it saw. Absent classes get an all-zero row + a large negative
    # bias so they never win (safe abstain).
    W = np.zeros((len(classes), dim), dtype="float32")
    b = np.full(len(classes), -1e3, dtype="float32")
    coef = clf.coef_
    inter = clf.intercept_
    if coef.shape[0] == 1:  # binary edge case → expand to 2 rows
        coef = np.vstack([-coef[0], coef[0]])
        inter = np.array([-inter[0], inter[0]])
    for row, ci in enumerate(used_classes):
        W[ci] = coef[row].astype("float32")
        b[ci] = float(inter[row])

    # Calibrate temperature on the validation split (raw logits = W·x + b).
    val_logits = X[val] @ W.T + b
    T, nll = fit_temperature(val_logits, y_idx[val])
    print(f"temperature T={T:.3f}  val_nll={nll:.4f}")

    # Report cold-start REPRODUCTION: does the head (argmax over calibrated probs,
    # with the serve-time threshold) reproduce the keyword+gazetteer label on the
    # queries where the deterministic detector fires? (These queries are ALSO caught
    # by keyword/gazetteer at serve time, which take precedence — so exact head
    # agreement here is a training-fit diagnostic, not the do-no-harm guarantee.)
    def predict(logits, thr):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p = p / p.sum(axis=1, keepdims=True)
        out = []
        for row in p:
            j = int(row.argmax())
            out.append(NONE if (classes[j] == NONE or row[j] < thr) else classes[j])
        return out

    all_logits = X @ W.T + b
    pred = np.array(predict(all_logits, args.threshold))
    scoped_mask = y != NONE
    if scoped_mask.sum():
        repro = float((pred[scoped_mask] == y[scoped_mask]).mean())
    else:
        repro = float("nan")
    # abstention on the deterministic-_NONE_ queries (the head should mostly abstain).
    none_mask = y == NONE
    abstain_rate = float((pred[none_mask] == NONE).mean()) if none_mask.sum() else float("nan")
    print(f"scoped-label reproduction (head argmax@thr == det label): {repro:.3f} "
          f"(over {int(scoped_mask.sum())} det-scoped queries)")
    print(f"abstention on det-_NONE_ queries: {abstain_rate:.3f} "
          f"(over {int(none_mask.sum())} queries)")

    # Merge the classifier entry into qhead.json (leave every other knob untouched).
    spec = json.load(open(args.qhead))
    spec.setdefault("knobs", {})["country_scope"] = {
        "driver": "head", "kind": "classifier", "enabled": True,
        "classes": classes, "none_class": NONE,
        "temperature": round(T, 5), "threshold": args.threshold,
        "input_dim": dim,
        "W": W.tolist(), "b": b.tolist(),
        "train": {"n": int(n), "C": args.C, "val_frac": args.val_frac,
                  "label_dist": dist,
                  "scoped_reproduction": round(repro, 4),
                  "none_abstention": round(abstain_rate, 4)},
        "note": ("ADR-062 knob 13 (amended Q2): learned calibrated abstaining soft "
                 "country scope. Cold-started from _query_jurisdiction + gazetteer; "
                 "generalises to markerless forms. Serve-time precedence keyword > "
                 "gazetteer > this head; confidence scales the off-country penalty."),
    }
    spec.setdefault("phase", "")
    spec["phase_2b"] = "country_scope-learned"
    json.dump(spec, open(args.out, "w"))
    # human-readable sidecar (no giant W)
    rep = {k: v for k, v in spec["knobs"]["country_scope"].items() if k not in ("W", "b")}
    json.dump({"country_scope": rep},
              open(report_path, "w"), indent=1)
    print(f"\nwrote country_scope classifier into {args.out} "
          f"(C={len(classes)} classes, dim={dim})")
    print("classes:", classes)
    return 0


def run(argv=None, prog=None):
    """`main` with the contract's refusals reported as exit code 2."""
    try:
        return main(argv, prog=prog)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
