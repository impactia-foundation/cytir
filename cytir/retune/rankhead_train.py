"""Train the ADR-036 trust head (`AW_RANKHEAD`, variant v2) -- `cyt retune rankhead-train`. EMIT-ONLY.

Origin: moved, not rewritten, from the retired task-222 and task-206 exploration scripts (frozen in the private tree),
the task-222 script being the 206 script with one more training source. Left behind on purpose: the original's closing "routing probes" (they encode a
handful of queries with the GPU embedder and only print); the head file below is what they wrote as the
deployable artefact.

Mechanism: a binary logistic regression on the FROZEN 1024-d arctic query vector, P(general intent). The served
beta is `beta_gen * P` (heads.py). Three training sources, each a `.npy` of frozen query vectors plus a labels
file with the same number of rows:

  synth  (task 204)  JSONL, one `{"intent_hint": ...}` per row; general iff hint in factual_lookup/definition/howto
  real   (task 206)  JSONL, `{"intent": "general"|..., "flood_prone": bool?}`; flood-prone general rows get --flood-weight
  neg    (task 222)  JSONL, one row per vector; ALL general, up-weighted by --neg-weight

The output is a NEW head file in the schema the server loads (`cytir.serve.rankhead_schema`). The trainer refuses to
write onto an existing file or onto any head the server would load; promotion is a Tim PR (ruling 27 T4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENERAL_HINTS = frozenset({"factual_lookup", "definition", "howto"})
SPLIT_SEED = 222   # the original's train_test_split random_state; kept so the validation split is the same rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def load_sources(synth_vecs, synth_meta, real_vecs, real_labels, neg_vecs, neg_labels, *,
                 flood_weight: float, neg_weight: float):
    """Merge the three sources into `(X, y, w, src)`; a vector/label row-count mismatch is refused."""
    import numpy as np

    def one(vec_path, label_path, name):
        x = np.load(vec_path)
        rows = _rows(label_path)
        if len(rows) != x.shape[0]:
            raise ValueError(f"{name}: {len(rows)} label rows vs {x.shape[0]} vectors")
        return x, rows

    xs, meta = one(synth_vecs, synth_meta, "synth")
    ys = np.array([1 if m["intent_hint"] in GENERAL_HINTS else 0 for m in meta])
    xr, real = one(real_vecs, real_labels, "real")
    yr = np.array([1 if r["intent"] == "general" else 0 for r in real])
    wr = np.array([flood_weight if (r["intent"] == "general" and r.get("flood_prone")) else 1.0
                   for r in real], dtype="float32")
    xn, neg = one(neg_vecs, neg_labels, "neg")
    if not (xs.shape[1] == xr.shape[1] == xn.shape[1]):
        raise ValueError("the three vector files differ in dimension")
    X = np.vstack([xs, xr, xn])
    y = np.concatenate([ys, yr, np.ones(len(neg), dtype=int)])
    w = np.concatenate([np.ones(len(ys), dtype="float32"), wr, np.full(len(neg), neg_weight, dtype="float32")])
    src = np.array(["synth"] * len(ys) + ["real"] * len(yr) + ["neg"] * len(neg))
    return X, y, w, src


def train(X, y, w, src, *, neg_weight: float, beta_gen: float) -> tuple[dict, dict]:
    """Fit the head. Returns `(head, audit)`; `head` is the deployable file, `audit` the validation numbers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte, wtr, _wte, _str, ste = train_test_split(
        X, y, w, src, test_size=0.10, random_state=SPLIT_SEED, stratify=y)
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr, sample_weight=wtr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred))
    per_source = {s: {"n": int((ste == s).sum()), "acc": float(accuracy_score(yte[ste == s], pred[ste == s]))}
                  for s in ("synth", "real", "neg") if (ste == s).any()}
    wv = clf.coef_[0].astype("float32")
    bv = float(clf.intercept_[0])
    comment = ("ADR-036 rank-head (V2 soft-interp) trained by `aw retune rankhead-train`: logistic on the frozen "  # rename-codemod: keep (persisted: comment written into the rank-head model file)
               f"query vector, neg weight {neg_weight}. beta = beta_gen*P(general).")
    head = {"_comment": comment, "variant": "v2", "beta_gen": beta_gen, "beta_max": 0.9,
            "trust_scale": 1.0, "w": wv.tolist(), "b": bv}
    audit = {"val_acc": acc, "val_f1": f1, "n_train": int(len(ytr)), "neg_weight": neg_weight,
             "n_by_source": {s: int((src == s).sum()) for s in ("synth", "real", "neg")}, "val_by_source": per_source}
    return head, audit


def _served_heads() -> set[Path]:
    from cytir.serve.settings import SETTINGS
    here = Path(__file__).resolve().parents[1] / "serve"
    runtime = SETTINGS.runtime
    return {here / "rankhead.json", here / "rankhead_v2.json",
            Path(runtime.rankhead_path).resolve(), Path(runtime.rankhead_v2_path).resolve()}


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    for name in ("synth-vecs", "synth-meta", "real-vecs", "real-labels", "neg-vecs", "neg-labels"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="the NEW head file (never an existing or served path)")
    ap.add_argument("--report", type=Path, help="optional audit JSON: input sha256s and validation numbers")
    ap.add_argument("--neg-weight", type=float, default=2.5)
    ap.add_argument("--flood-weight", type=float, default=2.0)
    ap.add_argument("--beta-gen", type=float, default=0.7)
    args = ap.parse_args(argv)

    outputs = [p for p in (args.out, args.report) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("out and report must be distinct")
    if args.out.resolve() in _served_heads():
        raise ValueError("refusing to write a head the server loads; emit a candidate and promote it by PR")
    if any(p.exists() for p in outputs):
        raise ValueError("refusing to overwrite an output artifact")
    inputs = {n: getattr(args, n) for n in ("synth_vecs", "synth_meta", "real_vecs", "real_labels",
                                            "neg_vecs", "neg_labels")}
    for name, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"--{name.replace('_', '-')} is not a file: {path}")

    from cytir.serve.rankhead_schema import validate_head

    X, y, w, src = load_sources(*inputs.values(), flood_weight=args.flood_weight, neg_weight=args.neg_weight)
    head, audit = train(X, y, w, src, neg_weight=args.neg_weight, beta_gen=args.beta_gen)
    validate_head(head)
    args.out.write_text(json.dumps(head))
    if args.report:
        audit["inputs_sha256"] = {n: _sha256(p) for n, p in inputs.items()}
        args.report.write_text(json.dumps(audit, indent=1, sort_keys=True))
    print(f"MERGED N={len(y)} {audit['n_by_source']}  val acc={audit['val_acc']:.4f} f1={audit['val_f1']:.4f}"
          f"\nwrote {args.out} (candidate only; promotion is a Tim PR)")
    return 0


def run(argv: list[str] | None = None, prog: str | None = None) -> int:
    """`main` with the contract's refusals reported as exit code 2."""
    try:
        return main(argv, prog=prog)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
