"""Test support: the real LEGI/LODA articles through the `loda_exact` family door, and the sealed
stage the drain build tests prove identity against (task 1280; was inside two test modules).

Product logic stays in `cytir.sources.families.loda` and `cytir.sources.run_family`; this file only
stages fixtures and stands in for the embed tokenizer (the family refuses a run without it, task 706;
CI has no `transformers`, and the sample articles are short).
"""
from __future__ import annotations

import gzip
import json
import pathlib
import shutil
import sys
from unittest import mock

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import drain_transaction_harness as H  # noqa: E402

FIX = HERE / "fixtures" / "legi_loda_full"
FAMILY = "loda_exact"
RELEASE = "aw-test-loda-v1"
TODAY = "2026-09-30"


def _family(dump, out, **kw):
    """Run the `loda_exact` family on a LEGI view; the token counter is replaced for the call."""
    from cytir.sources import run_family
    from cytir.sources.families import loda
    with mock.patch.object(loda, "embed_token_counter", lambda: (lambda text: len(text) // 3)):
        return run_family.run_family(
            FAMILY, out_dir=out, today=TODAY,
            consent=lambda h, d: pytest.fail(f"consent for {h}"),
            options={"root": str(dump), "release": RELEASE}, log=lambda m: None, **kw)


def vectors_for(stage: pathlib.Path) -> None:
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        ids = [json.loads(x)["point_id"] for x in fh]
    H.write_vectors(stage / "vectors", ids)


def loda_stage(tmp: pathlib.Path) -> pathlib.Path:
    """The three real LEGI articles through the `loda_exact` family door, staged by hand: the
    `family_stage` bridge refuses this family today (contract unknowns), `compose` does not."""
    root = tmp / "loda-view"
    branch = root / "legi" / "global" / "code_et_TNC_en_vigueur" / "TNC_en_vigueur" / "JORF"
    branch.mkdir(parents=True)
    for i, f in enumerate(sorted(FIX.glob("*.xml"))):
        shutil.copy(f, branch / f"LEGIARTI9{i:011d}.xml")
    _family(root, tmp / "loda-run")
    rows = [json.loads(x) for x in
            (tmp / "loda-run" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    stage = tmp / "loda-stage"
    (stage / "chunks").mkdir(parents=True)
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (stage / "PACK.json").write_text(json.dumps(
        {"schema": "aw.family-stage.v1", "family_id": "loda_exact", "pack_id": "loda-exact-v1-shard0",
         "rows": len(rows)}))
    vectors_for(stage)
    return stage
