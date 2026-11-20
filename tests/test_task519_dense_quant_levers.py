"""Task 519 unit tests — the dense-leg quantizer policy is a FLAG, and its
default is exactly what shipped before it existed.

Per verification-doctrine 1 these are triage, not coverage: the ranking effect
of the quantizer is measured by `gate_authority` on 1,418 queries. What a gate
cannot see, and what these tests exist for, is the thing that made task 516's
`-36 golds` uninterpretable:

  1. the DEFAULT dense search params are PQ codes with rescore OFF and
     oversampling 1.0 — so a gate leg that sets no flag does NOT measure the
     ratified serving config (the serving config module: rescore=True,
     oversampling=3.0). A test that pins the default is how the next reader
     learns which arm an un-flagged leg actually ran;
  2. each of the three levers moves exactly one field of
     `QuantizationSearchParams` and nothing else;
  3. the EXACT arm uses `quantization.ignore`, never `SearchParams.exact` —
     `exact=True` is a brute-force full scan over 31M points, i.e. a different
     retrieval algorithm, and would not be a one-variable arm.

Run: python3 -m pytest tests/test_task519_dense_quant_levers.py
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

_APP = pathlib.Path(__file__).resolve().parent.parent
_PROBE = (
    "from cytir.serve import retrieval as r;"
    "sp = r._DENSE_SP;"
    "import json;"
    "print(json.dumps({'hnsw_ef': sp.hnsw_ef, 'exact': sp.exact,"
    " 'ignore': sp.quantization.ignore, 'rescore': sp.quantization.rescore,"
    " 'oversampling': sp.quantization.oversampling}))"
)


def _interpreter_can_import_the_server():
    """The probe needs `qdrant_client`. Refuse LOUDLY rather than pass vacuously
    if this interpreter cannot import the module under test."""
    return subprocess.run(
        [sys.executable, "-c", "import qdrant_client"],
        capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _interpreter_can_import_the_server(),
    reason=f"{sys.executable} cannot import qdrant_client — run these under an "
           "interpreter that has the serving deps (e.g. the retune venv), or "
           "they would pass by not running")


def _dense_sp(**flags):
    """Import `retrieval` in a FRESH interpreter under `flags` and read
    `_DENSE_SP`. The module reads its env at import time, so a same-process
    monkeypatch would test a value the server never uses."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AW_DENSE_", "AW_QDRANT", "AW_COLLECTION"))}
    env["PYTHONPATH"] = str(_APP / "server")
    env.update({k: str(v) for k, v in flags.items()})
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_is_pq_codes_rescore_off_oversampling_one():
    """The un-flagged default — what task 516's gate legs actually ran."""
    sp = _dense_sp()
    assert sp["ignore"] is False
    assert sp["rescore"] is False
    assert sp["oversampling"] == 1.0
    assert sp["exact"] is False
    assert sp["hnsw_ef"] is None


def test_ratified_arm_is_rescore_on_oversampling_three():
    sp = _dense_sp(AW_DENSE_RESCORE="1", AW_DENSE_OVERSAMPLING="3.0")
    assert sp["rescore"] is True
    assert sp["oversampling"] == 3.0
    assert sp["ignore"] is False          # still the codes, then rescored


def test_exact_arm_ignores_the_quantizer_and_is_not_a_full_scan():
    sp = _dense_sp(AW_DENSE_IGNORE_QUANT="1")
    assert sp["ignore"] is True
    assert sp["exact"] is False, (
        "the exact ARM must stay an HNSW search scored on the stored "
        "originals; SearchParams.exact=True is a brute-force scan and would "
        "change the retrieval algorithm, not the quantizer")


def test_each_lever_moves_exactly_one_field():
    base = _dense_sp()
    for flag, field, value in (("AW_DENSE_IGNORE_QUANT", "ignore", True),
                               ("AW_DENSE_RESCORE", "rescore", True)):
        sp = _dense_sp(**{flag: "1"})
        assert sp[field] == value
        assert {k: v for k, v in sp.items() if k != field} == \
               {k: v for k, v in base.items() if k != field}


def test_hnsw_ef_lever_is_independent_of_the_quantizer_levers():
    """Prod sets AW_DENSE_HNSW_EF=384; the quantizer arms must not disturb it."""
    sp = _dense_sp(AW_DENSE_HNSW_EF="384", AW_DENSE_IGNORE_QUANT="1")
    assert sp["hnsw_ef"] == 384
    assert sp["ignore"] is True
