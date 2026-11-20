"""`cytir.ops.serve_split_map`: the S10 design (every serving def -> its target module) stays in step
with the code, and the spec's generated def table is current (task 1061)."""
from __future__ import annotations

import pathlib

import pytest

from cytir.ops import serve_split_map as m

APP = pathlib.Path(__file__).resolve().parent.parent


def test_every_serving_def_has_a_target_and_every_target_a_def():
    rows, problems = m.def_map()
    assert problems == []
    assert {r["status"] for r in rows} <= {"done", "pending"}
    assert all(r["target"] in m.LAYERS for r in rows)


def test_the_spec_def_table_is_current():
    spec = APP / "specs" / "s10-serve-split.md"
    if not spec.is_file():
        pytest.skip("the S10 spec is a maintainer document; the public cut does not ship it")
    rows, _ = m.def_map()
    assert m.defs_current(spec, rows), \
        "run: python -m cytir.ops.serve_split_map --write specs/s10-serve-split.md"


def test_a_new_def_without_a_target_is_reported_and_a_moved_def_is_done(tmp_path):
    srv = tmp_path / "serve"
    srv.mkdir()
    (srv / "retrieval.py").write_text("def search():\n    pass\n\ndef finish():\n    pass\n\n"
                                      "def _brand_new():\n    pass\n")
    (srv / "textmatch.py").write_text("def _content_terms(text, lang=None):\n    return set()\n")
    (srv / "test_x.py").write_text("def _ignored():\n    pass\n")
    rows, problems = m.def_map(srv)
    assert "unmapped def in retrieval.py: _brand_new" in problems
    by = {r["name"]: r for r in rows}
    assert by["_content_terms"]["status"] == "done"
    assert by["finish"]["status"] == "pending" and by["finish"]["target"] == "respond"
    assert "_ignored" not in by


def test_knob_reads_count_strings_not_comments(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text('import os\n# AW_IN_A_COMMENT is not read\n'
                 'X = float(os.environ.get("AW_REAL", WEIGHTS.get("real_key", 0.0)))\n'
                 'Y = os.environ.get("AW_PLAIN", "0") == "1"\n')
    assert m._string_knobs(f) == {"AW_REAL", "AW_PLAIN"}
    assert m._weights_default_knobs(f) == {"AW_REAL": "real_key"}


def test_write_then_check_round_trips(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text(f"# t\n\n{m.BEGIN}\n{m.END}\n\ntail\n")
    rows, _ = m.def_map()
    assert not m.defs_current(spec, rows)
    m.write_block(spec, m.defs_markdown(rows) + "\n\nknobs")
    assert m.defs_current(spec, rows)
    assert spec.read_text().endswith("tail\n")


def test_the_aw_ops_door_runs_the_check(tmp_path):
    from cytir import cli
    spec = tmp_path / "spec.md"
    spec.write_text(f"{m.BEGIN}\n{m.END}\n")
    assert cli._ops_main(["serve-split-map", "--check", str(spec)]) == 1      # stale
    m.write_block(spec, m.defs_markdown(m.def_map()[0]))
    assert cli._ops_main(["serve-split-map", "--check", str(spec)]) == 0


def test_frozen_echo_skips_only_inside_the_settings_frozen_block(tmp_path):
    other = tmp_path / "mod.py"
    other.write_text('X = os.environ.get("AW_HIDDEN", "0")  # frozen-echo\n')
    assert m._string_knobs(other) == {"AW_HIDDEN"}          # tag elsewhere: still a read
    st = tmp_path / "settings.py"
    st.write_text('FROZEN = {\n    "AW_ECHO": (1, "r.x"),  # frozen-echo\n}\n'
                  'Y = os.environ.get("AW_SNEAK", "0")  # frozen-echo\n')
    assert m._string_knobs(st) == {"AW_SNEAK"}               # skipped in FROZEN only
