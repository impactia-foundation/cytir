"""The `pack` payload route: receipt discovery, and the declared stage override.

The pack route carries the largest share of the build's payload, and both
behaviours tested here were added because a source that does not resolve is a
source whose rows are silently absent from the collection while every guard
stays green. So each test pins a direction of REFUSAL, not a happy path.

`assembly_binder` imports numpy and qdrant_client at module scope; only the two
pure helpers are needed here, so the module is loaded with those stubbed if the
runtime does not have them (a light box may have numpy but not qdrant_client; a full
build box has both). The functions under test touch neither.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

for _name in ("numpy", "qdrant_client"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)
            sys.modules[f"{_name}.models"] = types.ModuleType(f"{_name}.models")
            setattr(sys.modules[_name], "models", sys.modules[f"{_name}.models"])
            # Marks this as a stand-in, not the real package -- so a test
            # file collected LATER in the same pytest run (module caching
            # means its own stubbing loop never fires once this one has
            # already populated sys.modules) can tell the difference and
            # skip a test that needs the real package rather than fail
            # against a stub too thin for it (task 511, cross-file
            # `test_assembly_binder_adr103.py` collection-order failure).
            sys.modules[_name]._AW_TEST_STUB = True

from cytir.index import assemble_inputs as AI  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("data/assembly/pack-stage-overrides.v1.json")


def test_the_real_override_file_declares_what_the_binder_will_read():
    """A malformed override would fail closed at plan time, in the build."""
    overrides = AI.load_pack_stage_overrides()
    assert set(overrides) == {
        "swisstopo-structured-v3-exact",
        "bfs-pxweb-structured-v2-field-aware",
        "gesetze-historical-direct-v4",
        # task 513b declared the flip-lane source that lives under a different
        # release; it carries the same producer-v1 bare-URI drift, so it needs
        # an override like the other three.
        "service-public-dila-fiche-v4b"}
    for source, row in overrides.items():
        assert row["stage_dir"].startswith("artifact://"), source
        assert int(row["receipt_records"]) > 0, source
        assert int(row["shards"]) > 0, source


def test_overrides_are_keyed_by_the_bare_source_name_not_the_dir_name():
    """The source DIRECTORY is `source=<name>`; the table is keyed by `<name>`.

    Keying the lookup on the directory name silently matched nothing and every
    override stayed unresolved while looking exactly like a correct refusal.
    """
    overrides = AI.load_pack_stage_overrides()
    assert not any(k.startswith("source=") for k in overrides)
    assert "source=gesetze-historical-direct-v4".removeprefix(
        "source=") in overrides


def test_an_absent_override_file_declares_nothing_rather_than_raising(
        tmp_path, monkeypatch):
    """Nothing declared means every bare source stays unresolved -- which is
    the pre-override behaviour, and the safe direction."""
    monkeypatch.setenv("AW_REPO_ROOT", str(tmp_path))
    assert AI.load_pack_stage_overrides() == {}


def test_an_override_file_with_no_overrides_key_is_empty_not_a_crash(
        tmp_path, monkeypatch):
    d = tmp_path / "data" / "assembly"
    d.mkdir(parents=True)
    (d / "pack-stage-overrides.v1.json").write_text(json.dumps({"schema": "x"}))
    monkeypatch.setenv("AW_REPO_ROOT", str(tmp_path))
    assert AI.load_pack_stage_overrides() == {}


@pytest.mark.parametrize("source,records,shards", [
    ("swisstopo-structured-v3-exact", 473022, 10),
    ("bfs-pxweb-structured-v2-field-aware", 7337, 1),
    ("gesetze-historical-direct-v4", 90, 1),
])
def test_each_override_pins_the_counts_it_was_verified_against(
        source, records, shards):
    """The binder re-checks these at plan time and refuses if the receipt has
    moved, so an override cannot outlive the evidence it was granted on."""
    row = AI.load_pack_stage_overrides()[source]
    assert int(row["receipt_records"]) == records
    assert int(row["shards"]) == shards
