import json
import gzip
from pathlib import Path

from cytir.ops.audits import openalex_academic_pilot as subject


def test_profile_methods_match_every_academic_dump_requirement():
    registry = json.loads((Path(__file__).parents[1]
                           / "data/source-preservation-profiles.v1.json").read_text())
    profile = registry["profiles"]["academic_dump"]
    methods = subject.build_profile_methods(
        selected=259, selected_parts=1, parents=258, chunks=258,
        external_links=18613,
        empty=0, token_max=1813)
    assert set(methods) == set(profile["required_stages"])
    for stage in profile["required_stages"]:
        assert len(methods[stage]) == len(profile.get(stage, {}).get("requirements", []))
    assert "source relation/JSON-path context" in methods["outlinks"][0]
    assert "explicit-empty" in methods["outlinks"][2]


def test_profile_methods_report_multipart_tranche_truthfully():
    methods = subject.build_profile_methods(
        selected=105882, selected_parts=4, parents=105876, chunks=105900,
        external_links=10986416, empty=0, token_max=2560)
    assert "4 selected whole parquet part(s)" in methods["original"][0]
    assert "pilot" not in " ".join(methods["original"])


def test_streams_three_ordered_shards_and_one_recovery_without_reordering(tmp_path):
    artifacts = []
    rows = [
        [{"source_record_id": "W1"}, {"source_record_id": "W2"}],
        [{"source_record_id": "W3"}],
        [{"source_record_id": "W4", "status": "recovery_required"}],
    ]
    for index, values in enumerate(rows):
        path = tmp_path / f"shard-{index:04d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
        artifacts.append({"uri": f"artifact://{path.name}"})
    streamed = list(subject._iter_receipt_rows(
        tmp_path, {"artifacts": artifacts}))
    assert [row["source_record_id"] for row in streamed] == [
        "W1", "W2", "W3", "W4"]
    assert streamed[-1]["status"] == "recovery_required"


def test_abstract_fidelity_accepts_only_renderer_boundary_whitespace_normalization():
    source = "A full source-declared abstract with exact internal spacing. "
    markdown = "# Title\n\n## Abstract\n\n" + source.rstrip() + "\n"
    assert subject.exact_abstract_section(markdown) == source.rstrip()
    assert subject.abstract_fidelity_matches(source, markdown)

    corrupted = markdown.replace("exact internal spacing", "changed internal spacing")
    assert not subject.abstract_fidelity_matches(source, corrupted)
    assert not subject.abstract_fidelity_matches(
        source, "# Title\n\nThe abstract appears outside its required section.\n")


def test_abstract_fidelity_preserves_heading_like_text_inside_source_abstract():
    source = (
        "This paper introduces the subject.\n\n"
        "## Abstract\n\n"
        "This nested heading and the following technical content are source text."
    )
    markdown = "# Title\n\n## Abstract\n\n" + source + "\n"

    assert subject.abstract_fidelity_matches(source, markdown)
    assert subject.exact_abstract_section(markdown) == source
    assert not subject.abstract_fidelity_matches(
        source, markdown.replace("technical content", "changed content"))
