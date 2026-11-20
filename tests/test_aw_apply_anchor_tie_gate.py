"""Task 933 (T25): the apply gate's content-pack call passes --anchor-tie; a tie is reported, never a loss."""
import json
import subprocess
import textwrap
from pathlib import Path

from cytir.index import apply as mod

DATA = Path(__file__).parent / "data"
ROWS = json.loads((DATA / "probe_anchor_tie_fedlex_rows.json").read_text())
SPEC = DATA / "probe_anchor_tie_art257_de_spec.json"

# Stand-in comparator: the REAL probe guard and printer, fed the art257-de readings by rows key.
STUB = textwrap.dedent('''
    import json, sys
    from cytir.gates import compare_corpus_arms as cca
    rows = json.load(open(sys.argv[1])); spec = json.load(open(sys.argv[2])); new_key = sys.argv[3]
    specs = {pid: {"id": pid} for pid in cca.PROBES}; specs["art257-de"] = spec
    cca.probe_specs = lambda path=None: specs
    def doc(urls):
        d = {pid: {"spec": specs[pid], "urls": ["https://x.example/a"], "rank": 1} for pid in cca.PROBES}
        rank = next((i for i, u in enumerate(urls, 1) if u == rows["target"]), None)
        d["art257-de"] = {"spec": spec, "urls": urls, "rank": rank}
        return {"probes": d}
    try:
        cca.print_anchor_ties(cca.probe_non_regression(
            doc(rows["p0"]), doc(rows[new_key]), allow_anchor_tie="--anchor-tie" in sys.argv))
    except cca.EvidenceError as exc:
        print(exc); sys.exit(3)
''')


def run_gate(tmp_path, new_key):
    stub = tmp_path / "stub.py"; stub.write_text(STUB)
    argv = mod.content_gate_argv(stub, "BASE", "CAND", [])
    argv = [argv[0], argv[1], str(DATA / "probe_anchor_tie_fedlex_rows.json"), str(SPEC), new_key,
            *[a for a in argv[4:] if a.startswith("--")]]
    root = Path(__file__).resolve().parent.parent
    return subprocess.run(argv, text=True, capture_output=True, cwd=root, env={"PYTHONPATH": str(root), "PATH": ""})


def test_gate_call_site_passes_anchor_tie_and_keeps_the_rest():
    argv = mod.content_gate_argv("c.py", "b", "k", ["--null-arm", "n"])
    assert "--anchor-tie" in argv and argv[-2:] == ["--null-arm", "n"] and "content-pack" in argv


def test_art257_tie_passes_and_is_reported(tmp_path):
    r = run_gate(tmp_path, "p1_swap")
    assert r.returncode == 0, r.stdout + r.stderr
    ties = mod.parse_anchor_ties(r.stdout)
    assert len(ties) == 1 and ties[0]["probe"] == "art257-de"
    assert (ties[0]["gold_rank"], ties[0]["found_rank"]) == (5, 6)
    a, b = ties[0]["old_anchors"], ties[0]["new_anchors"]
    assert a != b and sorted(a) == sorted(b)


def test_other_document_in_the_slot_still_exits_3(tmp_path):
    r = run_gate(tmp_path, "p1_other_doc_enters")
    assert r.returncode == 3 and "regressed 5->6" in r.stdout
    assert mod.parse_anchor_ties(r.stdout) == []


def test_no_tie_output_parses_to_empty():
    assert mod.parse_anchor_ties("fine\n") == []
