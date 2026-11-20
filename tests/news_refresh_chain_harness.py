"""Parity harness for the news stage / delta steps (task 1989): recorder interpreters, a stub
`qdrant_client`, a loopback twin, and one scenario per step.

A step is run against FAKE interpreters that record every command line (argv, a few env names,
stdin hash) and, where a later step reads a file the real tool would have written, write a small
deterministic one. The inline scripts, the `-c` one-liner and `cytir.news.refresh_chain` run for real
(with the stub client). No network (the twin is a stdlib server on 127.0.0.1), no Qdrant, no box path.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import pathlib
import sys
import threading

APP = pathlib.Path(__file__).resolve().parents[1]
DAY = "2026-01-02"

_RECORDER = '''#!{py}
import hashlib, json, os, subprocess, sys
name = os.path.basename(sys.argv[0])
argv = sys.argv[1:]
real = {py!r}
if name == "python3" and argv[:2] in (["-m", "cytir.core.paths"], ["-m", "cytir.news.source_queue"]):
    os.execv(real, [real] + argv)
stdin = sys.stdin.read() if "-" in argv else None
keys = ("PYTHONPATH", "AW_INPROC_EXTRACT", "AW_FOLD_ACCENTS", "AW_PIPE", "PYTHONDONTWRITEBYTECODE")
rec = {{"tool": name, "argv": argv,
       "env": {{k: os.environ.get(k) for k in keys}},
       "stdin_sha": hashlib.sha256(stdin.encode()).hexdigest() if stdin is not None else None,
       "stdin": stdin}}
with open(os.environ["AW_TEST_CALL_LOG"], "a") as fh:
    fh.write(json.dumps(rec) + "\\n")

def opt(flag):
    return argv[argv.index(flag) + 1]

real_env = dict(os.environ)
real_env["PYTHONPATH"] = os.pathsep.join(p for p in (os.environ.get("PYTHONPATH"), os.environ["AW_TEST_STUB"],
                                                      os.environ["AW_TEST_REPO"]) if p)
if "-" in argv or argv[:1] == ["-c"] or argv[:3] == ["-m", "cytir.news.refresh_chain", "manifest"]:
    r = subprocess.run([real] + argv, input=stdin, text=True, env=real_env)
    sys.exit(r.returncode)
if argv[:2] == ["-m", "cytir.index.dump_netnew"]:
    out = opt("--out")
    open(out, "wb").write(b"gz:" + opt("--staging-collection").encode())
    open(out + ".count", "w").write(json.dumps({{"changed": len(opt("--staging-collection")) % 5 + 1}}))
elif argv[:2] == ["-m", "cytir.news.dump_points"]:
    for flag in ("--out", "--successor-candidates-out", "--receipt-out"):
        open(opt(flag), "w").write(flag + "\\n")
elif argv[:2] == ["-m", "cytir.index.news_window_filter"]:
    open(opt("--delete-ids-out"), "w").write("p1\\np2\\n")
elif "routes" in argv:
    print(json.dumps({{"srf.ch": "chmedia", "swissinfo.ch": "swissinfo"}}))
elif "status-hosts" in argv:
    print("srf.ch swissinfo.ch")
elif "collection-name" in argv:
    print("coll-" + opt("--label"))
'''

_STUB_CLIENT = '''
class _Info:
    def __init__(self, n):
        self.points_count = n
        self.status = "CollectionStatus.GREEN"

class QdrantClient:
    def __init__(self, url=None, timeout=None, check_compatibility=None):
        self.url = url
    def get_collection(self, name):
        return _Info(7 + len(name))
'''


class Twin:
    """A loopback stand-in for the canonical twin's `GET /collections/<name>`."""

    def __init__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # a body the real qdrant client parses (task 2069: write_baseline uses it)
                body = json.dumps({"status": "ok", "result": {
                    "status": "green", "optimizer_status": "ok", "points_count": 41,
                    "segments_count": 1, "payload_schema": {},
                    "config": {"params": {"vectors": {"size": 4, "distance": "Cosine"}},
                               "hnsw_config": {"m": 16, "ef_construct": 100,
                                               "full_scan_threshold": 10000},
                               "optimizer_config": {"deleted_threshold": 0.2,
                                                    "vacuum_min_vector_number": 1000,
                                                    "default_segment_number": 0,
                                                    "flush_interval_sec": 5}}}})
                if self.path != "/collections/twin_c":
                    body = json.dumps({"status": {"error": "gone"}})   # real Qdrant: 404
                self.send_response(200 if self.path == "/collections/twin_c" else 404)
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *a):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        outer.url = self.url

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _fakes(root: pathlib.Path) -> pathlib.Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("python3", "lg", "qpy"):
        f = bin_dir / name
        f.write_text(_RECORDER.format(py=sys.executable))
        f.chmod(0o755)
    stub = root / "stub" / "qdrant_client"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "__init__.py").write_text(_STUB_CLIENT)
    return bin_dir


def hand_manifest(spool: pathlib.Path, arms: dict) -> pathlib.Path:
    """A `stage_manifest.json` built with the ORIGINAL recipe (hash of the body, key added last)."""
    body = {"schema": "aw.news-stage.v2", "day": DAY, "mode": "daily",
            "representation": "aw_b1-native", "arms": arms, "source_status": {"hn": {"status": "complete"}},
            "mrl_dim": 128}
    payload = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    body["manifest_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    path = spool / "stage_manifest.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


class Scenario:
    """One fixed layout under `root`; `reset()` clears what a run wrote so the same layout can be
    run twice (script, then Python) and the receipts compared as exact bytes."""

    def __init__(self, kind: str, root: pathlib.Path, twin: Twin | None = None):
        self.kind, self.root, self.twin = kind, root, twin
        self.bin = _fakes(root)
        self.work = root / "work"
        self.spool = self.work / f"stage-{DAY}"
        self.delta = root / "delta"
        self.log = root / "calls.jsonl"
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(root),
            "AW_TEST_CALL_LOG": str(self.log), "AW_TEST_OVERLAY_LOOKUP": "",
            "AW_TEST_STUB": str(root / "stub"), "AW_TEST_REPO": str(APP),
            "AW_NOW": DAY, "AW_MODE": "daily", "AW_RELEASE_DIR": str(APP),
            "AW_NEWS_RELEASE_MANIFEST": str(root / "m.json"),
            "AW_NEWS_RELEASE_MANIFEST_SHA256": "a" * 64, "AW_NEWS_RELEASE_SHA": "b" * 40,
            "AW_NEWS_LANGGATE_REQUIREMENTS_SHA256": "c" * 64,
            "AW_NEWS_WORK": str(self.work), "AW_NEWS_LANGGATE_PY": str(self.bin / "lg"),
            "AW_QPY": str(self.bin / "qpy"), "AW_NEWS_EXPECTED_HOSTS": "www.srf.ch",
            "AW_NEWS_TWIN_URL": twin.url if twin else "http://twin.invalid:1",
            "AW_NEWS_TWIN_COLLECTION": "twin_c",
        }
        if kind == "delta":
            self.env.update({
                "DELTA_DIR": str(self.delta), "AW_NEWS_RETAIN_MONTHS": "36",
                "AW_NEWS_POINT_CANDIDATES": str(root / "cand.jsonl"),
                "AW_NEWS_POINT_CANDIDATES_SHA256": "d" * 64,
                "AW_STAGE_MANIFEST": str(self.spool / "stage_manifest.json"),
            })
        self.prepare()

    def prepare(self):
        self.spool.mkdir(parents=True, exist_ok=True)
        if self.kind == "stage":
            sp = self.spool
            (sp / "news_rows.enriched.jsonl").write_text('{"a": 1}\n\n{"a": 2}\n')
            (sp / "hn_rows.enriched.jsonl").write_text('{"h": 1}\n')
            (sp / "state.json").write_text(json.dumps(
                {"hosts": {"srf.ch": {"status": "ok", "note": "n"}}}))
            (sp / "hn_source_receipt.json").write_text(json.dumps({"completeness": "time-window-complete"}))
            (sp / "gated").mkdir(exist_ok=True)
            rows = [{"domain": "www.srf.ch", "u": 1}, {"domain": "www.swissinfo.ch", "u": 2},
                    {"domain": "news.ycombinator.com", "u": 3}, {"domain": "www.srf.ch", "u": 4}]
            (sp / "gated" / "passages_ok.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self._inputs = {p.name for p in sp.rglob("*") if p.is_file()}
        else:
            hand_manifest(self.spool, {
                "chmedia": {"collection": "news_chmedia_x", "points_count": 5, "status": "green"},
                "hn": {"collection": "news_hn_x", "points_count": 3, "status": "green"}})
            (self.root / "cand.jsonl").write_text("{}\n")
            self._inputs = {"stage_manifest.json"}

    def reset(self):
        import shutil
        if self.log.exists():
            self.log.unlink()
        if self.kind == "stage":
            for p in list(self.spool.rglob("*")):
                if p.is_file() and str(p.relative_to(self.spool)) not in {
                        "news_rows.enriched.jsonl", "hn_rows.enriched.jsonl", "state.json",
                        "hn_source_receipt.json", "gated/passages_ok.jsonl"}:
                    p.unlink()
        else:
            shutil.rmtree(self.delta, ignore_errors=True)

    def receipt_files(self) -> list[pathlib.Path]:
        if self.kind == "stage":
            return [self.spool / n for n in ("stage_input_manifest.json", "stage_manifest.json",
                                             "all.enriched.jsonl", "gated_chmedia.jsonl",
                                             "gated_swissinfo.jsonl", "gated_hn.jsonl")]
        return sorted(p for p in self.delta.iterdir() if p.is_file())

    def snapshot(self) -> dict[str, bytes]:
        base = self.spool if self.kind == "stage" else self.delta
        return {str(p.relative_to(base)): p.read_bytes() for p in self.receipt_files()}

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text().splitlines():
            rec = json.loads(line)
            rec["stdin_text"] = rec.pop("stdin")
            pp = rec["env"].get("PYTHONPATH")
            if pp:      # the conftest's subprocess guard rides on PYTHONPATH: not the step's
                kept = [x for x in pp.split(os.pathsep) if pathlib.Path(x).name != "live_port_site"]
                rec["env"]["PYTHONPATH"] = os.pathsep.join(kept) or None
            out.append(rec)
        return out

    def normalize(self, value):
        subs = [(str(APP), "<REPO>"), (str(self.root), "<TMP>")]
        if self.twin:
            subs.append((self.twin.url, "<TWIN>"))
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            for src, dst in sorted(subs, key=lambda p: -len(p[0])):
                value = value.replace(src, dst)
            return value
        if isinstance(value, list):
            return [self.normalize(v) for v in value]
        if isinstance(value, dict):
            return {k: self.normalize(v) for k, v in value.items()}
        return value

    def digests(self, snap: dict[str, bytes]) -> dict[str, str]:
        """sha256 per receipt, paths normalized. `stage_manifest.json` carries a hash of its OWN
        bytes (paths included), so it is re-hashed after normalizing, by the original recipe."""
        out = {}
        for name, raw in sorted(snap.items()):
            text = self.normalize(raw)
            if name == "stage_manifest.json":
                body = json.loads(text)
                body.pop("manifest_payload_sha256")
                body["manifest_payload_sha256"] = hashlib.sha256(
                    (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
                text = json.dumps(body, indent=2, sort_keys=True) + "\n"
            out[name] = hashlib.sha256(text.encode()).hexdigest()
        return out

    def comparable_calls(self) -> list[dict]:
        """The calls a port must reproduce: argv, env names, tool. The inline-script starts
        (stdin / `-c`) and the `manifest` start are the in-process edits and are compared apart."""
        out = []
        for rec in self.calls():
            argv = rec["argv"]
            if "-" in argv or argv[:1] == ["-c"] or argv[:3] == ["-m", "cytir.news.refresh_chain", "manifest"]:
                continue
            out.append(self.normalize({"tool": rec["tool"], "argv": argv, "env": rec["env"]}))
        return out

    def inline_calls(self) -> list[dict]:
        return [self.normalize({"tool": r["tool"], "argv": r["argv"]}) for r in self.calls()
                if "-" in r["argv"] or r["argv"][:1] == ["-c"]
                or r["argv"][:3] == ["-m", "cytir.news.refresh_chain", "manifest"]]

    # ---- runs
    def run_script(self, script: pathlib.Path) -> tuple[int, str]:
        import subprocess
        r = subprocess.run(["bash", str(script)], env=self.env, cwd=self.root, capture_output=True,
                           text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def run_python(self) -> tuple[int, str]:
        from cytir.news import refresh_chain as RC
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = (RC.run_stage if self.kind == "stage" else RC.run_delta)(dict(self.env))
        return rc, buf.getvalue()
