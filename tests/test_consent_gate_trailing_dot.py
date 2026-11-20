"""Task 824 (audit 818 B35): a row whose URL host ends in the FQDN dot is gated under the dot-free host.

The entry is the one the task 642 sweep roster carried: `tilman-rossmy.de.` (the 642 sweep task doc, section 5). The
resolver keys its verdicts and cache files by the dot-free host; the gate looked them up by the dotted one
and filed the row as `unknown`. The test runs the REAL `resolve_hosts` over a cache holding the host's
receipt (no network)."""
import json
import types
from datetime import datetime, timezone

import cytir.consent.gate as CG

HOST = "tilman-rossmy.de."          # as it stood in the 642 roster


def _receipt():
    return {"host": "tilman-rossmy.de", "status": "ok", "search_ok": True, "train_ok": True,
            "reasons": ["no opt-out signal found"], "signals": {}, "mode": "cc",
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_host_of_drops_the_trailing_dot_and_nothing_else():
    assert CG._host_of(f"https://{HOST}/seite") == "tilman-rossmy.de"
    assert CG._host_of(f"https://WWW.{HOST}:8443/x") == "www.tilman-rossmy.de"
    assert CG._host_of("https://tilman-rossmy.de/seite") == "tilman-rossmy.de"
    assert CG._host_of("") == "" and CG._host_of("notaurl") == ""


def test_dotted_row_is_gated_by_its_hosts_verdict(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "tilman-rossmy.de.json").write_text(json.dumps(_receipt()))   # the key the 642 sweep wrote
    rows = [{"url": f"https://{HOST}/a", "text": "a"}, {"url": "https://tilman-rossmy.de/b", "text": "b"}]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    CG.cmd_gate(types.SimpleNamespace(passages=str(tmp_path / "p.jsonl"), state="", cache_dir=str(cache),
                                      out_dir=str(out), concurrency=1, direct_crawl=False, corpus_wave="",
                                      accept_search_grant=False, search_indexing=False, unclear_register=""))
    ok = [json.loads(x)["url"] for x in (out / "passages_ok.jsonl").read_text().splitlines()]
    assert ok == [f"https://{HOST}/a", "https://tilman-rossmy.de/b"]
    unk = out / "passages_unknown.jsonl"
    assert not unk.exists() or not unk.read_text()
