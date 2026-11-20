"""Task 820: `apply_pack_delta.py` and `promote/apply_delta.py` run the drain's card check first.

The package writers (`cytir.index.apply`, `cytir.index.upsert`) are tested in
`tests/test_aw_index_card_guard_writers.py`; these two live in the private build tree, so their
cases live here (the public-cut leak check refuses a `tests/` module that names them).

The ONE seam is `PSP.retrieve_payloads`, answered per ENDPOINT url: the stored card exists ONLY at
the target's url (`http://t`), never at the pack's (`http://p`). A guard that read the PACK instead
of the target finds no card, and the refusal test fails. The check itself is the real one.

Four cases per writer: stored card + non-card at one id -> refused, nothing written, text names the
writer, the id and the stored url; card over a card -> written; stored non-card -> written; sealed
waiver -> written and logged. MUTATION: removing the `guard_card_overwrites(` block in a writer's
`main` fails that writer's refusal and waiver tests (recorded in task 820's doc).
"""
from __future__ import annotations

import sys as _sys_1822
from pathlib import Path as _Path_1822

_sys_1822.path.insert(0, str(_Path_1822(__file__).resolve().parents[1] / "tests"))  # task 1822: the surface-delta fixtures live in tests/

import gzip
import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cytir.index.pack_delta as APD  # noqa: E402
import test_aw_index_surface_delta as T546  # noqa: E402
from cytir.index import card_waivers as CW  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.index import supersede as US  # noqa: E402

PID = "11111111-0000-5000-8000-00000000000a"
STORED_CARD = {"url": "https://a.ch/", "source_type": "entity_card", "entity_kind": "company",
               "register_id": "CHE-100.000.001", "register_scheme": "CHE-UID", "title": "A AG"}
SHOP = {"url": "https://a.ch/", "source_type": "official", "title": "Shop"}
REFRESHED_CARD = {**STORED_CARD, "title": "A AG (refreshed)"}
STORED_PAGE = {"url": "https://a.ch/", "source_type": "official", "title": "old page"}
WAIVER_PR, WAIVER_SHA = "#820", "c" * 64


def _waiver_file(tmp_path):
    row = {"id": PID, "url": "https://a.ch/", "wrong_card_identity": sorted(US.card_identity(STORED_CARD)),
           "reason": "task 820 test", "evidence_sha256": WAIVER_SHA, "approved_by": "tim",
           "approved_at": "2026-10-09", "pr": WAIVER_PR}
    body = (json.dumps({"schema": CW.WAIVER_SCHEMA}) + "\n" + json.dumps(row) + "\n").encode()
    path = tmp_path / "waivers.jsonl"
    path.write_bytes(body + CW.seal_line(body, 1).encode())
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def stored(monkeypatch):
    holder = SimpleNamespace(by_url={}, calls=[])

    def fake(url, collection, ids, fields, batch=500, workers=1):
        holder.calls.append(url)
        rows = holder.by_url.get(url, {})
        return {i: dict(rows[i]) for i in ids if i in rows}

    monkeypatch.setattr(PSP, "retrieve_payloads", fake)
    return holder


class Fake:
    def __init__(self, points):
        self.points = points
        self.writes: list[str] = []

    def upsert(self, collection, points, wait=False):
        for p in points:
            self.writes.append(str(p.id))
            self.points[str(p.id)] = p.payload

    def count(self, *a, **k):
        return SimpleNamespace(count=len(self.points))

    def retrieve(self, collection, ids, with_payload=True, with_vectors=True):
        return [SimpleNamespace(id=i, payload=self.points[i], vector={"dense": [0.1]})
                for i in ids if i in self.points]

    def get_collection(self, name):
        from qdrant_client import models
        return SimpleNamespace(status=models.CollectionStatus.GREEN)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="T")])


def _argv_waiver(tmp_path, use_waiver):
    if not use_waiver:
        return []
    path, sha = _waiver_file(tmp_path)
    return ["--card-waivers", path, "--card-waivers-sha256", sha]


def run_pack_delta(stored, tmp_path, monkeypatch, *, stored_row, new, use_waiver=False, dry=False):
    stored.by_url["http://t"] = {PID: stored_row}          # the card exists ONLY on the target
    stored.by_url["http://p"] = {}
    tgt = Fake({PID: stored_row})
    pack = SimpleNamespace(
        scroll=lambda name, limit, offset=None, with_payload=True, with_vectors=True, **k:
        ([SimpleNamespace(id=PID, payload=new, vector={"dense": [0.1]} if with_vectors else None)], None)
        if offset is None else ([], None),
        count=lambda name, exact=True: SimpleNamespace(count=1),
        get_collection=lambda name: None)
    made = {"pack": pack}
    monkeypatch.setattr(APD, "qdrant_connect",
                        lambda url=None, timeout=None: made.pop("pack") if "pack" in made else tgt)
    monkeypatch.setattr(APD.SC, "shape_from_info", lambda info: {"dense_size": 1})
    monkeypatch.setattr(APD.SC, "assert_dim_matches", lambda *a, **k: None)
    monkeypatch.setattr(US, "_gold_audit", lambda urls, log: 0)
    # the fake target answers to its url; the pack fake has none (a guard reading it fails)
    monkeypatch.setattr(APD, "client_url", lambda client: "http://t" if client is tgt else "http://p")
    T546.install_persisted_index(monkeypatch, tmp_path, [{"url": new["url"], "point_id": PID}])
    argv = ["apply_pack_delta.py", "--pack", "P", "--pack-url", "http://p", "--target", "T",
            "--target-url", "http://t", "--no-supersede"] + _argv_waiver(tmp_path, use_waiver)
    if dry:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    try:
        APD.main()
    except SystemExit as exc:
        if str(exc) not in ("0", "None"):
            return exc, tgt.writes
    return None, tgt.writes


def _apply_delta_module():
    return importlib.import_module("cytir.index.apply_delta")


def run_apply_delta(stored, tmp_path, monkeypatch, *, stored_row, new, use_waiver=False):
    stored.by_url["http://t"] = {PID: stored_row}
    AD = _apply_delta_module()
    tgt = Fake({PID: stored_row})
    inp = tmp_path / "delta.jsonl.gz"
    with gzip.open(inp, "wt") as fh:
        fh.write(json.dumps({"id": PID, "vector": {"dense": [0.1]}, "payload": new, "_delta": "modify"}) + "\n")
    monkeypatch.setattr(AD, "qdrant_connect", lambda *a, **k: tgt)
    monkeypatch.setattr(AD.LT, "refuse_live_target", lambda *a, **k: None)
    monkeypatch.setattr(AD, "client_url", lambda client: "http://t")
    T546.install_persisted_index(monkeypatch, tmp_path, [{"url": new["url"], "point_id": PID}])
    monkeypatch.setattr(sys, "argv", ["apply_delta.py", "--collection", "T", "--in", str(inp),
                                      "--no-supersede"] + _argv_waiver(tmp_path, use_waiver))
    try:
        AD.main()
    except SystemExit as exc:
        if str(exc) not in ("0", "None"):
            return exc, tgt.writes
    return None, tgt.writes


WRITERS = {"apply_pack_delta": run_pack_delta, "apply_delta": run_apply_delta}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_card_overwritten_by_a_non_card_is_refused_and_nothing_is_written(
        writer, stored, tmp_path, monkeypatch):
    exc, writes = WRITERS[writer](stored, tmp_path, monkeypatch, stored_row=STORED_CARD, new=SHOP)
    assert exc is not None, f"{writer} wrote over a stored card"
    assert writes == []
    text = str(exc)
    assert writer in text and PID in text and "https://a.ch/" in text and "would be overwritten" in text
    assert set(stored.calls) == {"http://t"}          # it read the TARGET, not the pack


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_card_replacing_a_card_is_written(writer, stored, tmp_path, monkeypatch):
    exc, writes = WRITERS[writer](stored, tmp_path, monkeypatch, stored_row=STORED_CARD,
                                  new=REFRESHED_CARD)
    assert exc is None and writes == [PID]


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_stored_non_card_is_not_the_guards_business(writer, stored, tmp_path, monkeypatch):
    exc, writes = WRITERS[writer](stored, tmp_path, monkeypatch, stored_row=STORED_PAGE, new=SHOP)
    assert exc is None and writes == [PID]


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_sealed_waiver_admits_the_overwrite_and_is_recorded(
        writer, stored, tmp_path, monkeypatch, capsys):
    exc, writes = WRITERS[writer](stored, tmp_path, monkeypatch, stored_row=STORED_CARD, new=SHOP,
                                  use_waiver=True)
    assert exc is None and writes == [PID]
    assert f"{writer}: waiver admitted id {PID} pr={WAIVER_PR} evidence_sha256={WAIVER_SHA}" \
        in capsys.readouterr().out


def test_apply_pack_delta_dry_run_refuses_too(stored, tmp_path, monkeypatch):
    exc, writes = run_pack_delta(stored, tmp_path, monkeypatch, stored_row=STORED_CARD, new=SHOP, dry=True)
    assert exc is not None and "would be overwritten" in str(exc) and writes == []
