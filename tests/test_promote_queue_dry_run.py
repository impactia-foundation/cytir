#!/usr/bin/env python3
"""A dry drain never writes (2026-10-03).

THE DEFECT. `drain(apply=False)` -- the mode agents and operators use to prove a pack before
enqueueing it -- resumed a sealed pack's prod leg whenever the pack sat in the "twin written, prod
owed" state: `dry_plan()` raised `AlreadyLandedError`, prod lacked the ids, and the branch called
`resume_prod_only()` without looking at `apply` or the gate profile. Reproduced on a throwaway
Qdrant 1.18.3: the dry run upserted the pack into the prod collection, took both writer locks and
the prod writer lock, appended the resume ledger, wrote DONE and a LANDED receipt. Both shapes: the
pack's own `twin-written-ids` artifact, and a scratch queue with no artifact (live twin diff).

What these tests hold, against an in-memory Qdrant behind the real `urllib` calls and a recorder
behind `subprocess.run` (the ssh of every lock) -- nothing below the drain is stubbed on the write
path:

1. A dry drain over a resume-state pack writes nothing to either collection, runs no ssh, writes
   nothing into the queue tree (hashed before and after), and prints what an apply run WOULD do.
   An `--apply` run over the same state still resumes (the fake is not vacuous).
2. Every write primitive the drain can reach refuses under `dry_run.guard` before any I/O.
3. The roster of raw network call sites in the drain's import closure is pinned: a new one fails
   here until it is guarded or shown to be a read.
4. `--full-persisted-index` without `--apply` is refused before anything is deleted; the incremental
   index top-up stays (the one documented write of a dry run, see cytir/index/dry_run.py).
5. A structural scan of the drain's static import closure (function-level imports and aliases
   included) for file writes, renames, deletes, `shutil` and `subprocess` call sites that are
   neither under a dry-run guard nor declared in `FILE_WRITE_SITES` -- a declared baseline that
   can only shrink.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    """Operator identity for each test, restored after it. A module-level setdefault leaked it into every later test."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402

from cytir.gates import rollback_restore_null_arm as RB  # noqa: E402
from cytir.index import dry_run as DRY  # noqa: E402
from cytir.index import prod_guard as PG  # noqa: E402
from cytir.index import point_delete as PD  # noqa: E402
from cytir.index import twin_first as TFA  # noqa: E402
from cytir.core import paths as PATHS  # noqa: E402
from cytir.ops import write_lock as WL  # noqa: E402

FIXTURE = HERE / "fixtures" / "promote" / "task629_sealed_pack"

@pytest.fixture(autouse=True)
def _operator_values(monkeypatch, tmp_path_factory):
    """The values the operator overlay supplies on a configured box; a stranger's suite runs with
    none, so this module pins neutral stand-ins (it asserts behaviour, not the box's real values)."""
    monkeypatch.setenv("AW_PROD_LOCK_HOST", "prod-lock-host")
    monkeypatch.setenv("AW_TWIN_SSH_HOST", "twin-ssh-host")
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path_factory.getbasetemp() / "work-root"))

TWIN_URL = "http://twin.invalid:1"
PROD_URL = "http://prod.invalid:1"
TWIN, PROD = "aw_e8", "aw_e9"
MANIFEST = {
    "schema": Q.MANIFEST_SCHEMA,
    "pack_id": "resume-state-pack",
    "family": "technical_manufacturer_catalog",
    "pack_chunks": str(FIXTURE / "chunks"),
    "vectors": str(FIXTURE / "vectors"),
    "mrl": 256,
    "collection": TWIN,
    "identity_check": "pack-declared",
    "identity_reason": "test: Pool A ids are not reproducible; module docstring",
    "family_payload": {"source_type": "docs", "trust_tier": 0.6},
    "title_from": ["source_metadata.title"],
}
PACK_IDS = sorted(str(r["point_id"]) for r in PSP.read_records(MANIFEST["pack_chunks"]))


# ------------------------------------------------------------------ the fakes
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeQdrant:
    """Two collections on two hosts, `aw_prod` -> aw_e9 on the prod host. Answers the calls the
    drain makes; records every call `dry_run.is_mutating_http` calls a write, and refuses any
    endpoint it does not know (so a new call shows up here, not as a silent no-op)."""

    def __init__(self):
        self.points = {(TWIN_URL, TWIN): {}, (PROD_URL, PROD): {}}
        self.writes: list[tuple[str, str]] = []

    def count(self, url, coll) -> int:
        return len(self.points[(url, coll)])

    def urlopen(self, req, timeout=None):
        if isinstance(req, str):
            full, method, body = req, "GET", None
        else:
            full, method = req.full_url, req.get_method()
            body = json.loads(req.data) if req.data else None
        u = urllib.parse.urlsplit(full)
        base, path = f"{u.scheme}://{u.netloc}", u.path
        if DRY.is_mutating_http(method, path):
            self.writes.append((method, f"{base}{path}"))
        parts = path.strip("/").split("/")
        if method == "GET" and path == "/aliases":
            aliases = [{"alias_name": "aw_prod", "collection_name": PROD}] if base == PROD_URL else []
            result = {"aliases": aliases}
        elif parts[:1] == ["collections"] and len(parts) >= 3 and parts[2] == "points":
            store = self.points[(base, parts[1])]
            tail = parts[3:]
            if method == "POST" and not tail:                      # retrieve by id
                result = []
                for i in body["ids"]:
                    if str(i) not in store:
                        continue
                    row, st = {"id": i}, store[str(i)]
                    if isinstance(body.get("with_payload"), list):      # task 717: the readback
                        row["payload"] = {k: v for k, v in (st.get("payload") or {}).items()
                                          if k in body["with_payload"]}
                    if isinstance(body.get("with_vector"), list):
                        row["vector"] = {k: v for k, v in (st.get("vector") or {}).items()
                                         if k in body["with_vector"]}
                    result.append(row)
            elif method == "PUT" and not tail:                     # upsert
                for p in body["points"]:
                    store[str(p["id"])] = p
                result = {"status": "completed"}
            elif method == "POST" and tail == ["delete"]:
                for i in body.get("points", []):
                    store.pop(str(i), None)
                result = {"status": "completed"}
            else:
                raise AssertionError(f"FakeQdrant: unhandled {method} {path}")
        else:
            raise AssertionError(f"FakeQdrant: unhandled {method} {path}")
        return _Resp(json.dumps({"status": "ok", "result": result}).encode())


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The fake Qdrant behind the real `urllib`, a recorder behind `subprocess.run`, and the
    three plan inputs a unit test has no host for (persisted inputs, scoreable check, layout)."""
    fake = FakeQdrant()
    ssh: list = []

    def fake_run(args, *a, **k):
        ssh.append(args if isinstance(args, str) else " ".join(map(str, args)))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    # The fake store's two ".invalid" endpoints stand in for twin AND prod. They are not the serving
    # box. Since #660 a dry drain is a replay, and a replay never reads an endpoint the prod gate
    # calls prod (`aw_e9` is `serving: true`): so the stand-ins are declared throwaways, as
    # `tests/throwaway_qdrant` does for a real throwaway Qdrant. The REAL behaviour (a dry drain
    # refuses the prod membership read) is pinned in test_promote_queue.py.
    monkeypatch.setattr(PG, "_THROWAWAY", set(PG._THROWAWAY))
    PG.mark_throwaway(TWIN_URL)
    PG.mark_throwaway(PROD_URL)
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda url, coll, **_k: {"points_count": fake.count(url, coll)})
    # Task 717: aw_e9 declares the BM25 prune rule, so a prod write needs its term set. A tiny real
    # one: the fixture's own rows decide nothing about the rule (tests/test_aw_sparse_rule.py does).
    from cytir.core import paths as _paths
    from cytir.index import payload_profile as _PP
    from cytir.index import sparse_rule as _SR
    drop_file = tmp_path / "drop-terms.u32"
    written = _SR.write_drop_terms(drop_file, [1, 2, 3])
    profiles = json.loads(_PP.PROFILES_PATH.read_text())
    rule = profiles["collections"]["aw_e9"]["sparse_rule"]
    rule.update(terms=written["terms"], drop_terms_sha256=written["sha256"])
    (tmp_path / "profiles.json").write_text(json.dumps(profiles))
    monkeypatch.setattr(_PP, "PROFILES_PATH", tmp_path / "profiles.json")
    monkeypatch.setenv("AW_SPARSE_RULE_DROP_TERMS", str(drop_file))
    _paths.reset_settings_cache()
    queue = tmp_path / "queue"
    pack = queue / MANIFEST["pack_id"]
    pack.mkdir(parents=True)
    (pack / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    (pack / "READY").write_text(
        hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest() + "\n")
    contract = tmp_path / "gate-contract.json"

    def set_gate(profile: str):
        contract.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110",
                                        "pr": 302, "profile": profile}))
    set_gate("mechanical")
    return {"fake": fake, "ssh": ssh, "queue": queue, "artifacts": queue / "_artifacts",
            "contract": contract, "set_gate": set_gate}


def twin_written_prod_owed(w, *, own_artifact: bool, prod_has_them: bool = False):
    """The state a kill between the legs leaves: twin holds every id of the pack, prod none."""
    for pid in PACK_IDS:
        w["fake"].points[(TWIN_URL, TWIN)][pid] = {"id": pid}
        if prod_has_them:
            w["fake"].points[(PROD_URL, PROD)][pid] = {"id": pid}
    if own_artifact:
        w["artifacts"].mkdir(parents=True, exist_ok=True)
        (w["artifacts"] / f"{MANIFEST['pack_id']}-twin-written-ids.json").write_text(
            json.dumps(PACK_IDS))


def tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def run_drain(w, *, apply: bool, **extra):
    return Q.drain(w["queue"], twin_url=TWIN_URL, prod_url=PROD_URL, task="dry-run-test",
                   apply=apply, batch=10, pause_s=0, artifact_root=w["artifacts"],
                   contract_path=w["contract"], refresh_index=False, **extra)


# ------------------------------------------- 1. the dry drain over a resume state
@pytest.mark.parametrize("own_artifact", [True, False], ids=["own-artifact", "live-twin-diff"])
def test_a_dry_drain_over_a_resume_state_pack_writes_nothing(world, capsys, own_artifact):
    twin_written_prod_owed(world, own_artifact=own_artifact)
    before = tree(world["queue"])
    assert world["fake"].count(PROD_URL, PROD) == 0

    results = run_drain(world, apply=False)

    assert world["fake"].writes == []
    assert world["fake"].count(PROD_URL, PROD) == 0
    assert world["fake"].count(TWIN_URL, TWIN) == len(PACK_IDS)
    assert world["ssh"] == [], "a dry run ran ssh (a writer lock)"
    assert tree(world["queue"]) == before, "a dry run wrote into the queue"
    (rec,) = results
    assert rec["status"] == "PROVED-DRY-RUN"
    assert rec["would"].startswith(f"resume prod leg: {len(PACK_IDS)} ids")
    out = capsys.readouterr().out
    assert f"would resume prod leg: {len(PACK_IDS)} ids" in out
    assert "RESUMING" not in out


def test_the_same_state_with_apply_does_resume(world):
    """Control: the fake is a real path. `--apply` resumes the prod leg and lands the pack."""
    twin_written_prod_owed(world, own_artifact=True)
    (rec,) = run_drain(world, apply=True)
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    assert world["fake"].count(PROD_URL, PROD) == len(PACK_IDS)
    assert (world["queue"] / MANIFEST["pack_id"] / "DONE").exists()
    assert any("set -C" in c for c in world["ssh"]), "the writer locks were taken"


def test_a_resume_obeys_the_gate_profile_with_apply(world):
    """The same branch ignored the gate profile too: a battery-required gate blocks it."""
    twin_written_prod_owed(world, own_artifact=True)
    world["set_gate"]("battery-required")
    (rec,) = run_drain(world, apply=True)
    assert rec["status"] == "PROVED-BLOCKED"
    assert world["fake"].writes == [] and world["ssh"] == []
    assert not (world["queue"] / MANIFEST["pack_id"] / "DONE").exists()


def test_a_dry_drain_never_marks_done_when_both_targets_hold_every_id(world):
    twin_written_prod_owed(world, own_artifact=True, prod_has_them=True)
    before = tree(world["queue"])
    (rec,) = run_drain(world, apply=False)
    assert rec["status"] == "PROVED-DRY-RUN" and rec["would"].startswith("mark DONE")
    assert tree(world["queue"]) == before


def test_dry_receipts_go_to_their_own_dir_never_the_queue(world, tmp_path):
    twin_written_prod_owed(world, own_artifact=True)
    before = tree(world["queue"])
    run_drain(world, apply=False, dry_receipt_dir=tmp_path / "dry")
    assert tree(world["queue"]) == before
    got = json.loads((tmp_path / "dry" / f"{MANIFEST['pack_id']}.receipt.json").read_text())
    assert got["status"] == "PROVED-DRY-RUN"
    with pytest.raises(Q.QueueError, match="inside the queue"):
        run_drain(world, apply=False, dry_receipt_dir=world["queue"] / "x")


# -------------------------------- 2. every write primitive refuses under the guard
WRITE_CALLS = {
    "PSP.upsert (PUT points)":
        lambda t: PSP.upsert(TWIN_URL, TWIN, [{"id": "a", "vector": {}, "payload": {}}], 10),
    "PSP._api POST points/delete":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/delete?wait=true", {"points": ["a"]}),
    "PSP._api POST points/payload (set)":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/payload", {"payload": {}}),
    "PSP._api PUT points/payload (overwrite)":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/payload", {"payload": {}}, method="PUT"),
    "PSP._api POST points/payload/delete":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/payload/delete", {"keys": ["k"]}),
    "PSP._api PUT points/vectors":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/vectors", {"points": []}, method="PUT"),
    "PSP._api POST points/batch":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/points/batch", {"operations": []}),
    "PSP._api PUT collection (create)":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}", {"vectors": {}}, method="PUT"),
    "PSP._api PATCH collection":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}", {"optimizers_config": {}}, method="PATCH"),
    "PSP._api DELETE collection":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}", method="DELETE"),
    "PSP._api POST collections/aliases":
        lambda t: PSP._api(PROD_URL, "/collections/aliases", {"actions": []}),
    "PSP._api PUT payload index":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/index", {"field_name": "x"}, method="PUT"),
    "PSP._api POST snapshot":
        lambda t: PSP._api(TWIN_URL, f"/collections/{TWIN}/snapshots", {}),
    "PD.delete_points_verified":
        lambda t: PD.delete_points_verified(TWIN_URL, TWIN, ["a"]),
    "PD._api POST points/payload/clear":
        lambda t: PD._api(TWIN_URL, f"/collections/{TWIN}/points/payload/clear", {"points": ["a"]}),
    "RB.delete_ids (rollback)":
        lambda t: RB.delete_ids(TWIN_URL, TWIN, ["a"]),
    "TFA.ProductionWriterLock.acquire (ssh)":
        lambda t: TFA.ProductionWriterLock("host", "/tmp/x.lock", "p").acquire(),
    "TFA.ProductionWriterLock.release (ssh)":
        lambda t: TFA.ProductionWriterLock("host", "/tmp/x.lock", "p").release(),
    "TFA.append_ledger":
        lambda t: TFA.append_ledger(t / "ledger.jsonl", {"status": "X"}),
    "WL.acquire (ssh)":
        lambda t: WL.acquire("TWIN-WRITE-LOCK", "t", "p"),
    "WL.release (ssh)":
        lambda t: WL.release("TWIN-WRITE-LOCK", "t"),
    "Q.atomic_write_text":
        lambda t: Q.atomic_write_text(t / "DONE", "x\n"),
    "Q.atomic_write_bytes":
        lambda t: Q.atomic_write_bytes(t / "receipt.json", b"{}"),
}


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_every_write_primitive_refuses_under_a_dry_guard_before_any_io(name, tmp_path, monkeypatch):
    def no_io(*_a, **_k):
        raise AssertionError(f"{name}: reached I/O under a dry guard")
    monkeypatch.setattr(urllib.request, "urlopen", no_io)
    monkeypatch.setattr(subprocess, "run", no_io)
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            WRITE_CALLS[name](tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reads_pass_under_a_dry_guard(monkeypatch):
    fake = FakeQdrant()
    fake.points[(TWIN_URL, TWIN)]["a"] = {"id": "a"}
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    with DRY.guard(True):
        assert PSP.already_present(TWIN_URL, TWIN, ["a", "b"]) == ["a"]
        assert Q.resolve_prod_alias(PROD_URL) == PROD
    assert fake.writes == []


@pytest.mark.parametrize("method,path,mutating", [
    ("GET", "/collections/aw_e8", False),
    ("GET", "/aliases", False),
    ("POST", "/collections/aw_e8/points", False),                 # retrieve by id
    ("POST", "/collections/aw_e8/points/scroll", False),
    ("POST", "/collections/aw_e8/points/count", False),
    ("POST", "/collections/aw_e8/points/query", False),
    ("POST", "/collections/aw_e8/points/search/batch", False),
    ("POST", "/collections/aw_e8/facet", False),                   # the real 1.18.3 route
    ("POST", "/collections/aw_e8/points/facet", True),             # does not exist there (404)
    ("PUT", "/collections/aw_e8/points?wait=true", True),          # upsert
    ("POST", "/collections/aw_e8/points/delete?wait=true", True),
    ("POST", "/collections/aw_e8/points/payload", True),
    ("PUT", "/collections/aw_e8/points/payload", True),
    ("POST", "/collections/aw_e8/points/payload/clear", True),
    ("PUT", "/collections/aw_e8/points/vectors", True),
    ("POST", "/collections/aw_e8/points/vectors/delete", True),
    ("POST", "/collections/aw_e8/points/batch", True),
    ("PUT", "/collections/aw_e8", True),
    ("PATCH", "/collections/aw_e8", True),
    ("DELETE", "/collections/aw_e8", True),
    ("POST", "/collections/aliases", True),
    ("PUT", "/collections/aw_e8/index", True),
    ("POST", "/collections/aw_e8/snapshots", True),
    ("POST", "/collections/aw_e8/points/somethingnew", True),     # unknown: refused
])
def test_the_qdrant_read_write_classifier(method, path, mutating):
    assert DRY.is_mutating_http(method, path) is mutating


def test_a_swallowed_refusal_still_fails_the_dry_run(tmp_path):
    """A caller may catch the refusal (the drain turns most per-pack errors into a REFUSED
    receipt); the guard raises again on exit, so the bug cannot pass as "nothing to do"."""
    with pytest.raises(DRY.DryRunWriteRefused, match="attempted 1 write"):
        with DRY.guard(True):
            try:
                Q.atomic_write_text(tmp_path / "x", "y")
            except DRY.DryRunWriteRefused:
                pass
    assert not DRY.active()


def test_an_apply_drain_is_not_guarded(tmp_path):
    with DRY.guard(False):
        Q.atomic_write_text(tmp_path / "x", "y")
    assert (tmp_path / "x").read_text() == "y"


# ------------------------------- 3. the raw network call sites of the drain, pinned
#: Every call site in the drain's import closure that can reach another host -- `urlopen`,
#: `subprocess.*`, `QdrantClient(` and the factory calls `qdrant_connect(` / `qdrant_client(` -- and why a dry drain cannot write through it. A new site
#: fails `test_the_raw_network_call_sites_of_the_drain_are_all_accounted_for` until it is added
#: here with its reason, and a "guarded" one must really carry the guard.
NETWORK_SITES = {
    # consent step 5 (task 1648): the index tools reach Qdrant through ONE door; its callers carry
    # the guard (`test_the_guarded_primitives_really_carry_the_guard` pins DRY.refuse_http in the
    # write callers promote_sealed_pack._api and point_delete._api).
    "cytir.core.internal_http:make_internal_client": "the door's lazy wrapper over cytir.sources.http.make_internal_client; it builds the client, `send` below is the only caller",
    "cytir.core.internal_http:send": "the one door to Qdrant; write callers guard with DRY.refuse_http, reads (layout._get, prod_guard._fetch_aliases) are GETs",
    "cytir.index.twin_first:ProductionWriterLock._run":
        "guarded: only acquire/release call it, both @refuses_when_dry",
    "cytir.ops.write_lock:ssh":
        "guarded: acquire/release are @refuses_when_dry; read_lock is a `cat`",
    "cytir.index.drain.plan:persisted_index_is_stale": "read: `find -newer` on the local corpus",
    "cytir.index.supersede:SSHTunnel.__enter__": "not reached: the drain uses only supersede.is_card",
    "cytir.index.supersede:connect_twin": "not reached: the drain uses only supersede.is_card",
    "cytir.index.supersede:main": "not reached: supersede's own CLI",
    "cytir.core.bm25_saturate:client": "not reached: bm25_saturate's own runtime commands",
    "cytir.core.qdrant:qdrant_connect": "constructor only: builds a client object, sends no request; the writes are the callers' own CLIs (apply/upsert/supersede), not the drain",
    "cytir.core.qdrant:qdrant_client": "constructor only: builds a client object, sends no request; used by the dump/apply tools, not the drain",
}

_NET_CALL_NAMES = {"urlopen", "make_internal_client", "QdrantClient", "qdrant_connect", "qdrant_client"}
_SUBPROCESS_FNS = {"run", "Popen", "call", "check_call", "check_output"}


def _drain_closure() -> list[str]:
    """The `aw.*` modules a fresh interpreter loads for the drain (and its lazily loaded lock
    module) -- derived, not hand-listed."""
    code = ("import sys, cytir.index.promote_queue, cytir.ops.write_lock; "
            "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('cytir.'))))")
    env = {**os.environ, "AW_OPERATOR_URL": "https://websear.ch",
           "AW_OPERATOR_CONTACT": "bender@impactia.org",
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=str(ROOT), check=True).stdout
    return out.split()


def _network_sites(module: str) -> set[str]:
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()

    def visit(node, scope: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + [child.name])
                continue
            if isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                base = f.value.id if (isinstance(f, ast.Attribute)
                                      and isinstance(f.value, ast.Name)) else None
                if name in _NET_CALL_NAMES or (base == "subprocess" and name in _SUBPROCESS_FNS):
                    found.add(f"{module}:{'.'.join(scope) or '<module>'}")
            visit(child, scope)
    visit(tree, [])
    return found


def test_the_raw_network_call_sites_of_the_drain_are_all_accounted_for():
    modules = _drain_closure()
    assert "cytir.index.promote_queue" in modules and "cytir.ops.write_lock" in modules, modules
    sites = set().union(*(_network_sites(m) for m in modules))
    unaccounted = sorted(sites - set(NETWORK_SITES))
    assert not unaccounted, (
        f"new raw network call site(s) in the drain's import closure: {unaccounted}. Guard the "
        "write (cytir/index/dry_run.py) or show it is a read, then add it to NETWORK_SITES.")
    stale = sorted(set(NETWORK_SITES) - sites)
    assert not stale, f"NETWORK_SITES names call sites that no longer exist: {stale}"


def test_the_guarded_primitives_really_carry_the_guard():
    import inspect
    for fn in (PSP._api, PD._api):
        assert "DRY.refuse_http(" in inspect.getsource(fn), fn
    for fn in (TFA.ProductionWriterLock.acquire, TFA.ProductionWriterLock.release,
               TFA.append_ledger, WL.acquire, WL.release, Q.atomic_write_bytes):
        assert getattr(fn, "__dry_run_guarded__", None), fn


# ----------------------- 4. --full-persisted-index and the index top-up in a dry run
def _fake_index(tmp_path, monkeypatch):
    """A persisted-index directory with the sqlite file and its wal/shm, as the hourly unit has
    it (`AW_PERSISTED_INDEX_SQLITE_DIR`), and a recorder behind the incremental update."""
    idx = tmp_path / "persisted-index"
    idx.mkdir()
    for name, body in (("persisted-index.sqlite", b"db"), ("persisted-index.sqlite-wal", b"wal"),
                       ("persisted-index.sqlite-shm", b"shm"), ("INDEX.json", b"{}")):
        (idx / name).write_bytes(body)
    monkeypatch.setenv("AW_PERSISTED_INDEX_SQLITE_DIR", str(idx))
    monkeypatch.setattr(Q.PI, "default_index_dir", lambda: str(idx))
    monkeypatch.setattr(Q.PI, "sqlite_index_path", lambda d: str(idx / "persisted-index.sqlite"))
    calls: list = []

    def update(index_dir, *a, **k):
        calls.append(sorted(p.name for p in idx.iterdir()))
        return {"files_changed": 0, "rows_indexed": 0, "seconds": 0}
    monkeypatch.setattr(Q.PI, "update_index_sqlite", update)
    return idx, calls


def _drain_with_index(w, **extra):
    return Q.drain(w["queue"], twin_url=TWIN_URL, prod_url=PROD_URL, task="t", batch=10,
                   pause_s=0, artifact_root=w["artifacts"], contract_path=w["contract"],
                   refresh_index=True, **extra)


def test_a_dry_drain_refuses_full_persisted_index_and_deletes_nothing(world, tmp_path, monkeypatch):
    idx, calls = _fake_index(tmp_path, monkeypatch)
    before, queue_before = tree(idx), tree(world["queue"])
    with pytest.raises(Q.QueueError, match="needs --apply"):
        _drain_with_index(world, apply=False, full_persisted_index=True)
    assert tree(idx) == before and calls == []
    assert tree(world["queue"]) == queue_before and not DRY.active()


def test_the_cli_refuses_full_persisted_index_without_apply_and_deletes_nothing(
        world, tmp_path, monkeypatch):
    idx, calls = _fake_index(tmp_path, monkeypatch)
    before = tree(idx)
    with pytest.raises(SystemExit) as exc:
        Q.main(["--queue-dir", str(world["queue"]), "--full-persisted-index",
                "--gate-contract", str(world["contract"])])
    assert exc.value.code not in (0, None) and "needs --apply" in str(exc.value.code)
    assert tree(idx) == before and calls == []


def test_a_dry_drain_still_tops_up_the_index_incrementally_and_never_wipes_it(
        world, tmp_path, monkeypatch):
    """The one documented write (cytir/index/dry_run.py): incremental, idempotent, never a wipe. A
    stale index would false-refuse real packs, so the dry plan needs it."""
    idx, calls = _fake_index(tmp_path, monkeypatch)
    before = tree(idx)
    _drain_with_index(world, apply=False)
    assert len(calls) == 1 and "persisted-index.sqlite" in calls[0]
    assert tree(idx) == before


def test_an_apply_drain_with_full_persisted_index_still_wipes_then_rebuilds(
        world, tmp_path, monkeypatch):
    """The --apply path is unchanged: `full` still deletes the sqlite files before the update."""
    idx, calls = _fake_index(tmp_path, monkeypatch)
    _drain_with_index(world, apply=True, full_persisted_index=True)
    assert calls == [["INDEX.json"]]


# ------------------------- 5. file writes in the drain's closure: a baseline that only shrinks
_FILE_OS_FNS = {"replace", "rename", "remove", "unlink", "renames", "system"}
_FILE_PATH_ATTRS = {"write_text", "write_bytes", "unlink"}


def _module_file(module: str) -> pathlib.Path | None:
    base = ROOT / module.replace(".", "/")
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _static_closure(roots: list[str]) -> list[str]:
    """Every `aw.*` module reachable by an import statement ANYWHERE in the source (function-level
    imports included -- the runtime `sys.modules` closure above misses those)."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        m = todo.pop()
        path = _module_file(m)
        if m in seen or path is None:
            continue
        seen.add(m)
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names if a.name.split(".")[0] == "cytir"]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").startswith("cytir"):
                todo.append(n.module)
                todo += [f"{n.module}.{a.name}" for a in n.names]
    return sorted(seen)


def _file_write_sites(source: str, module: str) -> set[str]:
    """`module:Scope [kind]` for every file write / rename / delete / shutil / subprocess call
    that is not inside a function guarded by the dry-run switch. Aliased forms
    (`import os as o`, `from subprocess import run`) and in-function imports are followed."""
    tree = ast.parse(source)
    alias: dict[str, tuple[str, str | None]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in ("os", "shutil", "subprocess"):
                    alias[a.asname or a.name] = (a.name, None)
        elif isinstance(n, ast.ImportFrom) and n.module in ("os", "shutil", "subprocess"):
            for a in n.names:
                alias[a.asname or a.name] = (n.module, a.name)

    def open_kind(call: ast.Call) -> str | None:
        mode = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg == "mode"), None)
        if mode is None:
            return None
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            return "open(write)" if set(mode.value) & set("wax+") else None
        return "open(dynamic mode)"

    def kind(call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Name):
            if f.id == "open":
                return open_kind(call)
            mod, attr = alias.get(f.id, (None, None))
            if attr and (mod in ("shutil", "subprocess") or (mod == "os" and attr in _FILE_OS_FNS)):
                return f"{mod}.{attr}"
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name):
                mod, attr = alias.get(f.value.id, (None, "-"))
                if attr is None and (mod in ("shutil", "subprocess")
                                     or (mod == "os" and f.attr in _FILE_OS_FNS)):
                    return f"{mod}.{f.attr}"
            if f.attr == "open":
                return open_kind(call)
            if f.attr in _FILE_PATH_ATTRS:
                return f".{f.attr}"
        return None

    def guarded(fn) -> bool:
        return (any("refuses_when_dry" in ast.unparse(d) for d in fn.decorator_list)
                or any(isinstance(n, ast.Call) and ast.unparse(n.func).startswith("DRY.refuse")
                       for n in ast.walk(fn)))

    found: set[str] = set()

    def visit(node, scope: list[str], is_guarded: bool):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(c, scope + [c.name], is_guarded or guarded(c))
                continue
            if isinstance(c, ast.ClassDef):
                visit(c, scope + [c.name], is_guarded)
                continue
            if isinstance(c, ast.Call) and not is_guarded and (k := kind(c)):
                found.add(f"{module}:{'.'.join(scope) or '<module>'} [{k}]")
            visit(c, scope, is_guarded)
    visit(tree, [], False)
    return found


_OWN_CLI = "an own CLI/verb; not judged individually, reachability from the drain not proven"
_BASELINE = "BASELINE: not judged individually; may only be removed, never added"
_TOPUP = "the index top-up's own cache (corpus root, not the queue): the documented dry-run write"
#: Declared findings of the scan, each with its reason. A NEW write in the drain's closure fails
#: `test_no_new_unguarded_file_write_in_the_drains_closure` until it is guarded by the dry-run
#: switch or declared here with a reason; an entry that no longer matches must be deleted
#: (the list can only shrink).
FILE_WRITE_SITES = {
    "cytir.index.change_log:_seq_lock [open(dynamic mode)]":
        "the lock file is opened only when `DRY.active()` is false: a dry drain returns before the open "
        "(`_seq_lock` yields first) and the segment write behind it refuses; "
        "tests/test_change_log_direct.py pins that a dry run creates no `.seq.lock`",
    "cytir.index.drain.receipts:_dry_run_receipt [.write_text]":
        "by design: only under DRY.receipt_dir(), a directory outside the queue",
    "cytir.index.drain.receipts:_dry_run_receipt [os.replace]":
        "by design: only under DRY.receipt_dir(), a directory outside the queue",
    "cytir.index.drain.plan:refresh_persisted_index [.unlink]":
        "`full=True` only; refused up front in a dry run (FULL_INDEX_NEEDS_APPLY)",
    "cytir.index.drain.plan:persisted_index_is_stale [subprocess.run]": "read: `find -newer`",
    "cytir.index.drain.queue_scope:claim_state_entry [.unlink]":
        "a Single's own state entry only (never the queue): removes a dead claimer's leftover tmp",
    "cytir.index.drain.locks:hold_unit_lock [open(write)]":
        "the `cyt index promote` verb's lock file (a flock target, no content), not reached by drain()",
    "cytir.index.drain.markers:mark_carried_by_flip [.unlink]":
        "separate verb (--mark-carried-by-flip), never reached by drain()",
    "cytir.index.drain.markers:finalize_carried_by_flip [.unlink]":
        "separate verb (--finalize-carried-by-flip), never reached by drain()",
    "cytir.index.twin_first:ProductionWriterLock._run [subprocess.run]":
        "guarded: only acquire/release call it, both @refuses_when_dry",
    "cytir.ops.write_lock:ssh [subprocess.run]":
        "guarded: acquire/release are @refuses_when_dry; read_lock is a `cat`",
    "cytir.index.supersede:SSHTunnel.__enter__ [subprocess.Popen]":
        "not reached: the drain uses only supersede.is_card",
    "cytir.corpus.persisted:_stamp_cache_registry [.write_text]": _TOPUP,
    "cytir.corpus.persisted:_stamp_cache_registry [os.replace]": _TOPUP,
    "cytir.corpus.persisted:_write_cache_entry [.write_text]": _TOPUP,
    "cytir.corpus.persisted:_write_cache_entry [open(write)]": _TOPUP,
    "cytir.corpus.persisted:_write_cache_entry [os.replace]": _TOPUP,
    "cytir.corpus.persisted:build_index [.write_bytes]": "the index builder; " + _BASELINE,
    "cytir.corpus.persisted:build_index [.write_text]": "the index builder; " + _BASELINE,
    "cytir.corpus.persisted:build_index [open(write)]": "the index builder; " + _BASELINE,
    "cytir.core.bm25_saturate:cmd_stats [.write_text]": _OWN_CLI,
    "cytir.gates.authority:build_cache [open(write)]": _BASELINE,
    "cytir.gates.authority:main [open(write)]": _OWN_CLI,
    "cytir.gates.authority:run [open(write)]": _BASELINE,
    "cytir.gates.convergence:main [.write_text]": _OWN_CLI,
    "cytir.gates.gold_guard:assert_no_golds [open(write)]": _BASELINE,
    "cytir.gates.rollback_restore_null_arm:main [.write_text]": _OWN_CLI,
    "cytir.index.news_window_filter:main [open(write)]": _OWN_CLI,
    "cytir.index.payload_carry:main [.write_text]": _OWN_CLI,
    "cytir.index.promote_sealed_pack:main [.write_text]": _OWN_CLI,
}


def test_the_file_write_scanner_sees_aliases_and_function_level_imports():
    src = (
        "import os as o\nimport subprocess as sp\nfrom shutil import copy as cp\n"
        "from pathlib import Path\n"
        "def a(p):\n    open(p, 'w')\n    open(p, mode='ab')\n    open(p)\n    open(p, 'r')\n"
        "def b(p):\n    o.replace(p, p)\n    o.rename(p, p)\n    sp.run(['x'])\n    cp(p, p)\n"
        "def c(p):\n    Path(p).write_text('x')\n    p.unlink()\n    open(p, m)\n"
        "def d(p):\n    from subprocess import run\n    run(['y'])\n"
        "def e(p):\n    '/x'.replace('a', 'b')\n"
        "@DRY.refuses_when_dry('w')\ndef f(p):\n    open(p, 'w')\n"
        "def g(p):\n    DRY.refuse('x')\n    open(p, 'w')\n")
    assert _file_write_sites(src, "m") == {
        "m:a [open(write)]", "m:b [os.replace]", "m:b [os.rename]", "m:b [subprocess.run]",
        "m:b [shutil.copy]", "m:c [.write_text]", "m:c [.unlink]", "m:c [open(dynamic mode)]",
        "m:d [subprocess.run]"}


def test_no_new_unguarded_file_write_in_the_drains_closure():
    modules = _static_closure(["cytir.index.promote_queue", "cytir.ops.write_lock"])
    assert "cytir.index.promote_queue" in modules and "cytir.corpus.persisted" in modules, modules
    sites: set[str] = set()
    for m in modules:
        sites |= _file_write_sites(_module_file(m).read_text(), m)
    new = sorted(sites - set(FILE_WRITE_SITES))
    assert not new, (
        f"new unguarded file write / rename / delete / shutil / subprocess call site(s) in the "
        f"drain's import closure: {new}. Guard the write with the dry-run switch "
        "(cytir/index/dry_run.py: `@refuses_when_dry` or `refuse()`), or declare it in "
        "FILE_WRITE_SITES with the reason a dry drain cannot reach it.")
    stale = sorted(set(FILE_WRITE_SITES) - sites)
    assert not stale, f"FILE_WRITE_SITES names sites that no longer exist, delete them: {stale}"


# ===================================================================================================
# 5. the `Single` transaction (slice 8 step 2a): a dry `Single` build writes nothing either
#
# `Single` adds branches the twin-first drain does not have. The same in-memory Qdrant behind the real
# `urllib` calls and the same ssh recorder, over a clean BUILD endpoint (no alias, no live collection):
#  * a dry `Single` over a FRESH pack, and over a RESUME-STATE pack (twin-first state in the queue: a
#    twin leg written, prod owed, its ledger artifact, a cached receipt), writes NOTHING anywhere: not
#    the queue, not the artifact root, not the transaction's state dir (it is never created: no shadow
#    copy of a pack dir), no ssh (no lock), no Qdrant write. The trees are byte-identical.
#  * "the target already holds every id" says "would mark DONE" in a dry run and writes DONE only with
#    `--apply` (control below).
#  * a dry run's only output on disk is `dry_receipt_dir`, outside the queue and the state dir.
#  * the apply-only primitive of the transaction refuses under the guard (the structural backstop).
# ===================================================================================================
SINGLE_URL = "http://build.invalid:1"
TARGET = "aw_build"


class FakeWithBuildEndpoint(FakeQdrant):
    """The in-memory Qdrant plus a clean BUILD endpoint (no alias, no live collection) and the
    `GET /collections` listing a live-target check reads."""

    def __init__(self):
        super().__init__()
        self.points[(SINGLE_URL, TARGET)] = {}

    def urlopen(self, req, timeout=None):
        full, method = (req, "GET") if isinstance(req, str) else (req.full_url, req.get_method())
        if method == "GET" and full.rstrip("/").endswith("/collections"):
            base = full.rsplit("/collections", 1)[0]
            names = sorted({c for (u, c) in self.points if u == base})
            return _Resp(json.dumps({"status": "ok", "result": {
                "collections": [{"name": n} for n in names]}}).encode())
        return super().urlopen(req, timeout)


@pytest.fixture
def single_world(world, monkeypatch, tmp_path):
    fake = FakeWithBuildEndpoint()
    world["fake"] = fake
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda url, coll, **_k: {"points_count": fake.count(url, coll)})
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    # the content check's language detector (a model file a runner may or may not have) is a per-pack
    # MANIFEST override in every drain-level test: the same override here
    real_check = Q.mechanical_content_status
    monkeypatch.setattr(Q, "mechanical_content_status",
                        lambda manifest, joined: real_check({**manifest, "detect_language": False},
                                                            joined))
    live = tmp_path / "the-live-queue"
    live.mkdir()
    monkeypatch.setenv("AW_QUEUE_DIR", str(live))
    PATHS.reset_settings_cache()
    world["state"] = tmp_path / "state"
    world["dry_receipts"] = tmp_path / "dry-receipts"
    yield world
    PATHS.reset_settings_cache()


def run_single(w, *, apply: bool, **extra):
    tx = Q.Single(url=SINGLE_URL, collection=TARGET, payload_profile_of=TWIN,
                  state_dir=w["state"])
    return Q.drain(w["queue"], transaction=tx, task="single-dry-test", apply=apply, batch=10,
                   pause_s=0, artifact_root=w["artifacts"], contract_path=w["contract"],
                   refresh_index=False, **extra)


def everything(w) -> dict:
    """Every place a dry `Single` could write: queue, artifact root, state dir, receipts dir."""
    return {name: (tree(p) if p.exists() else None)
            for name, p in (("queue", w["queue"]), ("artifacts", w["artifacts"]),
                            ("state", w["state"]), ("dry_receipts", w["dry_receipts"]))}


def assert_wrote_nothing(w, before):
    assert w["fake"].writes == [], "a dry Single wrote to Qdrant"
    assert w["ssh"] == [], "a dry Single ran ssh (a writer lock)"
    assert everything(w) == before, "a dry Single wrote a file"
    assert not w["state"].exists(), "a dry Single created its state dir (a shadow pack copy)"


def test_a_dry_single_over_a_fresh_pack_writes_nothing_anywhere(single_world, capsys):
    before = everything(single_world)
    (rec,) = run_single(single_world, apply=False)
    assert rec["status"] == "PROVED-DRY-RUN", rec
    assert rec["transaction_key"] == f"single|{SINGLE_URL}|{TARGET}"
    assert_wrote_nothing(single_world, before)
    assert "dry-run receipt (not written to the queue)" in capsys.readouterr().out


@pytest.mark.parametrize("own_artifact", [True, False], ids=["own-artifact", "live-twin-diff"])
def test_a_dry_single_over_a_resume_state_pack_writes_nothing_anywhere(single_world, own_artifact):
    """The queue holds the twin-first drain's resume state for the pack (twin written, prod owed,
    its ledger-side artifact, a cached refusal from an apply run). The Single reads none of it as
    its own, and writes none of it either."""
    w = single_world
    twin_written_prod_owed(w, own_artifact=own_artifact)
    pack = w["queue"] / MANIFEST["pack_id"]
    (pack / "receipt.json").write_text(json.dumps({"status": "ABORTED", "stage": "live-apply"}))
    (w["artifacts"] / "ledger.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (w["artifacts"] / "ledger.jsonl").write_text('{"status": "TWIN-DONE"}\n')
    before = everything(w)
    (rec,) = run_single(w, apply=False, dry_receipt_dir=None)
    assert rec["status"] == "PROVED-DRY-RUN", rec
    assert_wrote_nothing(w, before)


def test_a_dry_single_over_a_half_written_target_refuses_and_writes_nothing(single_world):
    w = single_world
    for pid in PACK_IDS[:1]:
        w["fake"].points[(SINGLE_URL, TARGET)][pid] = {"id": pid}
    before = everything(w)
    (rec,) = run_single(w, apply=False)
    assert rec["status"] == "REFUSED" and "partly written" in rec["reason"], rec
    assert_wrote_nothing(w, before)


def test_the_target_already_holds_every_id_branch_needs_apply(single_world, capsys):
    """The branch wrote DONE with no look at `--apply`: under the dry guard it killed the drain
    (a refused write), and a guard-less caller got a DONE from a dry run."""
    w = single_world
    for pid in PACK_IDS:
        w["fake"].points[(SINGLE_URL, TARGET)][pid] = {"id": pid}
    before = everything(w)
    (rec,) = run_single(w, apply=False)
    assert rec["status"] == "PROVED-DRY-RUN" and rec["would"].startswith("mark DONE"), rec
    assert_wrote_nothing(w, before)
    assert "would mark DONE" in capsys.readouterr().out
    # control: the same state with --apply marks DONE, in the state dir and nowhere else
    (rec,) = run_single(w, apply=True)
    assert rec["status"] == "LANDED" and "already held every id" in rec["note"], rec
    assert (w["state"] / MANIFEST["pack_id"] / "DONE").exists()
    assert not (w["queue"] / MANIFEST["pack_id"] / "DONE").exists()
    assert w["fake"].writes == []


def test_a_dry_single_writes_its_full_receipt_only_to_the_dry_receipt_dir(single_world):
    w = single_world
    before = everything(w)
    (rec,) = run_single(w, apply=False, dry_receipt_dir=w["dry_receipts"])
    got = json.loads((w["dry_receipts"] / f"{MANIFEST['pack_id']}.receipt.json").read_text())
    assert got["status"] == "PROVED-DRY-RUN" and got["transaction_key"] == rec["transaction_key"]
    after = everything(w)
    assert after.pop("dry_receipts") is not None and before.pop("dry_receipts") is None
    assert after == before, "a dry Single wrote outside its dry receipt dir"
    assert w["fake"].writes == [] and w["ssh"] == []
    assert not w["state"].exists()
    with pytest.raises(Q.QueueError, match="inside the queue"):
        run_single(w, apply=False, dry_receipt_dir=w["queue"] / "x")


def test_the_apply_only_primitive_of_the_transaction_refuses_under_the_guard(tmp_path):
    tx = Q.Single(url=SINGLE_URL, collection=TARGET)
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            Q.single_apply(pack_id="p", manifest={}, single=tx, task="t", batch=1,
                           artifact_root=tmp_path / "art")
    assert not (tmp_path / "art").exists()


def test_a_dry_single_refuses_a_live_endpoint_and_still_writes_nothing(single_world):
    """The endpoint check is a READ: it runs in a dry run too, and refuses before any pack."""
    w = single_world
    tx = Q.Single(url=PROD_URL, collection="aw_innocent_name", state_dir=w["state"])
    before = everything(w)
    with pytest.raises(Q.QueueError, match="LIVE endpoint"):
        Q.drain(w["queue"], transaction=tx, task="t", apply=False, batch=10, pause_s=0,
                artifact_root=w["artifacts"], contract_path=w["contract"], refresh_index=False)
    assert everything(w) == before and w["fake"].writes == []
