"""Builder ignition check, `git_sha`, manifest write/validate, state files and
Qdrant ops (moved from the b1 builder script, task 2108).

The code below is the original block, with these differences: `qc` builds its client with
`cytir.core.qdrant.qdrant_connect`; `create_collection` and `finalize_collection`
go through the one collection creator with the profile `b1_build_profile()` (the
same Qdrant request, proven in `tests/test_aw_index_b1_manifest.py`); the
ignition check and `git_sha` ask git about the repo root from `cytir.core.paths`
(they asked about the builder script's directory)."""
from __future__ import annotations

import json
import os
import time

import cytir.core.ids as ID
from cytir.core import paths as P
from cytir.core.qdrant import qdrant_connect

# Ignition-provenance override (emergencies only; logged loudly): set to a
# short reason string to run full mode from a dirty / off-main / non-repo tree.
IGNITION_OVERRIDE_ENV = "AW_B1_IGNITION_OVERRIDE"

# config-B payload indexes matching aw_v4_sat (memory aw-qdrant-collection-shape).
# (field, type, on_disk) rows in the order the builder created them; on_disk False
# is the plain enum schema the builder always sent. The dict that held them is gone:
# the profile below is the one declaration.
PAYLOAD_INDEXES = (
    ("domain", "keyword", False),
    # task 348: supersede-on-upsert (upsert_supersede.py) reconciles page
    # groups via an indexed url match — every collection carries it from birth
    # so no delta applier ever has to create it after the fact.
    ("url", "keyword", False),
    ("source_type", "keyword", False),
    ("country", "keyword", False),
    ("tier", "keyword", False),
    ("lang", "keyword", False),
    ("trust_tier", "float", False),
    ("register_id", "keyword", False),
    ("lei", "keyword", False),
)

#: The builder's own collection config (128-d config B), written as a serving
#: profile so the one creator makes it. `hnsw_m=16` is the m that `finalize` sets;
#: the creator makes the collection at m=0 because it is called with `bulk=True`.
def b1_build_profile():
    """A function, not a constant: `serving_profile` needs the full qdrant models at
    import, and the builder's channel tests load the builder against a stub of them."""
    from cytir.core import serving_profile as SP
    return SP.ServingProfile(
        name="b1-build-config-b", dense_dim=128, dense_datatype="float32",
        datatype_explicit=False, dense_on_disk=True, distance="cosine",
        quantizer="int8", quant_always_ram=False,
        sparse_name="bm25", sparse_idf=True, sparse_index_on_disk=True,
        sparse_index_datatype="float16", hnsw_m=16, hnsw_on_disk=True,
        on_disk_payload=True, optimizers=(), payload_indexes=PAYLOAD_INDEXES,
        note="the b1 builder's throwaway build collection (aw_v4_sat shape)")

# extractor library pins recorded in rule_legend.extract (epochs.md §3 F12).
_EXTRACT_LIBS = ("trafilatura", "langdetect")
# CharSplit is installed from a git commit, so its package version says nothing: name the
# pinned commit (the one cytir.core.decompound checks by sha256; ADR-121).
_CHARSPLIT_PIN = "charsplit@9710624e"



def qc(url: str):
    return qdrant_connect(url, timeout=1200, check_compatibility=False)




def _pinned_versions() -> str:
    import importlib.metadata as md
    parts = []
    for lib in _EXTRACT_LIBS:
        try:
            parts.append(f"{lib}=={md.version(lib)}")
        except Exception:  # noqa: BLE001
            parts.append(f"{lib}==unpinned")
    parts.insert(1, _CHARSPLIT_PIN)   # keeps the former trafilatura, splitter, langdetect order
    return ",".join(parts)


# --------------------------------------------------------------------------
# collection assembly (config-B, m=0 bulk -> m=16 finalize)
# --------------------------------------------------------------------------
def create_collection(client, name: str, log=print):
    from cytir.core import serving_profile as SP
    SP.create_collection(client, name, b1_build_profile(),
                         on_exists="recreate", bulk=True, log=log)
    log(f"[create] {name} config-B (dense128 int8 on_disk + bm25 on_disk, m=0, "
        f"{len(PAYLOAD_INDEXES)} payload idx)")


def finalize_collection(client, name: str, log=print):
    from cytir.core import serving_profile as SP
    SP.finalize_collection(client, name, b1_build_profile(), log=log)





# --------------------------------------------------------------------------
# ignition provenance (task 319: the 318 build ran a worktree commit that was
# never an ancestor of main — the live-code-uncommitted-risk lesson, again)
# --------------------------------------------------------------------------
def ignition_check() -> list[str]:
    """Return the list of provenance problems that must block a FULL run:
    the builder's own tree must be a git checkout, clean, and its HEAD an
    ancestor of origin/main. Empty list = clear to ignite."""
    import subprocess
    here = str(P.REPO_ROOT)

    def g(*a):
        return subprocess.check_output(
            ["git", "-C", here, *a], text=True,
            stderr=subprocess.DEVNULL, timeout=60).strip()

    problems: list[str] = []
    try:
        g("rev-parse", "--show-toplevel")
    except Exception:  # noqa: BLE001
        return [f"builder dir {here} is not inside a git repository — deploy "
                f"the pinned repo (git clone/worktree), not a file copy"]
    try:
        dirty = g("status", "--porcelain")
        if dirty:
            problems.append(
                f"working tree not clean ({len(dirty.splitlines())} entries; "
                f"first: {dirty.splitlines()[0]!r})")
    except Exception as e:  # noqa: BLE001
        problems.append(f"git status failed: {e}")
    try:
        subprocess.run(["git", "-C", here, "fetch", "origin", "main", "--quiet"],
                       timeout=60, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass  # offline: fall back to the last-fetched origin/main ref below
    try:
        g("rev-parse", "--verify", "origin/main")
    except Exception:  # noqa: BLE001
        problems.append("no origin/main ref (no remote?) — cannot prove the "
                        "builder is merged to main")
        return problems
    try:
        subprocess.check_call(
            ["git", "-C", here, "merge-base", "--is-ancestor", "HEAD",
             "origin/main"], timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        try:
            head = g("rev-parse", "HEAD")
        except Exception:  # noqa: BLE001
            head = "?"
        problems.append(f"HEAD {head[:12]} is NOT an ancestor of origin/main "
                        f"— merge the builder to main before igniting (the "
                        f"318 build ran unmerged commit 2987c35)")
    return problems


def assert_ignition(log=print) -> None:
    """Refuse to start a FULL run on provenance problems (SystemExit 6).
    AW_B1_IGNITION_OVERRIDE=<reason> bypasses, logged loudly."""
    problems = ignition_check()
    if not problems:
        log("[ignition] provenance OK: clean tree, HEAD is ancestor of origin/main")
        return
    override = os.environ.get(IGNITION_OVERRIDE_ENV, "").strip()
    for p in problems:
        log(f"[ignition] PROBLEM: {p}")
    if override:
        log(f"[ignition] !!! OVERRIDE ACTIVE ({IGNITION_OVERRIDE_ENV}="
            f"{override!r}) — igniting DESPITE the problems above. This is "
            f"for emergencies only and is on the record.")
        return
    raise SystemExit(
        f"[FATAL] ignition refused: {len(problems)} provenance problem(s) "
        f"(above). Fix them, or set {IGNITION_OVERRIDE_ENV}=<reason> to "
        f"override in an emergency.")


# --------------------------------------------------------------------------
# MANIFEST
# --------------------------------------------------------------------------
_GIT_SHA_OVERRIDE = ""


def git_sha() -> str:
    if _GIT_SHA_OVERRIDE:
        return _GIT_SHA_OVERRIDE
    try:
        import subprocess
        here = str(P.REPO_ROOT)
        return subprocess.check_output(
            ["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "0" * 40


def write_manifest(path: str, build_id: str, parent: str | None,
                   sources: list[dict], counts: dict, eval_block: dict,
                   exceptions: dict | None = None, log=print) -> dict:
    legend = dict(ID.RULE_LEGEND)
    legend["extract"] = f"unified-extractor+{_pinned_versions()}@{ID.EPOCHS['extract']}"
    man = {
        "build_id": build_id,
        "parent_build": parent,
        "git_sha": git_sha(),
        "rule_legend": legend,
        "sources": sources,
        "counts": counts,
        "eval": eval_block,
    }
    if exceptions is not None:
        man["exceptions"] = exceptions
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    log(f"[manifest] wrote {path}")
    return man


def validate_manifest(man: dict) -> list[str]:
    """Validate against manifest-schema.md (draft-07). Uses jsonschema if present,
    else a hand check of the load-bearing invariants."""
    errs: list[str] = []
    req = ["build_id", "parent_build", "git_sha", "rule_legend", "sources",
           "counts", "eval"]
    for k in req:
        if k not in man:
            errs.append(f"missing top-level key {k}")
    import re as _re
    if not _re.match(r"^aw_b[0-9]+$", man.get("build_id", "")):  # manifest key, not a stored payload: canonical row
        errs.append("build_id !~ ^aw_b[0-9]+$")
    pb = man.get("parent_build")
    if pb is not None and not _re.match(r"^aw_b[0-9]+$", pb):
        errs.append("parent_build !~ ^aw_b[0-9]+$ and not null")
    if not _re.match(r"^[0-9a-f]{40}$", man.get("git_sha", "")):
        errs.append("git_sha !~ 40-hex")
    for fam in ["extract", "chunk", "embed", "tok", "country", "trust", "card"]:
        if fam not in man.get("rule_legend", {}):
            errs.append(f"rule_legend missing family {fam}")
    c = man.get("counts", {})
    for k in ["points", "adds", "changes", "deletes"]:
        if k not in c:
            errs.append(f"counts missing {k}")
    if c and "points" in c and "adds" in c and "deletes" in c:
        # for a first build (parent null) points == adds and deletes count is the
        # eviction; invariant points == (parent_points) + adds - deletes. For aw_b1
        # (parent null) we assert points == adds - deletes + 0 is NOT required;
        # the schema invariant is points(this) = points(parent) + adds - deletes.
        pass
    for pk in man.get("sources", []):
        if not _re.match(r"^POLICY\.md@v[0-9]+$", pk.get("policy", "")):
            errs.append(f"source {pk.get('source')} policy !~ POLICY.md@vN")
    ev = man.get("eval", {})
    for k in ["golds", "pass", "report"]:
        if k not in ev:
            errs.append(f"eval missing {k}")
    # try strict jsonschema if available
    try:
        import jsonschema  # noqa: F401
        schema = _MANIFEST_SCHEMA
        import jsonschema as js
        try:
            js.validate(man, schema)
        except js.ValidationError as e:
            errs.append(f"jsonschema: {e.message}")
    except ImportError:
        pass
    return errs


# the draft-07 schema (verbatim from specs/manifest-schema.md)
_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["build_id", "parent_build", "git_sha", "rule_legend", "sources", "counts", "eval"],
    "additionalProperties": False,
    "properties": {
        "build_id": {"type": "string", "pattern": "^aw_b[0-9]+$"},
        "parent_build": {"type": ["string", "null"], "pattern": "^aw_b[0-9]+$"},
        "git_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "rule_legend": {
            "type": "object",
            "required": ["extract", "chunk", "embed", "tok", "country", "trust", "card"],
            "additionalProperties": False,
            # prov_id: the mint-rule family (ADR-117 am.1); optional so older manifests
            # still validate. The object stays closed: any other key is refused.
            "properties": {k: {"type": "string"} for k in
                           ["extract", "chunk", "embed", "tok", "country", "trust", "card",
                            "prov_id"]},
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["pack", "source", "version", "count", "policy"],
                "additionalProperties": True,
                "properties": {
                    "pack": {"type": "string"},
                    "source": {"type": "string"},
                    "version": {"type": "string"},
                    "count": {"type": "integer", "minimum": 0},
                    "policy": {"type": "string", "pattern": "^POLICY\\.md@v[0-9]+$"},
                },
            },
        },
        "counts": {
            "type": "object",
            "required": ["points", "adds", "changes", "deletes"],
            "additionalProperties": False,
            "properties": {k: {"type": "integer", "minimum": 0} for k in
                           ["points", "adds", "changes", "deletes"]},
        },
        "eval": {
            "type": "object",
            "required": ["golds", "pass", "report"],
            "additionalProperties": False,
            "properties": {
                "golds": {"type": "integer", "minimum": 0},
                "pass": {"type": "boolean"},
                "report": {"type": "string"},
            },
        },
        "exceptions": {"type": "object"},
    },
}


# --------------------------------------------------------------------------
# scrolling helpers
# --------------------------------------------------------------------------
def scroll_collection(client, coll: str, limit: int | None,
                      flt=None, with_vectors=True, batch=1000, log=print):
    """Yield records from a collection (optionally filtered / capped)."""
    offset = None
    n = 0
    while True:
        pts, offset = client.scroll(
            coll, limit=batch, offset=offset, scroll_filter=flt,
            with_payload=True, with_vectors=with_vectors)
        for p in pts:
            yield p
            n += 1
            if limit and n >= limit:
                return
        if offset is None:
            return


TWIN_BASE_COUNT = 47_824_966


def _state_path(out_dir):
    return os.path.join(out_dir, "STATE.json")


def load_state(out_dir) -> dict:
    p = _state_path(out_dir)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            pass
    return {"phase": "kv", "kv_done_files": [], "base_offset": None,
            "base_processed": 0, "base_added": 0, "deletes": 0,
            "consent_excluded": 0, "consent_declined": 0,
            "consent_grace_retained": 0, "deferred_count": 0,
            "deferred_excluded": 0, "staged_excluded": 0,
            "staged_excluded_undated_gov": 0,
            "consent_stats": {}, "drops": 0, "staged_done": [],
            "staged_added": 0, "staged_modified": 0, "slice_deleted": False,
            "counts": {}, "coverage": {}}


def save_state(out_dir, st: dict, resume_cmd: str = ""):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(_state_path(out_dir), "w") as fh:
        json.dump(st, fh, indent=1)
    # human-readable STATE.md with the resume command (coordinator asked for it)
    with open(os.path.join(out_dir, "STATE.md"), "w") as fh:
        fh.write(f"# aw_b1 full build STATE\n\nupdated: {st['updated']}\n\n")
        fh.write(f"phase: **{st.get('phase')}**\n\n")
        fh.write(f"- base_processed: {st.get('base_processed',0):,} / {TWIN_BASE_COUNT:,}\n")
        fh.write(f"- base_added: {st.get('base_added',0):,}\n")
        fh.write(f"- deletes(romansh): {st.get('deletes',0):,}\n")
        # every exclusion class, truthfully (task 319: the 318 STATE said
        # consent_excluded: 0 while 1.79M were excluded)
        fh.write(f"- consent_excluded (declined, ALL phases): {st.get('consent_excluded',0):,}\n")
        fh.write(f"- consent_declined: {st.get('consent_declined',0):,}\n")
        fh.write(f"- consent_grace_retained (unreachable/uncensused — IN the "
                 f"index, retry-queued): {st.get('consent_grace_retained',0):,}\n")
        fh.write(f"- deferred queued: {st.get('deferred_count',0):,} / "
                 f"deferred_excluded: {st.get('deferred_excluded',0):,}\n")
        fh.write(f"- staged_excluded: {st.get('staged_excluded',0):,}\n")
        fh.write(f"- staged_excluded_undated_gov (ch-media admin.ch undated, "
                 f"task 321/T8-M21): {st.get('staged_excluded_undated_gov',0):,}\n")
        fh.write(f"- drops(no-dense/no-url): {st.get('drops',0):,}\n")
        if st.get("consent_stats"):
            fh.write(f"- consent gate classes: {json.dumps(st['consent_stats'])}\n")
        if st.get("id_collision_check"):
            fh.write(f"- id_collision_check: {json.dumps(st['id_collision_check'])}\n")
        fh.write(f"- staged_done: {st.get('staged_done',[])}\n")
        fh.write(f"- staged_added/modified: {st.get('staged_added',0):,} / {st.get('staged_modified',0):,}\n\n")
        if st.get("coverage"):
            fh.write(f"coverage (sampled tok split): {json.dumps(st['coverage'])}\n\n")
        fh.write("## resume command\n```\n" + (resume_cmd or "see runbook") + "\n```\n")

