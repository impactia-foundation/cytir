"""cytir.serve.settings -- the serving process's configuration: typed, declared once, loaded once.

S10 step 2 (task 1071; spec `specs/s10-serve-split.md` section 2). Every `AW_*` environment
variable the server reads is one field below: its env name, its kind (the exact coercion the old
`os.environ.get` read applied), its code default, for 12 fields the fusion-weights key that
supplies the default, and `why` (the task or ADR the old read cited). `SETTINGS = Settings.load()`
runs once, at import. The modules keep their module globals (tests and gates set them) and assign
them from `SETTINGS`.

Parsing keeps today's semantics exactly; a stricter parser is a behaviour change:
  flag    raw == "1"                      (a flag set to `true` is OFF; `parse_warnings` says so)
  truthy  raw.strip().lower() in 1/true/yes/on (the tokenizer's reading)
  float   float(raw)   int  int(raw)      (a value that does not parse fails the import, as before)
  str     raw, verbatim (the caller's own strip/split/lower stays at the read site)
  str_or_weights  env value if non-empty, else the weights file, else the code default
`raw` is the env value when the name is SET (even to ""), else the default: `os.environ.get(name,
default)`. Precedence for the weights fields: env > weights file > code (today's order).

Sources in `/config`: `env`, `weights:<key>` (the key is in the loaded weights file), `default`.
The process cannot tell a compose base value from an overlay value: both are `env`. Owners: most
fields are read through `SETTINGS`; `schema` (task 1180) reads the tokenizer's fields with
`current(name)` at its own import, so a test can load a fresh `schema` under a patched env;
`cytir.core` marks a name only `cytir.core` reads (echoed).
`call_time` fields are re-read at each call, as before (`current()`): the Qdrant URL and the
weights path. Imports: none from the server (this module is a leaf).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from cytir.core import envnames as _envnames

SCHEMA = "aw.serve-config.v1"
_HERE = os.path.dirname(__file__)
_HERE_ABS = os.path.dirname(os.path.abspath(__file__))
_LOG = logging.getLogger("aw_server.settings")
#: The sha of the release this tree was rendered from. `release-serving.sh` and
#: `rebuild-serving-image.sh` write it next to the code (the box's `.serving-release` is outside
#: the container); a checkout or a test run has no such file.
SERVING_RELEASE_FILE = os.path.join(_HERE_ABS, "SERVING_RELEASE")

FLAG, TRUTHY, FLOAT, INT, STR, STR_OR_WEIGHTS = "flag", "truthy", "float", "int", "str", "str_or_weights"
_PY_TYPE = {FLAG: bool, TRUTHY: bool, FLOAT: float, INT: int, STR: str, STR_OR_WEIGHTS: str}


def knob(env: str | None, kind: str, default: Any, *, weights: str | None = None, why: str = "",
         owner: str = "settings", call_time: bool = False):
    """One declared setting. `env=None`: no env override (the weights file or the code sets it)."""
    return field(default=None, metadata={"env": env, "kind": kind, "default": default,
                                         "weights": weights, "why": why, "owner": owner,
                                         "call_time": call_time})


def _truthy(v: "str | None") -> bool:
    """The tokenizer's reading of a truthy knob (`AW_FOLD_ACCENTS`, `AW_BM25_QSTOP`)."""
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def class_set(raw: str) -> frozenset:
    """A comma list of source classes as the levers read it: stripped, lower-cased, no blanks."""
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


def _read_serving_release(path: str = SERVING_RELEASE_FILE) -> str | None:
    """The release sha rendered next to this code, or None when the file is absent or empty."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class Runtime:
    """Where the process talks to and what it loads: Qdrant, TEI, the embedder, the artifact
    paths (weights, heads, tables), the HTTP shell, and the two read-only debug instruments."""
    collection: str = knob("AW_COLLECTION", STR, "aw_passages", why="task 182")
    qdrant_url: str = knob("AW_QDRANT_URL", STR, "http://localhost:6333", why="task 182", call_time=True)
    qdrant_timeout: float = knob("AW_QDRANT_TIMEOUT", FLOAT, "5", why="task 342")
    qdrant_health_timeout: float = knob("AW_QDRANT_HEALTH_TIMEOUT", FLOAT, "2", why="task 342")
    cards_collection: str = knob("AW_CARDS_COLLECTION", STR, "source_cards", why="ADR-021")
    tei_url: str = knob("AW_TEI_URL", STR, "", why="task 070")
    tei_timeout: float = knob("AW_TEI_TIMEOUT", FLOAT, "10", why="task 070; S10 D5: kept typed")
    embedder: str = knob("AW_EMBEDDER", STR, "arctic", why="ADR-018")
    mrl_dim: int = knob("AW_MRL_DIM", INT, "128", why="ADR-018")
    rerank: bool = knob("AW_RERANK", FLAG, "1", why="ADR-012")
    embed_memo: bool = knob("AW_EMBED_MEMO", FLAG, "0", why="task 342")
    models_dir: str = knob("AW_MODELS_DIR", STR, None, why="read by cytir.core (model assets)", owner="cytir.core")
    webapp: str = knob("AW_WEBAPP", STR, "/srv/webapp/index.html", why="the human search page")
    api_url: str = knob("AW_API_URL", STR, "http://localhost:8080", why="aw-mcp -> aw-api")
    mcp_port: int = knob("AW_MCP_PORT", INT, "8081", why="aw-mcp")
    mcp_host: str = knob("AW_MCP_HOST", STR, "0.0.0.0", why="aw-mcp bind address; the default flips to 127.0.0.1 in a later step (task 3008, Owed)")
    fusion_weights: str = knob("AW_FUSION_WEIGHTS", STR, os.path.join(_HERE, "fusion_weights.json"),
                               why="task 522", call_time=True)
    rankhead_path: str = knob("AW_RANKHEAD_PATH", STR, os.path.join(_HERE, "rankhead.json"), why="ADR-036")
    rankhead_v2_path: str = knob("AW_RANKHEAD_V2_PATH", STR, os.path.join(_HERE, "rankhead_v2.json"),
                                 why="ADR-080")
    fedlex_versions_path: str = knob("AW_FEDLEX_VERSIONS", STR, "", why="task 2709")
    qhead_path: str = knob("AW_QHEAD_PATH", STR, os.path.join(_HERE, "qhead.json"), why="ADR-062")
    hubfloor_table: str = knob("AW_HUBFLOOR_TABLE", STR, os.path.join(_HERE, "hub_hosts.json"), why="ADR-031")
    authority_table: str = knob("AW_AUTHORITY_TABLE", STR, os.path.join(_HERE, "authority_hosts.json"),
                                why="ADR-031; S10 D5: kept typed")
    gazetteer_path: str = knob("AW_GAZETTEER_PATH", STR, os.path.join(_HERE, "geo_gazetteer.json"),
                               why="ADR-045; S10 D5: kept typed")
    parasite_domains: str = knob("AW_PARASITE_DOMAINS", STR, os.path.join(_HERE, "parasite_domains.json"),
                                 why="task 125; S10 D5: kept typed")
    country_map: str = knob("AW_COUNTRY_MAP", STR, "", why="task 261")
    bm25_idf_table: str = knob("AW_BM25_IDF_TABLE", STR, os.path.join(_HERE_ABS, "bm25_idf.json"),
                               why="ADR-109")
    debug_scores: bool = knob("AW_DEBUG_SCORES", FLAG, "0", why="task 289")
    qhead_country_trace: bool = knob("AW_QHEAD_COUNTRY_TRACE", FLAG, "0", why="ADR-084")
    rankhead_v2_dump: bool = knob("AW_RANKHEAD_V2_DUMP", FLAG, "0", why="ADR-080")


@dataclass(frozen=True)
class Query:
    """Query understanding: the tokenizer, the learned query heads, intent levers."""
    fold_accents: bool = knob("AW_FOLD_ACCENTS", TRUTHY, None, why="task 290")
    bm25_qstop: bool = knob("AW_BM25_QSTOP", TRUTHY, None, why="ADR-109")
    bm25_qstop_idf: float = knob("AW_BM25_QSTOP_IDF", FLOAT, "1.40", why="ADR-109")
    bm25_qstop_mintokens: int = knob("AW_BM25_QSTOP_MINTOKENS", INT, "4", why="ADR-109")
    query_glued_article: bool = knob("AW_QUERY_GLUED_ARTICLE", FLAG, "0", call_time=True, why="task 2778")
    query_fused_split: bool = knob("AW_QUERY_FUSED_SPLIT", FLAG, "0", call_time=True, why="task 2856")
    article_exact_w: float = knob("AW_ARTICLE_EXACT_W", FLOAT, "0", call_time=True, why="task 2792")
    article_exact_statute: bool = knob("AW_ARTICLE_EXACT_STATUTE", FLAG, "0", call_time=True, why="task 2797")
    qhead: bool = knob("AW_QHEAD", FLAG, "0", why="ADR-036")
    qhead_knobs: str = knob("AW_QHEAD_KNOBS", STR, "all", why="ADR-062")
    rankhead: bool = knob("AW_RANKHEAD", FLAG, "0", why="ADR-036")
    rankhead_v2: bool = knob("AW_RANKHEAD_V2", FLAG, "0", why="ADR-080")
    country_scope_mode: str = knob("AW_COUNTRY_SCOPE_MODE", STR, "legacy", why="task 374")
    commercial: bool = knob("AW_COMMERCIAL", FLAG, "0", why="task 254")
    commercial_ref_w: float = knob(None, FLOAT, 0.0, weights="commercial_ref", why="task 254")
    commercial_gov_scale: float = knob(None, FLOAT, 1.0, weights="commercial_gov_scale", why="task 254")
    version_match: bool = knob("AW_VERSION_MATCH", FLAG, "0", why="task 665")
    version_match_w: float = knob("AW_VERSION_MATCH_W", FLOAT, "0.0", why="task 665")
    host_query_card: bool = knob("AW_HOST_QUERY_CARD", FLAG, "0", why="task 2050 (arm 4 of 2028; mechanism task 2044)")


@dataclass(frozen=True)
class Legs:
    """Stage 1: the dense and BM25 legs, pool sizes, the page cap, the country partitions."""
    dense_rescore: bool = knob("AW_DENSE_RESCORE", FLAG, "0", why="task 516")
    dense_oversampling: float = knob("AW_DENSE_OVERSAMPLING", FLOAT, "1.0", why="task 516")
    dense_ignore_quant: bool = knob("AW_DENSE_IGNORE_QUANT", FLAG, "0", why="task 516")
    dense_hnsw_ef: int = knob("AW_DENSE_HNSW_EF", INT, "0", why="task 328")
    two_stage: bool = knob("AW_TWO_STAGE", FLAG, "0", why="ADR-025")
    kstab: bool = knob("AW_KSTAB", FLAG, "1", why="ADR-035")
    pagecap: int = knob("AW_PAGECAP", INT, "0", why="task 582")
    pagecap_mode: str = knob("AW_PAGECAP_MODE", STR, "cap", why="task 582")
    pagecap_oversample: float = knob("AW_PAGECAP_OVERSAMPLE", FLOAT, "1.0", why="task 582")
    authleg: bool = knob("AW_AUTHLEG", FLAG, "0", why="task 202")
    authleg_filtered_timeout: int = knob("AW_AUTHLEG_FILTERED_TIMEOUT", INT, "0", why="task 2665")
    stage1_timeout: int = knob("AW_STAGE1_TIMEOUT", INT, "0", why="task 2686")
    batch_legs: bool = knob("AW_BATCH_LEGS", FLAG, "0", why="task 342")
    qhead_country_legs: bool = knob("AW_QHEAD_COUNTRY_LEGS", FLAG, "0", why="ADR-084")
    bm25_per_token: bool = knob("AW_BM25_PER_TOKEN", FLAG, "0", why="task 083")
    bm25_per_token_limit: int = knob("AW_BM25_PER_TOKEN_LIMIT", INT, "50", why="task 083")
    bm25_zero_guard: bool = knob("AW_BM25_ZERO_GUARD", FLAG, "0", why="task 606")


@dataclass(frozen=True)
class Fusion:
    """Stage 2 post-row levers: name gate, ref-canon, primacy, name-dense, name coverage, the
    required-term lever."""
    namegate: bool = knob("AW_NAMEGATE", FLAG, "0", why="task 404")
    namegate_w: float = knob("AW_NAMEGATE_W", FLOAT, "0.0", why="task 404")
    namegate_cov: float = knob("AW_NAMEGATE_COV", FLOAT, "0.999", why="task 404")
    namegate_maxterms: int = knob("AW_NAMEGATE_MAXTERMS", INT, "4", why="task 404")
    namegate_trustkeep: bool = knob("AW_NAMEGATE_TRUSTKEEP", FLAG, "0", why="task 406")
    namegate_shape_v2: bool = knob("AW_NAMEGATE_SHAPE_V2", FLAG, "0", why="task 406")
    refcanon: bool = knob("AW_REFCANON", FLAG, "0", why="task 404")
    refcanon_w: float = knob("AW_REFCANON_W", FLOAT, "0.0", why="task 404")
    refcanon_beta: float = knob("AW_REFCANON_BETA", FLOAT, "0.7", why="task 404")
    refcanon_maxbeta: float = knob("AW_REFCANON_MAXBETA", FLOAT, "0.25", why="task 404")
    refcanon_mincent: float = knob("AW_REFCANON_MINCENT", FLOAT, "0.0", why="task 406")
    refcanon_extra_classes: str = knob("AW_REFCANON_EXTRA_CLASSES", STR, "", why="task 404")
    refcanon_targets: str = knob("AW_REFCANON_TARGETS", STR, "reference", why="task 578")
    primacy: bool = knob("AW_PRIMACY", FLAG, "0", why="task 578")
    primacy_keep: float = knob("AW_PRIMACY_KEEP", FLOAT, "0.0", why="task 578")
    primacy_beta: float = knob("AW_PRIMACY_BETA", FLOAT, "0.5", why="task 578")
    primacy_mincent: float = knob("AW_PRIMACY_MINCENT", FLOAT, "0.3", why="task 578")
    primacy_extra_classes: str = knob("AW_PRIMACY_EXTRA_CLASSES", STR, "docs,model-hub", why="task 578")
    namedense: bool = knob("AW_NAMEDENSE", FLAG, "0", why="task 406")
    namedense_scale: float = knob("AW_NAMEDENSE_SCALE", FLOAT, "1.0", why="task 406")
    namecov_w: float = knob("AW_NAMECOV_W", FLOAT, "0.0", why="task 406")
    namecov_surface: str = knob("AW_NAMECOV_SURFACE", STR, "display", why="task 411")
    reqterm_w: float = knob("AW_REQTERM_W", FLOAT, "0.0", why="task 411")


@dataclass(frozen=True)
class Trust:
    """How trust and authority enter the score: the trust mode, the trust floors and gates, the
    authority boost, the hub floor and gate, the relevance floor, official/entity preferences."""
    trust_mode: str = knob("AW_TRUST_MODE", STR_OR_WEIGHTS, "additive", weights="trust_mode", why="task 103")
    trust_alpha: float = knob(None, FLOAT, 1.0, weights="trust_alpha", why="task 103")
    trust_gate_pctl: float = knob(None, FLOAT, 0.6, weights="trust_gate_pctl", why="task 103")
    trust_relgate: bool = knob("AW_TRUST_RELGATE", FLAG, "0", why="ADR-030")
    trust_relgate_beta: float = knob("AW_TRUST_RELGATE_BETA", FLOAT, "0.5", why="ADR-030")
    official_pref_beta: float = knob(None, FLOAT, 0.5, weights="official_pref_beta", why="task 144")
    legifrance_bulk_relgate: float = knob(None, FLOAT, 1.0, weights="legifrance_bulk_relgate", why="task 165")
    entity_card_beta: float = knob(None, FLOAT, 0.5, weights="entity_card_beta", why="task 163")
    entity_card_name_thr: float = knob(None, FLOAT, 0.6, weights="entity_card_name_thr", why="task 163")
    entity_card_cap: int = knob(None, INT, 1, weights="entity_card_cap", why="task 163")
    relfloor: bool = knob("AW_RELFLOOR", FLAG, "0", why="task 207")
    hubgate: bool = knob("AW_HUBGATE", FLAG, "0", why="task 083")
    hubfloor: bool = knob("AW_HUBFLOOR", FLAG, "0", why="ADR-031")
    hubfloor_w: float = knob("AW_HUBFLOOR_W", FLOAT, "0.0", why="ADR-031")
    hubfloor_minbeta: float = knob("AW_HUBFLOOR_MINBETA", FLOAT, "0.5", why="ADR-031")
    hubfloor_distinct_max: float = knob("AW_HUBFLOOR_DISTINCT_MAX", FLOAT, "0.0", why="ADR-031")
    hubfloor_dense_max: float = knob("AW_HUBFLOOR_DENSE_MAX", FLOAT, "1.0", why="ADR-031")
    trustfloor: bool = knob("AW_TRUSTFLOOR", FLAG, "0", why="task 207")
    trustfloor_tau: float = knob("AW_TRUSTFLOOR_TAU", FLOAT, "0.6", why="task 207")
    trustfloor_minbeta: float = knob("AW_TRUSTFLOOR_MINBETA", FLOAT, "0.24", why="task 207")
    trustfloor_pool_max: float = knob("AW_TRUSTFLOOR_POOL_MAX", FLOAT, "0.66", why="task 289")
    trustfloor_rel_beta: float = knob("AW_TRUSTFLOOR_REL_BETA", FLOAT, "0.7", why="task 289")
    trustfloor_abs: bool = knob("AW_TRUSTFLOOR_ABS", FLAG, "0",
                                  why="ADR-099; a FLAG (== \"1\") for the floor and for schema's "
                                      "WARNING log, so true/yes/on never armed either")
    trustfloor_undemoted: bool = knob("AW_TRUSTFLOOR_UNDEMOTED", FLAG, "0", why="task 406")
    authority: bool = knob("AW_AUTHORITY", FLAG, "0", why="ADR-019")
    authority_w: float = knob("AW_AUTHORITY_W", FLOAT, "0.0", why="ADR-019")
    authority_maxboost: float = knob("AW_AUTHORITY_MAXBOOST", FLOAT, "10.0", why="ADR-019")
    authority_min: float = knob("AW_AUTHORITY_MIN", FLOAT, "0.0", why="ADR-019")
    authority_relgate: float = knob("AW_AUTHORITY_RELGATE", FLOAT, "0.9", why="ADR-019")
    authority_min_centrality: float = knob("AW_AUTHORITY_MIN_CENTRALITY", FLOAT, "0.0", why="ADR-031")


@dataclass(frozen=True)
class Country:
    """Jurisdiction and country: off-country demotion, the serving-side country map, the
    country boost and gate."""
    off_country_w: float = knob("AW_OFF_COUNTRY_W", FLOAT, 0.0, weights="off_country", why="task 240")
    off_country_any: bool = knob("AW_OFF_COUNTRY_ANY", FLAG, "0", why="ADR-054")
    off_country_locbound_only: bool = knob("AW_OFF_COUNTRY_LOCBOUND_ONLY", FLAG, "0", why="ADR-054")
    off_country_nonbound_w: str = knob("AW_OFF_COUNTRY_NONBOUND_W", STR, "", why="ADR-054")
    off_country_nonbound_classes: str = knob("AW_OFF_COUNTRY_NONBOUND_CLASSES", STR, "ref,card,general",
                                             why="ADR-054")
    country_gate_beta: float = knob(None, FLOAT, 0.5, weights="country_gate_beta", why="task 164")


@dataclass(frozen=True)
class Output:
    """Result shaping after the ranking."""
    dedup_v2: bool = knob("AW_DEDUP_V2", FLAG, "0", why="task 229")


GROUPS = (("runtime", Runtime), ("query", Query), ("legs", Legs), ("fusion", Fusion),
          ("trust", Trust), ("country", Country), ("output", Output))


def specs():
    """(group, attr, metadata) for every declared field, in declaration order."""
    return [(g, f.name, f.metadata) for g, cls in GROUPS for f in dataclasses.fields(cls)]


ENV_NAMES = frozenset(m["env"] for _, _, m in specs() if m["env"])


def known_env_names() -> frozenset:
    """Names the process may see without reading them: server specs plus every name the package
    declares (`cytir.core.paths`). A build-side name such as AW_WORK_ROOT is read, not 'unread' (task 1792)."""
    from cytir.core import paths
    return ENV_NAMES | paths.env_names()

# S10 step 3 (task 1079): knobs whose only setters were deleted (#968/#969) are fixed constants in the
# module that used them. The name is no longer read; `GET /config` lists it under `frozen` so an operator
# who still sets it sees why nothing happens. {env name: (fixed value, owning module constant)}.
FROZEN = {
    "AW_COUNTRY_BOOST": (0.6, "retrieval._COUNTRY_BOOST"),  # frozen-echo
    "AW_HUBFLOOR_DF_MIN": (0.0, "retrieval._HUBFLOOR_DF_MIN"),  # frozen-echo
    "AW_HUBFLOOR_MAXTRUST": (1.0, "retrieval._HUBFLOOR_MAXTRUST"),  # frozen-echo
    "AW_REQTERM_MODE": ("all", "retrieval._REQTERM_MODE"),  # frozen-echo
    "AW_REQTERM_DFMAX": (0, "retrieval._REQTERM_DFMAX"),  # frozen-echo
    "AW_REQTERM_MAXQ": (0, "retrieval._REQTERM_MAXQ"),  # frozen-echo
}
_BY_ENV = {m["env"]: (g, a, m) for g, a, m in specs() if m["env"]}


# --------------------------------------------------------------------------- #
#  The fusion weights file. Moved from `retrieval` (it re-exports all three): the 12 fields above
#  whose default comes from the file need it before `retrieval` runs.
# --------------------------------------------------------------------------- #
_DEFAULT_WEIGHTS = {"dense": 1.0, "lexical": 0.8, "bm25": 0.8, "trust": 0.3,
                    "recency": 0.1, "exact": 0.0, "academic": 0.0, "machine": 0.0,
                    "title": 0.0, "canton_scope": 0.0, "foreign_gov": 0.0,
                    "parasite": 0.0, "official_pref": 0.0, "legifrance_bulk": 0.0,
                    "off_country": 0.0, "commercial_ref": 0.0,
                    "commercial_gov_scale": 1.0}


def _weights_path() -> str:
    return current("AW_FUSION_WEIGHTS")


def _read_weights_file(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 -- the old loader's fallback: code defaults (spec R5)
        return None


def _load_weights() -> dict:
    raw = _read_weights_file(_weights_path())
    return {**_DEFAULT_WEIGHTS, **raw} if raw is not None else dict(_DEFAULT_WEIGHTS)


def _coerce(kind: str, raw):
    if kind == FLAG:
        return raw == "1"
    if kind == TRUTHY:
        return _truthy(raw)
    if kind == FLOAT:
        return float(raw)
    if kind == INT:
        return int(raw)
    return raw


def _parse(meta: Mapping, env: Mapping, weights: Mapping, file_keys) -> tuple[Any, str]:
    """(value, source) of one field, exactly as the old read computed it."""
    name, kind, default, wkey = meta["env"], meta["kind"], meta["default"], meta["weights"]
    fallback = weights.get(wkey, default) if wkey else default
    fsrc = f"weights:{wkey}" if wkey and wkey in file_keys else "default"
    if kind == STR_OR_WEIGHTS:
        v = env.get(name)
        return ((v, "env") if v else (fallback, fsrc))
    if name is not None and name in env:
        return _coerce(kind, env[name]), "env"
    return _coerce(kind, fallback), fsrc


def current(name: str, env: Mapping | None = None):
    """A `call_time` field read now (today's semantics for the Qdrant URL and the weights path)."""
    g, a, meta = _BY_ENV[name]
    env = _envnames.mirrored(os.environ if env is None else env)
    return _parse(meta, env, {}, ())[0]


def _warnings(env: Mapping) -> list[str]:
    out = []
    for name, (_, _, m) in sorted(_BY_ENV.items()):
        if name in env and m["kind"] == FLAG and env[name] not in ("0", "1"):
            out.append(f"{name}={env[name]!r} reads as OFF: only '1' turns a flag on")
    return out


@dataclass(frozen=True)
class Settings:
    runtime: Runtime
    query: Query
    legs: Legs
    fusion: Fusion
    trust: Trust
    country: Country
    output: Output
    sources: Mapping[str, str]
    weights: dict
    weights_file_loaded: bool
    unknown_env: tuple
    parse_warnings: tuple
    serving_release: str | None = None    # FROZEN at startup, like every other field

    @classmethod
    def load(cls, env: Mapping | None = None) -> "Settings":
        given = os.environ if env is None else env
        env = _envnames.mirrored(given)
        raw = _read_weights_file(current("AW_FUSION_WEIGHTS", env))
        weights = {**_DEFAULT_WEIGHTS, **raw} if raw is not None else dict(_DEFAULT_WEIGHTS)
        file_keys = frozenset(raw or ())
        groups, sources = {}, {}
        for g, klass in GROUPS:
            kw = {}
            for f in dataclasses.fields(klass):
                kw[f.name], sources[f"{g}.{f.name}"] = _parse(f.metadata, env, weights, file_keys)
            groups[g] = klass(**kw)
        known = known_env_names()
        # each unknown knob ONCE, in the spelling the user set (a mirror-made twin is not listed)
        unknown = tuple(k for k in _envnames.user_set_names(given)
                        if k not in known and _envnames.twin(k) not in known)
        return cls(**groups, sources=sources, weights=weights, weights_file_loaded=raw is not None,
                   unknown_env=unknown, parse_warnings=tuple(_warnings(env)),
                   serving_release=_read_serving_release())


SETTINGS = Settings.load()
if SETTINGS.unknown_env:
    _LOG.warning("aw_server.settings: AW_* names set in the environment that no setting reads "
                 "(ignored): %s", ", ".join(SETTINGS.unknown_env))


# --------------------------------------------------------------------------- #
#  The echoes. `/health`'s `config` block (gates diff it) is `effective_config`, byte-identical to
#  the pre-split one; `GET /config` is `config_document`: every field, its source, the artifacts.
# --------------------------------------------------------------------------- #
def effective_config(s: Settings | None = None, *, rankhead_v2_loaded: bool,
                     fusion_weights: dict) -> dict:
    """The load-bearing knobs `/health` echoes (deployment-doctrine "No Invisible State"; tasks
    519, 578, 582, 606, 665: an arm that cannot echo its configuration cannot be compared). The
    caller supplies the two runtime facts: whether the v2 head loaded, and the weights file read
    fresh (`retrieval.weights_fingerprint`)."""
    s = s or SETTINGS
    f = s.fusion
    return {
        "collection": s.runtime.collection,
        "mrl_dim": s.runtime.mrl_dim,
        "dense_rescore": s.legs.dense_rescore,
        "dense_oversampling": s.legs.dense_oversampling,
        "dense_ignore_quant": s.legs.dense_ignore_quant,
        "rankhead": s.query.rankhead,
        # task 578: `requested` vs `loaded` -- a head the serving contract refuses loads nothing.
        "rankhead_v2": {"requested": s.query.rankhead_v2,
                        "loaded": rankhead_v2_loaded,
                        "path": s.runtime.rankhead_v2_path},
        "primacy": {"on": f.primacy, "keep": f.primacy_keep,
                    "beta": f.primacy_beta, "mincent": f.primacy_mincent,
                    "extra_classes": sorted(class_set(f.primacy_extra_classes))},
        "pagecap": {"on": bool(s.legs.pagecap), "cap": s.legs.pagecap,
                    "mode": s.legs.pagecap_mode, "oversample": s.legs.pagecap_oversample},
        "refcanon": {"on": f.refcanon, "w": f.refcanon_w,
                     "beta": f.refcanon_beta, "maxbeta": f.refcanon_maxbeta,
                     "mincent": f.refcanon_mincent,
                     "targets": sorted(class_set(f.refcanon_targets)),
                     "extra_classes": sorted(class_set(f.refcanon_extra_classes))},
        "bm25_zero_guard": s.legs.bm25_zero_guard,
        "version_match": {"on": s.query.version_match, "w": max(0.0, s.query.version_match_w)},
        "fusion_weights": fusion_weights,
    }


def redact(v):
    """URL userinfo and query strings never leave the process (there are no secrets among the
    fields today; a test asserts it, and this keeps a future URL with a token out of `/config`)."""
    if isinstance(v, str) and "://" in v:
        scheme, rest = v.split("://", 1)
        host, sep, path = rest.partition("/")
        host = host.rsplit("@", 1)[-1]
        path = path.split("?", 1)[0].split("#", 1)[0]
        if not sep:
            host = host.split("?", 1)[0].split("#", 1)[0]
        return f"{scheme}://{host}{sep}{path}"
    return v


def present(x) -> bool:
    """A loaded table or head counts as loaded when it is not None and not empty."""
    return x is not None and (not hasattr(x, "__len__") or len(x) > 0)


def _file_sha(path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001 -- reported as None: not readable
        return None


def config_document(s: Settings | None = None, *, artifacts: Mapping[str, tuple] = (),
                    env: Mapping | None = None) -> dict:
    """`GET /config` (schema `aw.serve-config.v1`): every field with `value`, `source`, `env`,
    `owner`; each artifact with path, sha256 and loaded-or-not; `unknown_env`; `parse_warnings`.
    `artifacts` = {name: (path, loaded)}."""
    s = s or SETTINGS
    env = _envnames.mirrored(os.environ if env is None else env)
    out = {}
    for g, a, m in specs():
        value = getattr(getattr(s, g), a)
        src = s.sources[f"{g}.{a}"]
        if m["call_time"]:
            value, src = current(m["env"], env), ("env" if m["env"] in env else "default")
        out[f"{g}.{a}"] = {"value": redact(value), "source": src, "env": m["env"],
                           "kind": m["kind"], "owner": m["owner"], "why": m["why"]}
    arts = {n: {"path": redact(p), "sha256": _file_sha(p) if p else None, "loaded": bool(ok)}
            for n, (p, ok) in dict(artifacts).items()}
    arts.setdefault("fusion_weights", {"path": _weights_path(), "sha256": _file_sha(_weights_path()),
                                       "loaded": s.weights_file_loaded})
    return {"schema": SCHEMA, "fields": out, "artifacts": arts,
            "tree": {"sha": s.serving_release, "file": os.path.basename(SERVING_RELEASE_FILE)},
            "unknown_env": list(s.unknown_env), "parse_warnings": list(s.parse_warnings),
            "frozen": [{"env": n, "value": v, "constant": c, "set_in_env": n in env,
                        "note": "fixed constant since S10 step 3 (task 1079); the env var is ignored"}
                       for n, (v, c) in sorted(FROZEN.items())]}
