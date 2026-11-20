"""S10 slice 1 (task 752) and step 1 (task 1061): the pure helper modules split out of `retrieval.py`.

Two contracts a later slice must not break quietly:

1. THE FACADE. Every moved name is still reachable as `retrieval.<name>` and is the SAME object
   as in its new module. Gates, trainers and tests read `retrieval._minmax`,
   `retrieval._detect_identifier`, ... (87 files import `retrieval`); a moved name that the
   facade dropped would break them, and a re-defined copy would fork the contract.
2. PURITY. `signals`, `query`, `levers` and `textmatch` read no environment, open no client and hold no
   state (`query` and `levers` read frozen constants from `settings` at import, as `retrieval` did): what they compute is a function of their arguments. Each imports only the standard
   library and the server modules named in `SERVER_IMPORTS` (`schema` is the tokenizer contract).
   The proof that the move did not change a ranking is the parity gate on the twin
   (`cytir.gates.serve_parity`), not this test; this test keeps the modules pure afterwards.
"""
from __future__ import annotations

import ast
import pathlib
from cytir.serve import SERVER_DIR
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cytir.serve import (fusion, geo, heads, legs, levers, operators, plan, qdrant_io, query, respond, retrieval,  # noqa: E402
                       signals, tables, textmatch, trace)

MOVED = {
    signals: ["_trust_tier", "_ACADEMIC_DOMAINS", "_domain", "_academic_penalty",
              "_MACHINE_URL_RE", "_machine_url", "_LF_HOST_RE", "_LF_BULK_RE", "_lf_bulk",
              "_exact", "_minmax", "_recency", "_registrable_domain"],
    query: ["_AC_WORDS", "_AC_PHRASES", "_AC_PAT", "_query_academic_intent",
            "_ID_SCHEME_SUFFIX", "_ID_UID_RE", "_ID_LEI_RE", "_ID_SIRET_RE", "_ID_SIREN_RE",
            "_ID_VAT_RE", "_detect_identifier",
            "_COMM_PHRASES", "_COMM_TOKENS", "_COMM_SUBSTR", "_COMM_LEGAL_GUARD",
            "_query_commercial_intent",
            # step 3 (task 1083)
            "_NAMEGATE_MAXTERMS", "_NAMEGATE_SHAPE_V2", "_NOBILIARY", "_person_name_tokens",
            "_person_name_query", "_name_token_primaries", "_required_word_groups",
            "_card_alias_in_query"],
    levers: ["_NAMEGATE_W", "_REFCANON", "_REFCANON_W", "_REFCANON_BETA",
             "_REFCANON_MAXBETA", "_REFCANON_EXTRA", "_REFCANON_TARGETS", "_REFCANON_MINCENT",
             "_name_token_coverage", "_sparse_name_coverage", "_apply_namegate",
             "_refcanon_canonical", "_apply_refcanon",
             # step 7 (task 1111)
             "_card_key", "_card_winners",
             # step 8b (task 1125): stage 2 + levers
             "_VERSION_MATCH", "_VERSION_MATCH_W", "_TRUST_GATE_PCTL", "_OFFICIAL_PREF_BETA",
             "_LFBULK_RELGATE", "_ENTITY_CARD_BETA", "_ENTITY_CARD_NAME_THR", "_ENTITY_CARD_CAP",
             "_CANTON_UNCONDITIONAL", "_OFF_COUNTRY_W", "_COMMERCIAL_REF_W", "_HUBFLOOR_W",
             "_HUBFLOOR_DF_MIN", "_HUBFLOOR_DISTINCT_MAX", "_HUBFLOOR_MAXTRUST",
             "_HUBFLOOR_DENSE_MAX", "_TRUSTFLOOR_ABS", "_TRUSTFLOOR_ABS_MAXTOK", "_NAMEGATE",
             "_PRIMACY", "_PRIMACY_BETA", "_PRIMACY_MINCENT", "_NAMEDENSE", "_NAMEDENSE_SCALE",
             "_NAMECOV_W", "_TRUSTFLOOR_UNDEMOTED", "_NAME_LEVERS_ARMED", "_NAMECOV_SPARSE_CLASSES",
             "_REQTERM_W", "_REQTERM_MODE", "_REQTERM_DFMAX", "_REQTERM_MAXQ", "_REQTERM_MINGROUPS",
             "_AUTHORITY_W", "_AUTHORITY_MAXBOOST", "_AUTHORITY_MIN", "_AUTHORITY_RELGATE",
             "_AUTHORITY_MIN_CENTRALITY"],
    fusion: ["_TRUSTFLOOR_TAU", "_TRUSTFLOOR_FLOOR", "_PRIMACY_KEEP", "_PRIMACY_EXTRA",
             "_trustfloor_exempt",
             "_trustfloor_scale", "_is_primary_class", "_primacy_refund",
             "_url_tie_len", "_order_ids", "_auth_admit", "_extend_pool", "_order_scored",
             # step 8a (task 1116)
             "_FUSION_DENSE_W", "_FUSION_BM25_W", "_W_DENSE_EFF", "_W_BM25_EFF", "_COUNTRY_BOOST",
             "_COUNTRY_GATE", "_COUNTRY_GATE_BETA",
             # step 8b (task 1125)
             "_TRUST_MODE", "_TRUST_ALPHA", "_COMMERCIAL_GOV_SCALE", "_GOV_SOURCE_TYPES",
             "_RELFLOOR_BETA", "_HUBGATE", "_HUBGATE_SCALE", "_HUBGATE_MINBETA",
             "_HUBGATE_MAXTRUST", "_TRUST_RELGATE", "_TRUST_RELGATE_BETA", "_TRUSTFLOOR",
             "_TRUSTFLOOR_MINBETA", "_TRUSTFLOOR_POOL_MAX", "_TRUSTFLOOR_REL_BETA",
             "_TRUSTFLOOR_MAXTRUST", "_TRUSTFLOOR_ABS_DENSE", "_NAMECOV_SURFACES",
             "_NAMECOV_SURFACE", "_NAMEGATE_TRUSTKEEP", "_RELFLOOR", "_NAMEGATE_COV"],
    # step 4 (task 1095)
    geo: ["_CANTON_NAMES", "_COUNTRY_MARKERS", "_SCOPE_COUNTRIES", "_EXEMPT_STAMPS",
          "_OFF_COUNTRY_ANY", "_demotable_scope", "_EU_MEMBERS", "_is_eu_supra", "_terms_re",
          "_CANTON_RES", "_COUNTRY_RES", "_query_jurisdiction", "_GAZ_PATH", "_load_gazetteer",
          "_GAZETTEER", "_gazetteer_scope", "_scope_penalties", "_OFF_COUNTRY_LOCBOUND_ONLY",
          "_country_is_binding", "_off_country_hit", "_COUNTRY_MAP_PATH", "_COUNTRY_MAP",
          "_effective_country", "_OFF_COUNTRY_NONBOUND_W_RAW", "_OFF_COUNTRY_NONBOUND_W",
          "_OFF_COUNTRY_NONBOUND_CLASSES", "_nonbound_class", "_learned_scope_conflict_penalty"],
    # step 5 (task 1099)
    heads: ["_QHEAD_COUNTRY_LEGS", "_RANKHEAD", "_RANKHEAD_PATH", "_RANKHEAD_V2",
            "_RANKHEAD_V2_PATH", "_RANKHEAD_V2_DUMP", "_LAST_V2_DUMP", "_load_rankhead",
            "_RANKHEAD_MODEL", "_load_rankhead_v2", "_RANKHEAD_V2_MODEL", "_head_beta", "_QHEAD",
            "_QHEAD_PATH", "_QHEAD_KNOB_ALIASES", "_QHEAD_KNOBS", "_SCOPE_ACTIVE",
            "_parse_country_scope_mode", "_COUNTRY_SCOPE_MODE", "_LEARNED_ONLY_SCOPE",
            "_POLICY_ONLY_SCOPE", "_QHEAD_DRIVEN", "_load_qhead", "_QHEAD_MODEL", "_qhead_vector",
            "_SCOPE_KNOB", "_load_scope_head", "_SCOPE_HEAD", "_load_query_policy_heads",
            "_QUERY_POLICY_HEADS", "_query_policy_vector", "_scope_head_predict", "_resolve_scope",
            "_effective_free_text_country_scope", "_country_scope_for_ranking", "_head_artifacts"],
    tables: ["_load_parasites", "_PARASITE_HOSTS", "_PARASITE_RES", "_parasite",
             "weights_fingerprint", "_HUBFLOOR", "_HUBFLOOR_TABLE", "_load_hub_hosts",
             "_HUB_HOSTS", "_hubness", "_AUTHORITY", "_AUTHORITY_TABLE",
             "_load_authority_hosts", "_AUTH_HOSTS", "_authority"],
    # step 6 (task 1104)
    qdrant_io: ["indexed_fields", "_identifier_lookup", "client", "health_client", "_leg"],
    plan: ["QueryPlan", "_country_query_plan", "_country_partition_filter", "candidate_pool",
           "fuse_margin"],
    respond: ["_is_tracking_param", "_canonical_key", "_RELFLOOR_ZS_K"],
    trace: ["_dbgc", "_LAST_COMPONENTS", "_QHEAD_COUNTRY_TRACE", "_LAST_QUERY_PLAN_TRACE",
            "_LAST_EXPERIMENT_TRACE", "_DEBUG_SCORES", "_LAST_SCORE_DEBUG"],
    # step 8a (task 1116): stage-1 constants travel with their readers
    legs: ["_CARD_LANG_EXEMPT", "_DENSE_HNSW_EF", "_DENSE_RESCORE", "_DENSE_OVERSAMPLING",
           "_DENSE_IGNORE_QUANT", "_DENSE_SP", "_TWO_STAGE", "_PAGECAP", "_PAGECAP_MODE",
           "_PAGECAP_OVERSAMPLE", "_AUTHLEG", "_AUTHLEG_THRESH", "_AUTHLEG_N", "_AUTHLEG_BETA",
           "_BATCH_LEGS", "_AUTH_LEG_KEY", "_RELFLOOR_DENSE_MAX", "_PER_TOKEN",
           "_PER_TOKEN_LIMIT", "_PER_TOKEN_MAXQ", "_PER_TOKEN_SUB_W", "_PER_TOKEN_K1",
           "_BM25_ZERO_GUARD", "_SPARSE_SCORE_LEGS", "stage1", "Stage1Request", "Stage1"],
    textmatch: ["_WORD", "_ALPHA_RE", "_STOPWORDS", "_content_terms", "_stem_match",
                "_title_centrality", "_card_aliases", "_card_name_match", "_query_terms",
                "_title_versions_match", "_title_centrality_v", "_exact_v"],
}

#: The server modules each split module may import (relative imports, by module name).
SERVER_IMPORTS = {signals: set(), query: {"textmatch", "schema", "settings"},
                  textmatch: {"schema", "signals"},
                  levers: {"article_tokens", "fusion", "query", "settings", "signals", "textmatch"},
                  fusion: {"settings"}}
#: The one package import a split module may carry: levers uses the core location-bound test.
CORE_IMPORTS = {levers: {"cytir.core.country"}, fusion: {"cytir.core.country"}}


#: geo and tables load files at import (gazetteer, country map, hub / authority / parasite
#: tables, fail-safe to empty), so they get std-lib file and json access but nothing else.
#: heads loads the learned-head artifacts at import (fail-safe to None) and does the matmuls (numpy).
LOADERS = {geo: {"cytir.core.country"}, tables: set(), heads: set()}
LOADER_STDLIB = {"__future__", "hashlib", "json", "logging", "pathlib", "re"}
LOADER_EXTRA_STDLIB = {heads: {"math", "numpy"}}
LOADER_SERVER = {geo: {"settings", "signals"}, tables: {"geo", "settings", "signals"},
                 heads: {"geo", "rankhead_schema", "rankhead_v2", "settings"}}


#: step 6: the I/O seam holds state (`global`), plan imports qdrant_client models, respond/trace are std-lib
#: plus settings. Their own contract is `test_the_step6_modules_...` below.
STEP6 = {qdrant_io, plan, respond, trace, legs, fusion, levers, operators}


#: step 8b (task 1125): the stage functions `search()` calls (new names, no facade).
STAGE_FUNCTIONS = {levers: ["coverage_probe", "build_rows", "apply_levers", "_reciprocal_competition_rank"],
                   fusion: ["score_rows"], respond: ["finish"], plan: ["qk"]}


def test_the_stage_functions_exist_and_take_keywords_only():
    import inspect
    for mod, names in STAGE_FUNCTIONS.items():
        for name in names:
            fn = getattr(mod, name)
            assert callable(fn), f"{mod.__name__}.{name}"
            if name in ("_reciprocal_competition_rank",):
                continue
            sig = inspect.signature(fn)
            kinds = [p.kind for n, p in sig.parameters.items() if not (name == "qk" and n in ("name", "default"))]
            assert all(k is inspect.Parameter.KEYWORD_ONLY for k in kinds), f"{mod.__name__}.{name}"


def test_every_moved_name_is_the_same_object_through_the_facade():
    for mod, names in MOVED.items():
        for name in names:
            assert getattr(retrieval, name) is getattr(mod, name), f"{mod.__name__}.{name}"


def test_no_moved_name_is_defined_twice():
    src = (SERVER_DIR / "retrieval.py").read_text()
    tree = ast.parse(src)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    twice = sorted(defined & {n for names in MOVED.values() for n in names})
    assert twice == [], f"defined in retrieval.py AND in a split module: {twice}"


def test_the_split_modules_are_pure():
    allowed = {"__future__", "datetime", "math", "re", "urllib.parse"}
    for mod in MOVED:
        if mod in LOADERS or mod in STEP6:
            continue          # geo and tables read tables off disk at import: their own test below
        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        imported, server = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level:
                server |= {n.module} if n.module else {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module or ".")
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                assert (n.value.id, n.attr) not in {("os", "environ"), ("os", "getenv")}
        assert imported <= allowed | CORE_IMPORTS.get(mod, set()), f"{mod.__name__} imports {sorted(imported - allowed)}"
        assert server <= SERVER_IMPORTS[mod], f"{mod.__name__} imports {sorted(server - SERVER_IMPORTS[mod])}"
        assert "global " not in pathlib.Path(mod.__file__).read_text()


def test_the_moved_helpers_still_answer_as_before():
    # a few anchors, so a careless edit in the new home is caught here and not only on the twin
    assert retrieval._detect_identifier("CHE-115.140.892 MWST") == (
        "register_id", "CHE-115.140.892")
    assert retrieval._query_academic_intent("transformer wikipedia") is False
    assert retrieval._query_academic_intent("resnet paper") is True
    assert retrieval._academic_penalty("https://arxiv.org/abs/1", False) == 1.0
    assert retrieval._machine_url("https://x.org/data.csv") == 1.0
    assert retrieval._lf_bulk("https://www.legifrance.gouv.fr/jorf/id/X") == 1.0
    assert retrieval._minmax({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}
    assert retrieval._minmax({"a": 0.0, "b": 2.0}, drop_nonpositive=True) == {"b": 1.0}
    assert retrieval._trust_tier({"trust_tier": None}) == 0.6
    assert retrieval._exact({"tim", "ohear"}, {"title": "Tim O'Hear"}) == 1.0
    assert retrieval._recency(None) == 0.0
    # step 1 (task 1061)
    assert retrieval._registrable_domain("https://www.roche.com/a") == "roche.com"
    assert retrieval._query_commercial_intent("Barkredit Zürich") is True
    assert retrieval._query_commercial_intent("Konsumkreditgesetz Art. 3") is False
    qc = retrieval._content_terms("machine learning neural networks", "en")
    assert qc == {"machine", "learning", "neural", "networks"}
    assert retrieval._title_centrality(qc, "Neural network") == 0.5
    # step 3 (task 1083)
    qn = retrieval._content_terms("Tim O'Hear", None)
    assert retrieval._person_name_tokens("Tim O'Hear", qn) == {"tim", "ohear"}
    assert retrieval._person_name_query("Tim O'Hear", qn) is True
    assert retrieval._refcanon_canonical("gov-law") is True
    assert retrieval._refcanon_canonical("blog") is False
    assert retrieval._card_name_match({"roche"}, "F. Hoffmann-La Roche AG", "Roche") == 1.0
    assert retrieval._query_terms("Tim O'Hear") == {"tim", "ohear"}
    # step 4 (task 1095)
    assert retrieval._query_jurisdiction("Mietrecht Kanton Zürich") == (frozenset({"ZH"}), "CH")
    assert retrieval._query_jurisdiction("TVA France") == (frozenset(), "FR")
    assert retrieval._is_eu_supra("https://eur-lex.europa.eu/x") is True
    assert retrieval._demotable_scope("FR") is True and retrieval._demotable_scope("INTL") is False
    assert retrieval._off_country_hit("CH", "FR", "blog") == 1.0
    assert retrieval._off_country_hit("FR", "DE", "official") == 0.0
    assert retrieval._country_is_binding("gov-law") is True
    assert retrieval._country_is_binding("entity_card", has_register=True) is True
    assert retrieval._country_is_binding("reference") is False
    assert retrieval._scope_penalties(frozenset(), "CH", "ZH", "FR", "official") == (1.0, 1.0)
    assert retrieval._nonbound_class("entity_card") == "card"
    assert retrieval._effective_country({"country": "CH"}) == "CH"
    assert set(tables.weights_fingerprint()) == {"path", "sha256"}
    assert retrieval._parasite("") == 0.0
    # step 5 (task 1099)
    assert retrieval._parse_country_scope_mode(None) == "legacy"
    assert retrieval._parse_country_scope_mode(" Policy_Only ") == "policy_only"
    assert retrieval._head_beta(None) == 0.0 and retrieval._qhead_vector(None) == {}
    assert retrieval._query_policy_vector(None) == {}
    assert retrieval._scope_head_predict(None) == (None, 0.0)
    assert retrieval._country_scope_for_ranking("FR", "q", "CH", None) == ("FR", None, 1.0)
    assert retrieval._effective_free_text_country_scope("q", "CH", None) == ("CH", None, 1.0)


def test_geo_tables_and_heads_import_no_retrieval_and_read_no_env():
    for mod, core in LOADERS.items():
        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        imported, server = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level:
                server |= {n.module} if n.module else {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module or ".")
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                assert (n.value.id, n.attr) not in {("os", "environ"), ("os", "getenv")}
        assert imported <= LOADER_STDLIB | LOADER_EXTRA_STDLIB.get(mod, set()) | core, \
            f"{mod.__name__} imports {sorted(imported)}"
        assert server <= LOADER_SERVER[mod], f"{mod.__name__} imports {sorted(server)}"
        assert "retrieval" not in server and not any(isinstance(n, ast.Global) for n in ast.walk(tree))


def test_location_bound_is_the_core_one_not_a_server_copy():
    from cytir.core import country
    assert not hasattr(retrieval, "_is_location_bound")
    assert not hasattr(retrieval, "_LOCATION_BOUND_TYPES")
    assert retrieval.is_location_bound is country.is_location_bound
    assert geo.is_location_bound is country.is_location_bound


def test_the_head_artifacts_splice_into_the_config_listing():
    listing = tables.loaded_artifacts(retrieval._head_artifacts())
    assert list(listing) == ["fusion_weights", "rankhead", "rankhead_v2", "qhead", "scope_head",
                             "query_policy_heads", "hub_hosts", "authority_hosts",
                             "parasite_domains", "gazetteer", "country_map"]


def test_a_bad_country_scope_mode_is_refused():
    import pytest
    with pytest.raises(ValueError):
        heads._parse_country_scope_mode("keyword_only")


def test_resolve_scope_precedence(monkeypatch):
    # keyword > gazetteer > learned head; the learned-only arm consults only the head
    monkeypatch.setattr(heads, "_scope_head_predict", lambda qfull: ("FR", 0.8))
    assert heads._resolve_scope("q", "CH", None) == ("CH", None, 1.0)
    assert heads._resolve_scope("q", None, None, learned_only=True) == (None, "FR", 0.8)
    monkeypatch.setattr(heads, "_gazetteer_scope", lambda q: (None, True))
    assert heads._resolve_scope("q", None, None) == (None, None, 1.0)      # homograph abstains
    monkeypatch.setattr(heads, "_gazetteer_scope", lambda q: ("DE", False))
    assert heads._resolve_scope("q", None, None) == ("DE", None, 1.0)
    monkeypatch.setattr(heads, "_gazetteer_scope", lambda q: (None, False))
    assert heads._resolve_scope("q", None, None) == (None, "FR", 0.8)


def test_effective_country_map_fallback(monkeypatch):
    # task 1099 (review 1097 gap): the AW_COUNTRY_MAP branch, not only the stamp branch
    monkeypatch.setattr(geo, "_COUNTRY_MAP", {"admin.ch": "CH", "gouv.fr": "FR"})
    assert geo._effective_country({"country": "DE", "domain": "admin.ch"}) == "DE"      # stamp wins
    assert geo._effective_country({"domain": "admin.ch"}) == "CH"
    assert geo._effective_country({"domain": "WWW.Gouv.FR"}) == "FR"                    # www. + case
    assert geo._effective_country({"url": "https://www.admin.ch/x"}) == "CH"           # from the url
    assert geo._effective_country({"domain": "example.org"}) is None                   # unmapped
    assert geo._effective_country({"country": "", "domain": "example.org"}) == ""      # `or c` keeps ""
    monkeypatch.setattr(geo, "_COUNTRY_MAP", {})
    assert geo._effective_country({"domain": "admin.ch"}) is None                      # map unset


def _imports(mod):
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    ext, server, attrs = set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            ext |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level:
            server |= {n.module} if n.module else {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            ext.add(n.module or ".")
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            attrs.add((n.value.id, n.attr))
    return tree, ext, server, attrs


def test_the_step6_modules_import_no_retrieval_and_read_no_env():
    allowed = {
        qdrant_io: ({"__future__", "qdrant_client", "cytir.core.qdrant"}, {"respond", "schema", "settings"}),
        plan: ({"__future__", "math", "dataclasses", "qdrant_client"}, {"heads", "settings"}),
        respond: ({"__future__", "urllib.parse"}, {"fusion", "settings", "trace"}),
        trace: ({"__future__"}, {"settings"}),
        # step 8a: stage 1 reads the client only through the seam, the typed settings, and the pure
        # layers below it; never `retrieval`.
        # task 2665: + logging, the authority-leg thread pool and the client's error types.
        legs: ({"__future__", "functools", "time", "dataclasses", "typing", "qdrant_client", "logging",
                "concurrent.futures", "contextvars", "qdrant_client.http.exceptions"},
               {"article_tokens",  # task 2778: pure query-token normaliser
                "fused_split_terms",  # task 2856: digit-first fused/split query variants
                "qdrant_io", "fusion", "geo", "heads", "operators", "plan", "respond", "schema", "settings",
                "signals", "trace"}),
        # step 8b: stage 2 + levers + respond; `levers` is the top of the stack and the only caller of
        # the seam from stage 2 (the coverage probe).
        # task 2640: the operator host table is one Qdrant facet read through the seam, cached here.
        operators: ({"__future__", "threading", "time", "qdrant_client"}, {"qdrant_io", "query", "schema"}),
        fusion: ({"__future__", "math", "cytir.core.country"}, {"geo", "settings", "signals", "trace"}),
        levers: ({"__future__", "cytir.core.country", "qdrant_client"},
                 {"article_tokens",  # task 2792
                  "fusion", "geo", "heads", "legs", "models", "qdrant_io", "query", "rankhead_v2", "schema",
                  "settings", "signals", "tables", "textmatch", "trace"}),
    }
    for mod, (ext, server) in allowed.items():
        _, got_ext, got_server, attrs = _imports(mod)
        assert got_ext <= ext, f"{mod.__name__} imports {sorted(got_ext - ext)}"
        assert got_server <= server, f"{mod.__name__} imports {sorted(got_server - server)}"
        assert "retrieval" not in got_server
        assert ("os", "environ") not in attrs and ("os", "getenv") not in attrs


def test_the_qdrant_client_is_built_only_through_aw_core_qdrant():
    src = pathlib.Path(qdrant_io.__file__).read_text()
    assert "QdrantClient(" not in src and "from cytir.core.qdrant import qdrant_client" in src
    assert "QdrantClient(" not in pathlib.Path(retrieval.__file__).read_text()


def test_the_seam_builds_both_clients_with_the_typed_url_and_timeouts(monkeypatch):
    import qdrant_client as qc
    built = []
    monkeypatch.setattr(qc, "QdrantClient", lambda *a, **kw: built.append((a, kw)) or object())
    monkeypatch.setattr(qdrant_io, "_client", None)
    monkeypatch.setattr(qdrant_io, "_health_client", None)
    url = qdrant_io._settings.current("AW_QDRANT_URL")
    qdrant_io.client(); qdrant_io.client(); qdrant_io.health_client()
    assert built == [((url,), {"timeout": qdrant_io._QDRANT_TIMEOUT}),
                     ((url,), {"timeout": qdrant_io._QDRANT_HEALTH_TIMEOUT})]


def test_nothing_assigns_the_client_through_the_retrieval_facade():
    """R2/ratchet: a write to `retrieval.client` (or `_client`, `_indexed_fields`, `_QDRANT_TIMEOUT`)
    reaches nothing once the reader moved to `qdrant_io`. Patch `qdrant_io.<name>`."""
    import re
    root = HERE.parent
    names = r"(client|health_client|_client|_health_client|_indexed_fields|_QDRANT_TIMEOUT|_QDRANT_HEALTH_TIMEOUT)"
    pat = re.compile(r"\bretrieval\.%s\s*=(?!=)|(?:setattr|setitem)\(\s*retrieval\s*,\s*[\"']%s[\"']" % (names, names))
    bad = []
    for d in ("server", "tests", "pipeline", "cytir", "scripts"):
        for f in (root / d).rglob("*.py"):
            if f.name == pathlib.Path(__file__).name:
                continue
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if pat.search(line):
                    bad.append(f"{f.relative_to(root)}:{i}")
    assert bad == [], bad


def test_the_authority_minmax_is_the_signals_one_with_its_flat_value():
    """D7 (task 1104): authority.py's nested `_minmax` differed from `signals._minmax` only when every
    score is equal (0.0 there, 1.0 in signals). One function now; the authority leg passes flat=0.0."""
    import importlib       # fresh lookup: other test files purge and re-import `cytir.serve.*`
    authority = importlib.import_module("cytir.serve.authority")
    signals = importlib.import_module("cytir.serve.signals")
    assert authority._minmax is signals._minmax
    assert "def _minmax" not in pathlib.Path(authority.__file__).read_text()
    s = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert signals._minmax(s, flat=0.0) == signals._minmax(s) == {"a": 1.0, "b": 0.0, "c": 0.5}
    assert signals._minmax({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert signals._minmax({"a": 2.0, "b": 2.0}, flat=0.0) == {"a": 0.0, "b": 0.0}
    assert signals._minmax({}, flat=0.0) == {}


def test_a_scheme_less_qdrant_url_is_refused_not_opened_as_a_local_store(monkeypatch):
    import pytest
    built = []
    monkeypatch.setattr(qdrant_io, "qdrant_client", lambda *a, **kw: built.append(a) or object())
    monkeypatch.setattr(qdrant_io, "_client", None)
    monkeypatch.setattr(qdrant_io, "_health_client", None)
    for bad in ("qdrant:6333", "/tmp/store", ":memory:"):
        monkeypatch.setattr(qdrant_io._settings, "current", lambda name, _b=bad: _b)
        for fn in (qdrant_io.client, qdrant_io.health_client):
            with pytest.raises(ValueError, match="CYTIR_QDRANT_URL"):
                fn()
    assert built == []
    monkeypatch.setattr(qdrant_io._settings, "current", lambda name: "http://x:6333")
    qdrant_io.client(); qdrant_io.health_client()
    assert built == [("http://x:6333",), ("http://x:6333",)]


def test_fusion_helpers_answer_as_before_and_import_no_retrieval():
    # step 7 (task 1111): anchors for the trust-floor decision, the refund and the tie order
    ex = fusion._trustfloor_exempt
    kw = dict(dense_raw=0.0, abs_dense_floor=0.45, abs_term_hit=False)
    assert ex(0.9, 1.0, 0.8, abs_on=False, **kw) is True       # pre-ADR-099: sufficient
    assert ex(0.9, 1.0, 0.8, abs_on=True, **kw) is False        # ADR-099: needs absolute evidence
    assert ex(0.9, 1.0, 0.8, abs_on=True, **{**kw, "abs_term_hit": True}) is True
    assert ex(0.1, 1.0, 0.8, abs_on=False, **kw) is False
    assert fusion._primacy_refund(0.5) >= 0.5
    assert fusion._order_ids({"b", "a"}, {"a": 1, "b": 1}.get) == ["a", "b"]
    assert fusion._url_tie_len("https://x.test/a") == len("x.test/a")
    src = pathlib.Path(fusion.__file__).read_text()
    assert "retrieval" not in {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert "os.environ" not in src
